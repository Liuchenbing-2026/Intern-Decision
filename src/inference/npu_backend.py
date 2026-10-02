"""Ascend decision inference using native torch-npu prefill operators."""

import torch

from src.inference.hf_backend import HFBackend


class NPUChunkRule:
    """Adapt HF's sequence-major, stateless prefill to the CANN GDN operator."""

    def __call__(
        self,
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        import torch_npu

        if initial_state is not None or output_final_state:
            raise ValueError("The NPU decision backend supports stateless prefill only")
        if query.dtype != torch.bfloat16 or query.shape[-1] != 128 or value.shape[-1] != 128:
            raise ValueError("CANN decision GDN requires BF16 and 128-dimensional heads")
        batch, length = query.shape[:2]
        if use_qk_l2norm_in_kernel:
            # Preserve HF's BF16 normalization order before the fused recurrence.
            from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

            query, key = l2norm(query), l2norm(key)
        state = torch.zeros(
            batch, value.shape[2], value.shape[-1], query.shape[-1], dtype=value.dtype, device=value.device
        )
        lengths = torch.full((batch,), length, dtype=torch.int32, device=query.device)
        output, _ = torch_npu.npu_chunk_gated_delta_rule(
            query.flatten(0, 1).contiguous(),
            key.flatten(0, 1).contiguous(),
            value.flatten(0, 1).contiguous(),
            beta=beta.flatten(0, 1).contiguous(),
            g=g.flatten(0, 1).float().contiguous(),
            initial_state=state,
            actual_seq_lengths=lengths,
            scale=query.shape[-1] ** -0.5,
        )
        return output.reshape(batch, length, value.shape[2], value.shape[-1]), None


class NPUBackend(HFBackend):
    """Reuse the official template, vision processor, weights and LM head."""

    def __init__(
        self,
        checkpoint,
        processor_path=None,
        media_root="",
        max_length=8192,
        device="npu:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        **kwargs,
    ):
        import torch_npu  # noqa: F401
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        if torch.device(device).type != "npu":
            raise ValueError("The npu backend requires device=npu or npu:<index>")
        if dtype != "bfloat16":
            raise ValueError("The npu backend requires dtype=bfloat16")
        if not torch.npu.is_available():
            raise RuntimeError("No Ascend NPU is available")
        torch.npu.set_device(device)
        super().__init__(
            checkpoint,
            processor_path,
            media_root,
            max_length,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self.gdn_layers = 0
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.chunk_gated_delta_rule = NPUChunkRule()
                self.gdn_layers += 1
        if not self.gdn_layers:
            raise ValueError("The checkpoint has no supported Qwen3.5 Gated DeltaNet layers")

    def synchronize(self):
        torch.npu.synchronize(self.device)
