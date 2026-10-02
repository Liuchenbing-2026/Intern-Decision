"""CPU contracts and optional real NPU numerical checks for decision prefill."""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.inference.config import InferenceConfig
from src.inference.hf_backend import HFBackend
from src.inference.npu_backend import NPUBackend, NPUChunkRule


class NPUContractTests(unittest.TestCase):
    def test_batch_padding_preserves_request_and_field_order(self):
        backend = object.__new__(HFBackend)
        backend.device = torch.device("cpu")
        backend.tokenizer = SimpleNamespace(pad_token_id=0)
        backend.synchronize = lambda: None
        encoded = [
            ("first", {"input_ids": torch.tensor([[10, 20]])}, torch.tensor([1, 0])),
            ("second", {"input_ids": torch.tensor([[30, 40, 50]])}, torch.tensor([2])),
        ]
        backend.encode = lambda row: encoded[row["index"]]

        def forward(input_ids, attention_mask, logits_to_keep, **kwargs):
            self.assertEqual(attention_mask.tolist(), [[1, 1, 0], [1, 1, 1]])
            return SimpleNamespace(logits=input_ids[:, logits_to_keep, None].float())

        backend.model = forward
        outputs = backend.score_batch([{"index": 0}, {"index": 1}])
        self.assertEqual([item[0] for item in outputs], ["first", "second"])
        self.assertEqual([item[2] for item in outputs], [2, 3])
        self.assertEqual(outputs[0][1].flatten().tolist(), [20, 10])
        self.assertEqual(outputs[1][1].flatten().tolist(), [50])

    def test_device_timing_waits_for_forward_completion(self):
        events = []

        class Batch(dict):
            def to(self, device):
                events.append("transfer")
                return self

        backend = object.__new__(NPUBackend)
        backend.device = torch.device("cpu")
        backend.encode = lambda row: (None, Batch(input_ids=torch.tensor([[1, 2]])), torch.tensor([0]))

        def forward(**kwargs):
            events.append("forward")
            return SimpleNamespace(logits=torch.ones(1, 1, 3))

        backend.model = forward
        with patch.object(torch, "npu", SimpleNamespace(synchronize=lambda device: events.append("sync")), create=True):
            _, logits, length, elapsed = backend.score({})
        self.assertEqual(events, ["sync", "transfer", "forward", "sync"])
        self.assertEqual(length, 2)
        self.assertEqual(logits.shape, (1, 3))
        self.assertGreaterEqual(elapsed, 0)

    def test_config_accepts_explicit_npu_backend(self):
        config = InferenceConfig(backend="npu", checkpoint="model", device="npu:0")
        self.assertEqual(config.backend, "npu")

    def test_prefill_preserves_layout_and_starts_fresh_state(self):
        q = torch.randn(1, 5, 2, 128, dtype=torch.bfloat16)
        captured = []

        def fake_op(query, key, value, **kwargs):
            captured.append(kwargs)
            self.assertEqual(query.shape, (5, 2, 128))
            self.assertEqual(kwargs["actual_seq_lengths"].tolist(), [5])
            self.assertEqual(kwargs["initial_state"].count_nonzero().item(), 0)
            kwargs["initial_state"].fill_(1)
            return value, kwargs["initial_state"]

        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu_chunk_gated_delta_rule=fake_op)}):
            rule = NPUChunkRule()
            for _ in range(2):
                output, state = rule(q, q, q, torch.zeros(1, 5, 2), torch.ones(1, 5, 2))
                torch.testing.assert_close(output, q, rtol=0, atol=0)
                self.assertIsNone(state)
        self.assertEqual(len(captured), 2)

    def test_rejects_stateful_decoding_and_unsupported_precision(self):
        q = torch.zeros(1, 1, 1, 128)
        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}):
            for options in ({"initial_state": q}, {"output_final_state": True}, {}):
                with self.assertRaises(ValueError):
                    NPUChunkRule()(q, q, q, q[..., 0], q[..., 0], **options)


@unittest.skipUnless(importlib.util.find_spec("torch_npu"), "requires torch-npu and an Ascend NPU")
class NPUKernelTests(unittest.TestCase):
    def test_fused_rule_matches_reference_and_does_not_leak_state(self):
        import torch_npu  # noqa: F401
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        if not torch.npu.is_available():
            self.skipTest("No Ascend NPU available")
        torch.manual_seed(42)
        for batch, length in ((1, 1), (1, 63), (1, 64), (1, 65), (1, 257), (2, 65), (8, 65)):
            with self.subTest(batch=batch, length=length):
                q, k, v = [torch.randn(batch, length, 2, 128, device="npu", dtype=torch.bfloat16) for _ in range(3)]
                g = -torch.rand(batch, length, 2, device="npu")
                beta = torch.rand(batch, length, 2, device="npu", dtype=torch.bfloat16)
                expected, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                rule = NPUChunkRule()
                actual, _ = rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                again, _ = rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(actual, again, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
