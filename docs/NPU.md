# Ascend NPU inference

This fork provides a standalone `npu` backend. It uses Transformers for Qwen3.5 weights, vision processing, causal attention and the LM head, and torch-npu's `npu_chunk_gated_delta_rule` for Gated DeltaNet prefill. It preserves global Transformers classes.

The official CPU/GPU `hf` and `xtuner` backends remain available. NPU imports are lazy. The trained assistant skeleton, `<decision>` predecessor alignment, candidate ordering, tie-breaking, score/noul decoding and temperature handling all reuse the existing implementation. Requests are not truncated.

## Environment

The tested combination is Ascend 910B4-1, CANN 9.1.0, Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4 and Transformers 5.14.1. Install the compatible CANN/driver first, then create a dedicated virtual environment:

```bash
uv venv --python 3.12 --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements-npu.txt -r requirements-eval.txt
```

If the platform supplies matching torch/torch-npu wheels, use those rather than replacing them with a CUDA torch distribution. Model checkpoints must contain the trained decision tokenizer and vision configuration. All three official sizes use BF16 with 128-dimensional GDN heads; other dtypes and stateful decoding are explicitly rejected by this backend.

## Inference and service

Copy `configs/inference/npu.json` to a local config and set `checkpoint` to the complete downloaded checkpoint directory. The default device is `npu:0`; device numbering is relative to the devices exposed to the process.

```bash
export OMP_NUM_THREADS=4
.venv/bin/python -m src.inference --config /path/to/npu.json \
  --input /path/to/request.json --output outputs/answer.json
```

The JSON input is the same `state`, optional local `images`, and `questions` schema as the original backend. Start the existing service with `INFERENCE_CONFIG=/path/to/npu.json PYTHON_BIN=$PWD/.venv/bin/python bash scripts/demo.sh`. The default endpoint is `http://127.0.0.1:7860/v1/decisions`. This backend computes each request independently and has no generation KV cache or prefix cache.

## Accuracy and latency

Run the original evaluator and scoring rules without changing labels or datasets:

```bash
.venv/bin/python -m src.eval.verify_bundle
.venv/bin/python -m src.eval.jev --backend npu \
  --checkpoint /path/to/Intern-Decision-0.8B --suite --batch-size 8 \
  --output outputs/npu-0.8b-accuracy
.venv/bin/python -m src.eval.benchmark --backend npu \
  --checkpoint /path/to/Intern-Decision-0.8B \
  --data benchmarks/accuracy-v1/jevbench/easy.jsonl \
  --warmup 16 --repeats 96 --output outputs/npu-0.8b-latency.json
```

Repeat with the 2B and 4B checkpoints and fresh output paths. The complete accuracy suite contains 10,751 rows and 12,351 decisions across seven tasks. Its headline mean gives each task equal weight. Do not replace it with a row-weighted mean.

The benchmark defaults to batch=1; pass `--batch-size 8` to measure eight-request batches. It includes template construction, tokenization, the synchronized device forward and probability decoding; it excludes HTTP, startup and model loading. It records every sample; mean/median/P95 are batch durations and requests/second counts all requests in each batch. Accuracy output records its batch size. The per-row `inference_ms` in batched predictions is the shared forward duration of that batch, not a separate single-request latency. Single-request latency and batch throughput are separate metrics; keep input rows, warmup and sample counts consistent between runs. HF and the fused CANN recurrence can differ numerically in BF16, so report candidate probabilities as well as top-1 decisions.

## Tests and scope

```bash
.venv/bin/python -m pytest -q tests/test_npu_backend.py tests/check_inference.py \
  tests/check_image_uploads.py tests/test_release.py tests/test_evaluation_workflow.py
```

CPU tests cover configuration, fresh recurrent state, layout conversion, invalid modes and synchronized timing, while preserving the existing service and evaluation checks. On an NPU the numerical test also compares five sequence lengths plus batch=2 and batch=8 with the official HF recurrence and checks repeat-call state isolation. The CI workflow runs the CPU contracts; the NPU numerical case is explicitly skipped without torch-npu/hardware.

The current implementation is eager and single-device. `DecisionEngine.predict_batch` supports text batches with right padding and per-request field restoration; batches containing images use the original per-request processor path. It does not claim graph capture, tensor parallelism, NPU training, or complete multimodal benchmark coverage. The upstream `release-manifest.json` remains the original release record; evaluation `complete.json` records the actual fork source hashes and checkpoint hashes.

## Released evaluation guard

The released Hard JSONL contains no `gold_probs` reference distributions. The original suite finalizer required exactly ten, so it could finish all predictions and then fail before writing `complete.json`. This fork checks the number of supplied reference distributions in each actual dataset instead, including shard merging; it still rejects missing distribution scores. This changes validation, not labels, predictions or the accuracy formula.
