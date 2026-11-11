# RUSA-EDGE-8 — Windows laptop and Android ARM64 profile

**Architecture:** RusaAi Unified CPU-First AI Architecture v2.0  
**Profile:** `rusa-edge-8`  
**Profile version:** 2.0.0  
**Status:** Architecture target — not yet deployable  
**Target devices:** 8 GB physical-RAM Windows laptop or Android ARM64 device  
**Acceleration:** CPU-only; GPU offload 0

## 1. Source-defined target

The architecture specifies:

- 8 GB physical RAM. Android virtual/extended RAM does not count as physical RAM.
- Laptop or Android ARM64 device.
- Quantized 1B–1.5B model target, Q4_K_M.
- 4,096-token context.
- One session and one active generation.
- Small, bounded pending work queue owned by Rusaai.
- llama.cpp CPU-compatible runtime and GPU offload 0.
- Close background applications and monitor free RAM.

Reference devices named by the architecture:

| Device | Processor | Physical RAM |
|---|---|---:|
| Lenovo V14 G3 IAP | Intel Core i5-1235U | 8 GB |
| Infinix GT 30 Pro | MediaTek Dimensity 8350 Ultimate | 8 GB |

The profile is declarative only. It does not claim that either device has been benchmarked or that a runnable model artifact exists.

## 2. Model-size alignment must be validated

The architecture's general 8 GB profile describes a 1B–1.5B quantized model range. The separate RusaAi-Edge reference architecture is 1,541,277,696 parameters (about 1.541B), slightly above that stated upper bound.

This profile records both source values rather than silently reconciling them. The RusaAi-Edge reference must not be assumed to fit either 8 GB device. Select a compatible non-Qwen checkpoint, audit its license and actual architecture, and benchmark the real Q4_K_M artifact on each target device. If the selected checkpoint differs from the reference configuration, document it as a derivative/adaptation.

Current model readiness remains:

- Checkpoint: `NOT_SELECTED`
- Tokenizer: `NOT_SELECTED`
- GGUF artifact: `NOT_BUILT`
- Device benchmark: `NOT_TESTED`

## 3. Runtime and resource policy

- Use a CPU-compatible llama.cpp build for the target OS and architecture.
- Keep context at 4K; do not inherit the 16 GB profile's 8K target.
- Keep one session and one active generation.
- Keep pending work bounded. The architecture does not specify a numeric queue capacity for this profile; Rusaai must define it.
- Use Q8_0 KV cache only if the selected runtime supports it and target-device validation passes.
- Keep GPU offload at 0.
- Monitor free physical RAM. Activate Rusaai memory protection below 2 GB free RAM.
- Close unnecessary background applications.
- Do not count Android virtual/extended RAM as physical capacity.

Rusaai owns the queue, resource backpressure, reasoning controller and workflow policy. RusaModels owns model loading, runtime and inference; it does not own agent scheduling or tool execution.

## 4. Deployment readiness sequence

1. Confirm exact device SKU, OS version, CPU architecture and physical RAM.
2. Select a non-Qwen open-weight checkpoint and record immutable revision and license.
3. Audit architecture, tokenizer and chat template.
4. Build or obtain a legally usable GGUF Q4_K_M artifact and record SHA-256.
5. Verify CPU-only runtime startup and model residency.
6. Run a 4K context test on the actual device.
7. Record minimum free RAM, CPU load, latency, TTFT and generation tokens/second.
8. Verify one session and one active generation; configure a small bounded queue in Rusaai.
9. Run a stable-session test and restart-recovery test.
10. Validate Windows and Android separately. A Windows pass does not prove Android ARM64 compatibility.

## 5. Acceptance boundary

Do not mark this profile deployable until the actual checkpoint, tokenizer, GGUF artifact and target-device measurements are recorded. CI can validate the YAML and repository code; it cannot prove memory fit or runtime performance on the named devices.
