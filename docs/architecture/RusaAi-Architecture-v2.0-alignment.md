# RusaAi Architecture v2.0 — RusaModels alignment

**Source:** RusaAi Unified CPU-First AI Architecture v2.0 (2026-11-11)  
**Repository:** RusaModels (Model Plane)  
**Scope:** model registry, artifact lifecycle, CPU runtime, inference API and model telemetry.

## Binding decisions

1. The v2.0 model target is `rusaai-edge-1.54b`, not Qwen3-4B.
2. A non-Qwen open-weight checkpoint must be selected and its license, immutable revision, actual architecture and tokenizer/chat template validated before a deployable artifact is claimed.
3. The 1,541,277,696 parameter Transformer description is a reference design. It is not a checkpoint or a runnable model. If the selected checkpoint differs, its actual configuration is authoritative and the result is a derivative/adaptation.
4. GGUF Q4_K_M is the deployment target. The artifact is currently not built; no filename or SHA-256 is assigned.
5. llama.cpp is the CPU runtime. GPU offload remains zero.
6. Start with 4K context. Enable 8K only after device RAM acceptance.
7. Initial runtime values: generation threads 4 (adjust to physical cores), batch threads 4, HTTP threads 2, batch/ubatch 128/64, one active generation.
8. Rusaai owns the pending queue (maximum 3), reasoning controller, context manager, memory/RAG, resource backpressure, ORM, PRM-Lite, Policy Engine and Tool Gateway.
9. RusaModels owns model/artifact lifecycle, model loading, inference and runtime telemetry. It does not own workflow decisions or tool execution.
10. Carbon and Astra are capability profiles over one resident model, not separate model copies.

## Historical compatibility

The Qwen3-4B GGUF Q4_K_M profile from RUSA AI CPU Deployment Configuration v1.0 is preserved under `configs/runtime/archive/` and `docs/archive/`. It must not be silently relabeled as RusaAi-Edge or used by the v2.0 launcher.

## Current implementation status

The v2.0 model manifest is deliberately `DESIGN_ONLY`:
- checkpoint `NOT_SELECTED`;
- tokenizer `NOT_SELECTED`;
- training `NOT_STARTED`;
- artifact `NOT_BUILT`;
- benchmark `NOT_TESTED`.

The native Windows launcher validates these gates and refuses to start until the required evidence is added to the manifest and the artifact checksum matches. CI success proves code/configuration validation only; it does not prove model availability or target-device acceptance.
