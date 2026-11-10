# Model Module Migration from RusaAi

## Purpose

This change establishes RusaModels as the owner of model-specific development, training, artifact verification, quantization integration, and local inference. RUSA AI remains the owner of agent orchestration, knowledge retrieval, memory, MCP, policy, workflow, and audit.

The migration is additive first. The original RusaAi modules remain in the source repository until the RusaModels branch is merged, the API integration is verified, and a separate RusaAi cutover change is approved. This avoids breaking the existing agent runtime during the transition.

## Module mapping

| Source in RusaAi | Destination in RusaModels | Decision |
|---|---|---|
| `llm-project/model/blueprint.py` | `rusamodels/blueprint.py` | Migrated as a research blueprint; not a trained model |
| `llm-project/configs/model.yaml` | `configs/architecture/model-blueprint.yaml` | Migrated |
| `llm-project/configs/inference.yaml` | `configs/runtime/inference-blueprint.yaml` | Migrated as a design reference |
| `llm-project/configs/training.yaml` | `configs/training/agent-sft.yaml` | Migrated with RusaModels paths |
| `llm-project/configs/training-t4.yaml` | `configs/training/agent-sft-t4.yaml` | Migrated as optional GPU experiment |
| `rusa_training/dataset.py` | `rusamodels/training/dataset.py` | Migrated and decoupled from the RUSA AI package |
| `llm-project/train/*` | `rusamodels/training/*` | SFT validation, QLoRA trainer and adapter merge migrated |
| `llm-project/eval/evaluate_agent_sft.py` | `rusamodels/evaluation/evaluate_agent_sft.py` | Migrated |
| `llm-project/data/curated/*` | `data/agent_sft/*` | Synthetic seed dataset and provenance migrated |
| `schemas/llm/*` | `schemas/llm/*` | Migrated and RusaModels schema IDs assigned |
| `rusa_agents/gguf.py` | `rusamodels/gguf.py` | Migrated; artifact verification belongs with model artifacts |
| `scripts/verify_gguf_model.py` | `scripts/verify_gguf_model.py` | Migrated with RusaModels environment names |
| `scripts/create_model_manifest.py` | `scripts/create_safetensors_manifest.py` | Migrated |
| `scripts/benchmark_inference.py` | `scripts/benchmark_inference.py` | Migrated to the authenticated RusaModels Model Server API |
| `scripts/train_rusa_agent_t4.sh` | `scripts/train_agent_sft_t4.sh` | Migrated |
| `agent-sft-training.yml` | `.github/workflows/agent-sft-training.yml` | Migrated as a manual, GPU-runner-only workflow |
| `notebooks/RUSA_AI_QLoRA_T4_Colab.ipynb` | `notebooks/RusaModels_QLoRA_T4_Colab.ipynb` | Migrated with repository paths and output locations updated |

## Components intentionally retained in RUSA AI

- `llm-project/serve/app.py`: RUSA AI application/API and task orchestration boundary.
- `rusa_agents/providers.py`: model client/provider adapter; it should call RusaModels through the authenticated OpenAI-compatible API.
- Supreme Orchestrator, agent runtime, worker contracts, Tool Gateway and Policy Engine.
- RAG, Knowledge, Memory, MCP, workflow, audit and autonomous DevOps modules.

These components consume model capabilities; they do not own model weights or model lifecycle.

## Components not copied

- `rusa_quant/pipeline.py`: its GPTQ/CUDA-specific path is not the CPU deployment quantizer. RusaModels already has a maintained GGUF conversion/quantization engine for Q8/Q6/Q5/Q4.
- `llm-project/serve/model_runtime.py`: this is a local Transformers/vLLM runtime coupled to the former RUSA-AI serving application. The CPU-first RusaModels deployment uses a private llama.cpp GGUF runtime instead. A separate Transformers backend can be added later if it has an explicit product requirement and independent tests.
- `llm-project/serve/app.py`: moving it would incorrectly move RUSA AI's application and agent API into the model platform.

## Training contract boundary

The SFT dataset validator needs role and tool definitions to reject invalid training examples. RusaModels therefore carries a versioned snapshot in `rusamodels/training/agent_contract.py`. This snapshot is used only for dataset validation and model evaluation. It is not a runtime permission source. RUSA AI's live Tool Gateway and Policy Engine remain authoritative and must be tested independently.

Update the snapshot and its source commit whenever the RUSA AI worker contract changes. Do not let a model-training contract grant runtime permissions.

## Runtime ownership

RusaModels owns:
- GGUF and safetensors model artifact lifecycle.
- Model catalog, versions, quantization and immutable artifact checksums.
- CPU LLM runtime and local embedding runtime.
- Authenticated OpenAI-compatible chat and embedding APIs.
- Training/evaluation utilities and model release evidence.

RUSA AI owns:
- RAG orchestration, chunking, retrieval, source attribution and vector storage.
- Memory lifecycle and agent context assembly.
- Model selection at the agent/workflow layer.
- Tool authorization and all side effects.

## CPU and training distinction

The Codespaces CPU profile is an inference/development profile: one GGUF generation model plus one small local embedding model. QLoRA training is an optional separate CUDA workflow (for example, an approved Colab T4 or trusted GPU runner). It is not part of the 16 GB CPU inference profile and is not run automatically.

## Cutover sequence

1. Merge and test the RusaModels migration.
2. Deploy the CPU model and embedding runtimes and verify both API endpoints.
3. Configure RUSA AI's LLM client to use the RusaModels Model Server URL and manager/service token.
4. Configure RUSA AI's embedding client to use the RusaModels embedding endpoint and update the RUSA AI vector dimension/schema to match the selected embedding model.
5. Run RUSA AI agent-contract, RAG, tenant-isolation and end-to-end tests.
6. Only then remove the deprecated model-training and local-inference copies from RusaAi in a separate reviewed change.
