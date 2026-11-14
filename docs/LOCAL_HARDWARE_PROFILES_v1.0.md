# Local Hardware Detection and Runtime Profile v1.0

- Repository: StrategyG/RusaModels
- Status: Hardware inventory utility implemented; recommendations are not device certification
- Scope: local-first desktop installer and model provisioning preflight

## Purpose

RusaModels provides a machine-readable hardware inventory for the RUSA AI local installer. It detects OS, architecture, CPU identity, logical core count, memory where exposed by the OS, free disk space, and NVIDIA GPUs discoverable through `nvidia-smi`.

Run:

```bash
python -m rusamodels.hardware --json
```

The command is read-only. It does not install drivers, reserve memory, download models, change runtime settings, or start inference.

## Conservative profile policy

| Detected physical RAM | Suggested profile | Waiting requests | Initial context |
|---:|---|---:|---:|
| Unknown | unknown | 1 | 2K |
| Below 8 GiB | minimal | 1 | 2K |
| 8–16 GiB | local-8gb | 1 | 2K or 4K by safe estimate |
| 16–32 GiB | local-16gb | 3 | 2K or 4K by safe estimate |
| 32 GiB or more | local-32gb | 3 | 2K or 4K by safe estimate |

The estimate reserves at least 2 GiB or 20% of detected RAM, whichever is larger. It is not a reservation or a model-fit guarantee.

## Safety and model selection

- CPU is the default inference target.
- NVIDIA detection only reports an acceleration candidate. It never enables CUDA or GPU offload.
- Initial concurrency is one generation and the waiting queue is bounded.
- The utility does not select a model by name or download a checkpoint.
- Model selection separately verifies immutable source revision, license, architecture, tokenizer/chat template, manifest, artifact SHA-256 and runtime compatibility.
- A model profile must pass target-device load, latency, thermal and soak tests before it is marked supported.
- Unknown hardware values fail conservatively.

## Integration contract

The installer consumes the JSON inventory and presents the recommended profile in its Hardware screen. Users can accept Automatic mode or open Advanced settings. A custom profile must be checked against the selected model manifest and actual free resources before activation.

This is an inventory/recommendation foundation. It does not yet implement dynamic model loading, a desktop installer, hardware benchmarking, or production certification.
