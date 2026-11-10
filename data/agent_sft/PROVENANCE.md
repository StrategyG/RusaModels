# RusaModels Agent SFT v1 — Dataset provenance

Migrated from `StrategyG/RusaAi` at source commit `c2866e2d9747b9de4ea9d5f9696b8ded11121006`. This is a synthetic seed corpus for experimentation, not evidence of trained weights or production quality.

## Status

- Training split: 120 examples (20 per runtime-supported role).
- Held-out evaluation split: 36 examples (6 per role).
- Evaluation blocked cases: 12/36 (33.3%); unblocked cases: 24/36 (66.7%).
- Format: JSONL matching `schemas/llm/agent-sft-example.schema.json`.
- Training status: dataset prepared; QLoRA has not been executed and no trained weights are included.

## Source basis

This seed corpus is grounded in the RUSA source materials and the current runtime contract:

1. `RUSA_AI_Blueprint_V2.md` — draft blueprint, role reconciliation, single workflow authority, evidence-based completion, least privilege, policy-before-action, and human control.
2. `RUSA_AI_All_Agents-2.md` — agent responsibilities, inputs/outputs, restrictions, workflow, and success criteria.
3. `RUSA_AI_AGENTIC_DEVOPS_LIFECYCLE(1).md` — Plan-to-Monitor lifecycle, handoffs, repository evidence, QA, security, and approval boundaries.
4. `ADR-RUSA-AI-Komunikasi-Agent(1).md` — task/message contracts, delivery, idempotency, retry, cancellation, security, and audit.
5. `RUSA_AI_Architecture.md` — the four product domains.
6. Repository runtime sources: `rusa_agents/agent_tools.py`, `rusa_agents/contracts.py`, `rusa_agents/autonomous.py`, `rusa_agents/tool_gateway.py`, the SFT JSON Schema, and the existing QLoRA trainer.

## Role mapping

The current SFT runtime contract supports exactly six worker roles: `analyst`, `architect`, `developer`, `debugger`, `tester`, and `security`. The architecture documents describe additional orchestration and lifecycle agents. Those identities are not fabricated as SFT roles here; they should be added only after runtime contracts and tool permissions support them.

## Construction and limitations

- Examples are synthetic, authored instruction/response pairs derived from documented role boundaries and current tool allowlists. They are not production conversations, observed traces, or human-labeled outcomes.
- Each response uses only the role's existing tool allowlist and the current worker JSON envelope.
- Evaluation prompts are distinct from training prompts and include both permitted and blocked scenarios.
- No private repository content, credentials, personal data, or live tenant data is included.
- This is a seed corpus for an initial experiment, not evidence of production-quality behavior. A domain owner should review and expand it with authorized, representative examples before production promotion.
- Keep evaluation examples held out. Do not move them into training after inspecting model results; create a new versioned evaluation set for later experiments.
