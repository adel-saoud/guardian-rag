# Architecture decision records

One file per significant decision, in [MADR 4](https://adr.github.io/madr/)
format. `proposed` means the reasoning still waits for the maintainer's
confirmation; facts and measurements cite the code or
[`results/manifest.json`](../../results/manifest.json).

| ADR | Title | Status |
|:--|:--|:--|
| [0001](0001-presidio-gliner-orchestration.md) | Presidio + GLiNER orchestration | superseded by 0010 |
| [0002](0002-gliner-over-privacy-filter.md) | GLiNER over `openai/privacy-filter` | superseded by 0010 |
| [0003](0003-conditional-deberta-validation.md) | Conditional DeBERTa validation | superseded by 0010 |
| [0004](0004-hierarchical-chunking.md) | Hierarchical chunking, parent 1024 / child 256 tokens | accepted |
| [0005](0005-typed-placeholders.md) | Typed placeholders, not blank tokens | accepted |
| [0006](0006-post-generation-audit.md) | Post-generation PII audit, last stage, fail-closed | accepted |
| [0007](0007-qdrant-over-chroma.md) | Qdrant over Chroma | accepted |
| [0008](0008-cross-encoder-reranking.md) | Cross-encoder reranking, built and shipped off | accepted |
| [0009](0009-prompt-injection-screening.md) | Prompt-injection screening: classifier on queries, rules on chunks | accepted |
| [0010](0010-three-stage-pii-detection.md) | Three-stage PII detection: Presidio, GLiNER, conditional DeBERTa | accepted |
| [0011](0011-faithfulness-reported-not-gated.md) | Faithfulness as a reported, non-gating metric, checked against hand labels | accepted |
| [0012](0012-qwen35-local-model-thinking-off.md) | `qwen3.5:4b` on Ollama for generation and judging, with thinking off | accepted |
| [0013](0013-redact-organisation-names.md) | Redact organisation names | accepted |
