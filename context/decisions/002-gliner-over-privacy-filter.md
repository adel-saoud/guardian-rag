# ADR 002: GLiNER over `openai/privacy-filter`

## Context

The widely-cited "privacy-safe RAG" pattern pairs Presidio with
`openai/privacy-filter` for the unstructured-PII layer. We needed to
decide whether to follow that path or switch to a different NER model.

## Decision

Use `urchade/gliner_multi_pii-v1` instead.

|                       | `openai/privacy-filter`          | `urchade/gliner_multi_pii-v1` |
|-----------------------|----------------------------------|-------------------------------|
| Parameters            | 1.4B                             | ~100M                         |
| HF Inference          | Unstable as of mid-2026          | Stable                        |
| Languages             | English only                     | EN, FR, DE, ES, PT, IT        |
| Architecture          | Transformer fine-tune            | GLiNER span-based, faster     |
| Licence               | Apache-2.0                       | Apache-2.0                    |
| Presidio integration  | Not supported                    | Custom `EntityRecognizer`     |

## Consequences

- ~14x parameter reduction makes the pipeline CPU-viable.
- Multilingual coverage matches Guardian's EU AI Act target audience.
- We accept the slight precision dip GLiNER shows on long-form
  organisation names. [ADR 003](003-conditional-deberta-validation.md) covers the
  spans with the highest false-positive risk.
- See `pii/gliner_recognizer.py` for the label → PII type mapping.
