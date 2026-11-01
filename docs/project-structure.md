# Project structure

```
src/guardian_rag/
├── config.py                  env-driven settings (GUARDIAN_*), see configuration.md
├── main.py                    ASGI entry point
├── pii/                       3-stage detector, redactor, post-generation audit
├── ingestion/                 loader, hierarchical chunker, pipeline, redaction store
├── retrieval/                 embedder, Qdrant store, BM25 index, hybrid retriever, reranker
├── injection/                 prompt-injection screeners (classifier, rules), quarantine
├── generation/                Ollama client, grounded prompt, citation verification
├── eval/                      golden corpus loader, recall + precision, judge,
│                              ranking + injection metrics, in-domain corpus generator
├── api/                       FastAPI surface (5 routes), lifespan, dependencies
└── dashboard/                 Streamlit, 3 panels (app.py pyright-excluded); detector.py builds the 3-stage detector

scripts/
├── seed_demo_corpus.py        regenerate the 30-document golden dataset
├── eval.py                    PII recall + precision against the corpus
├── eval_retrieval.py          nDCG / Recall / MRR, with and without reranking
├── eval_injection.py          injection detection and false-positive rates
├── eval_faithfulness.py       answer quality with original vs redacted context, judge agreement
├── reproduce.py               runs every eval above and writes results/manifest.json
├── demo.py                    end-to-end local demo
└── capture_dashboard.py       re-captures docs/dashboard.png from the running dashboard

docs/
├── architecture.md            diagram, components, data flow, failure modes
├── adr/                       architecture decision records (MADR 4), indexed in adr/README.md
├── configuration.md           every GUARDIAN_* setting
├── api.md                     HTTP routes and headers
└── roadmap.md                 deferred work and why

results/
└── manifest.json              every README number, with versions, datasets and sample sizes

eval/
├── golden/                    30 synthetic documents, 91 planted entities, 30 QA pairs
├── faithfulness/              30 judge cases labelled and reviewed by the maintainer
├── retrieval/                 golden smoke queries, in-domain topic specs (domain/) and 135 queries
├── injection/                 60-case adversarial set
└── results/                   committed benchmark scores (JSON)
```
