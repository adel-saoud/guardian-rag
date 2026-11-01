# Judge labels

`judge_labels.jsonl` holds 30 hand-labelled cases used to measure how far the
faithfulness judge can be trusted. Each case has a question and a reference
answer from the golden corpus, a candidate answer, and `label`: true when the
candidate says the same thing as the reference. Placeholders such as
`[PERSON_1]` and citations such as `[1]` do not change the label. An answer
that abstains ("I do not know") or leaves out the fact asked for is labelled
false, because every question is answerable from its document.

Every row is `labelled_by: maintainer`: the maintainer reviewed each label.
A new case starts as `maintainer-draft` and moves to `maintainer` on review.

`scripts/eval_faithfulness.py` runs the judge on every case and reports the
agreement rate, Cohen's kappa and the confusion counts in
`eval/results/faithfulness.json`.
