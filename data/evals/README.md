# Evaluation Sets

These are small hand-curated evaluation sets for DecisionLens.

They are meant to help with:

- tuning duplicate / related-case thresholds
- checking whether semantic search surfaces the right historical incidents
- checking whether GPT-assisted troubleshooting guidance is grounded in useful source cases

They are intentionally small because the project dataset is repetitive and templated. The goal is
not to create a large benchmark, but to create a reliable seed set for iteration.

## Files

- `duplicate_eval_set.csv`
  - labeled incident pairs with `same_issue`, `related`, or `different`
- `rag_eval_set.csv`
  - query incidents with expected source tickets and troubleshooting themes

## Suggested use

### Duplicate evaluation

Use `duplicate_eval_set.csv` to:

- run the current duplicate search on the `query_ticket_id`
- check whether `candidate_ticket_id` appears in the returned results
- compare the system output with the expected label
- tune thresholds for:
  - likely duplicate
  - related
  - new issue

### RAG evaluation

Use `rag_eval_set.csv` to:

- run similar-incident retrieval for the `query_ticket_id`
- check whether the expected source tickets appear near the top
- generate a GPT answer
- compare the generated answer against the expected troubleshooting themes

## Notes

- These are hand-reviewed seed examples, not exhaustive truth labels.
- With this dataset, exact duplicate quality is limited by repeated templates and low text variety.
- These files are best used for regression checks and tuning, not for claiming production-grade
  benchmark coverage.
