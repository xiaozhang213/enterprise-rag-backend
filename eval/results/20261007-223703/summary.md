# Ablation results (2026-10-07 22:37)

golden: `golden_set_ml.jsonl`, rows: 5, configs: A,B,C,D, model: `gpt-4o-mini`, top_k: 4, workers: 4, ragas: on

## By config

| config | name | n | hit@4 | MRR@4 | src_recall@4 | refusal_unans | false_refusal_ans | lat_p50 | lat_p95 | llm_calls | faithfulness | answer_relevancy | context_recall | factual_correctness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | vector only | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 3.484 | 3.788 | 1.000 | 0.750 | 0.746 | 0.750 | 0.700 |
| B | hybrid | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 1.111 | 1.658 | 1.000 | 0.750 | 0.734 | 0.750 | 0.750 |
| C | hybrid + condense | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 0.964 | 1.828 | 1.200 | 0.750 | 0.733 | 0.750 | 0.750 |
| D | agentic | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 2.816 | 6.559 | 3.400 | 0.750 | 0.734 | 1.000 | 0.750 |

## By config x type

| config | type | n | hit@4 | MRR@4 | src_recall@4 | refusal_unans | false_refusal_ans | faithfulness | factual_correctness | llm_calls |
|---|---|---|---|---|---|---|---|---|---|---|
| A | keyword | 1 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 1.000 |
| A | multi_hop | 1 | 1.000 | 1.000 | 1.000 | – | 1.000 | 0.000 | 0.000 | 1.000 |
| A | single_fact | 2 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 0.900 | 1.000 |
| A | unanswerable | 1 | – | – | – | 1.000 | – | – | – | 1.000 |
| B | keyword | 1 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 1.000 |
| B | multi_hop | 1 | 1.000 | 1.000 | 1.000 | – | 1.000 | 0.000 | 0.000 | 1.000 |
| B | single_fact | 2 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 1.000 |
| B | unanswerable | 1 | – | – | – | 1.000 | – | – | – | 1.000 |
| C | keyword | 1 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 1.000 |
| C | multi_hop | 1 | 1.000 | 1.000 | 1.000 | – | 1.000 | 0.000 | 0.000 | 1.000 |
| C | single_fact | 2 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 1.500 |
| C | unanswerable | 1 | – | – | – | 1.000 | – | – | – | 1.000 |
| D | keyword | 1 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 3.000 |
| D | multi_hop | 1 | 1.000 | 1.000 | 1.000 | – | 1.000 | 0.000 | 0.000 | 4.000 |
| D | single_fact | 2 | 1.000 | 1.000 | 1.000 | – | 0.000 | 1.000 | 1.000 | 3.000 |
| D | unanswerable | 1 | – | – | – | 1.000 | – | – | – | 4.000 |

Failed (skipped): none
