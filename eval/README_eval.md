# Retrieval ablation (`eval/ablation.py`)

Compares four retrieval strategies on the same golden set with the same generation call, so differences in the
metrics come from retrieval, not from the answer prompt.

| Config | Retrieval | Extra LLM calls |
|---|---|---|
| A vector only | `embed_query` → `vector_store.query_similar` (top 4) | – |
| B hybrid | `retrieval.hybrid_search` (vector + BM25, RRF, top 4) | – |
| C hybrid + condense | rows with `"follow_up": true`: `llm.condense_question` with a 2-turn history built from the previous golden row, then hybrid; all other rows: identical to B | 1 on follow-up rows |
| D agentic | `agent_graph` (plan → retrieve → grade → rewrite ≤ `agent_max_rewrites` → finalize) | plan + grade(s) + rewrite(s) |

Generation for every config: `llm.generate_answer_stream(query, chunks)` joined into one string (the non-streaming
`llm.generate_answer` is commented out in `app/services/llm.py`; this is the same prompt, model and temperature).
`query` is the original question, except for C on follow-up rows, where it is the condensed question. When D's
grader refuses, no generation call is made and the answer is the standard decline sentence, which is how
`/query/agentic` behaves.

## Running

Prerequisites: `.env` with OpenAI and Pinecone keys, and the 10 ML papers in `eval/sample_docs/pdfs/` ingested
(`eval/ingest_once.py`). No server is needed; everything runs in-process.

```bash
./.venv/bin/python eval/ablation.py --limit 5            # smoke test, < 2 min
./.venv/bin/python eval/ablation.py                      # full run, all 40 rows x 4 configs
./.venv/bin/python eval/ablation.py --configs B,D        # subset of configs
./.venv/bin/python eval/ablation.py --no-ragas           # retrieval/refusal/latency only, no judge cost
./.venv/bin/python eval/ablation.py --workers 1          # sequential, for trustworthy latency numbers
./.venv/bin/python eval/ablation.py --golden eval/golden_set.jsonl   # the Open RAG Bench set (needs eval/ingest_bench.py first)
```

`--limit N` takes rows round-robin across types, so even 5 rows cover every type.

Outputs go to `eval/results/<timestamp>/`:
- `results.csv`: one row per (config, question): retrieved sources, answer, refused, latency, LLM calls and every metric.
- `summary.md`: the config × metric table and a config × type breakdown. The same table is printed to stdout.

**Cache.** Answers and retrieval results are cached in `eval/results/cache.json`, keyed by `config|id`, so re-running
(for example to re-score with RAGAS) does not repeat generation. A cache entry is ignored if the row's question
text changed. Delete the file, or pass `--no-cache` (which neither reads nor writes it), after changing retrieval,
prompts or the index.

**Failures.** OpenAI `RateLimitError` is retried with exponential backoff (2, 4, 8, 16s). Any other failure is
logged, the question is skipped for that config, and it is listed under "Failed" in `summary.md`.

## Golden set (`eval/golden_set_ml.jsonl`)

40 rows on the 10 ingested ML papers: 15 `single_fact`, 8 `multi_hop`, 8 `keyword`, 9 `unanswerable`.

Fields: `id`, `question`, `reference_answer` (`"REFUSE"` for unanswerable), `expected_sources` (PDF filenames;
empty for unanswerable), `type`, `verified`, and optionally `follow_up`. Four `single_fact` rows (sf-02, sf-04,
sf-07, sf-14) are follow-ups that use a pronoun ("its big model", "those models") referring to the row just
before them in the file. They are what config C is designed for; A, B and D see the ambiguous question as is.

All rows have `"verified": false` until someone checks them against the PDFs.

`eval/golden_set.jsonl` is a different set, built from the Vectara Open RAG Benchmark by `eval/build_golden.py`.

## Metrics

| Metric | Rows | Meaning |
|---|---|---|
| `hit@4` | answerable | Any expected source among the top 4 retrieved chunks. |
| `MRR@4` | answerable | Mean of 1/rank of the first chunk from an expected source (0 if none in the top 4). |
| `src_recall@4` | answerable | Fraction of the expected sources present in the top 4. Mainly informative for `multi_hop`, which needs two papers. |
| `refusal_unans` | unanswerable | Share of answers containing the decline phrase. Higher is better. |
| `false_refusal_ans` | answerable | Share of answerable questions that were declined. Lower is better. |
| `lat_p50` / `lat_p95` | all | Wall time for retrieval + generation, in seconds. |
| `llm_calls` | all | Chat-completion calls per question (embedding calls not counted). |
| `faithfulness` | answerable | RAGAS: share of the answer's claims supported by the retrieved contexts. |
| `answer_relevancy` | answerable | RAGAS: how well the answer addresses the question (embedding similarity of generated questions). |
| `context_recall` | answerable | RAGAS: share of the reference answer's claims found in the retrieved contexts. |
| `factual_correctness` | answerable | RAGAS: claim-level F1 between the answer and the reference answer. |

"Refused" means the answer contains "no relevant information was found" (the phrase `SYSTEM_PROMPT` tells the model
to use) or "未找到相关信息". RAGAS is skipped for unanswerable rows. It uses the same setup as `eval/ragas_eval.py`
(ragas 0.3.9, `LangchainLLMWrapper` + `LangchainEmbeddingsWrapper`, judge = `settings.openai_chat_model`).

## Known limits

- **Small corpus.** 10 papers on distinct topics make retrieval easy: `hit@4` saturates near 1.0 for every config,
  so differences show up mainly in multi-hop, follow-up and refusal behaviour. Use the Open RAG Bench set
  (`--golden eval/golden_set.jsonl`, 25 more papers) for a harder retrieval test.
- **Small samples.** 8–15 rows per type means one question moves a per-type number by 7–12 points. Treat
  differences of a few points as noise.
- **LLM-judge variance.** RAGAS scores come from `gpt-4o-mini` and vary between runs on identical inputs. Compare
  configs within one run (they are scored together), and repeat runs before trusting small gaps.
- **Refusals score 0 in RAGAS.** A declined answerable question gets faithfulness and factual_correctness of 0, so
  the RAGAS averages mix answer quality with false refusals. Read them together with `false_refusal_ans`.
- **D sees more context.** D merges chunks from every sub-query and rewrite (often 8–14 chunks); A–C pass 4 to the
  generator. `hit@4`/`MRR@4` are computed on D's top 4 by score for comparability, but D's answers and RAGAS
  scores benefit from the extra context. D also has no condense step, unlike `/query/agentic`.
- **`answer_relevancy` uses 1 generation.** LangChain's ChatOpenAI wrapper returns one completion, so strictness is
  set to 1 instead of the default 3.
- **Latency under concurrency.** The default `--workers 4` runs questions in parallel. A warm-up call removes the
  cold-start bias, but use `--workers 1` for latency numbers you want to quote.
- **Golden set partly hand-checked.** Questions and reference answers were written from the papers but are
  `verified: false` until checked against the PDFs. A wrong reference answer lowers `context_recall` and
  `factual_correctness` for every config equally.
