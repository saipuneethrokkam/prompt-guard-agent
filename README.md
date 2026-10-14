# Prompt-Guard Agent

A LangGraph tool-using agent protected by a **fine-tuned prompt-injection classifier** that screens both
user input and tool output (indirect injection).

```
START -> input_guard -+-> refuse -> END
                      +-> agent -+-> END
                                 +-> tools -> tool_guard -> agent
```

## Run it (Colab with a free GPU works fine)

```bash
pip install -r requirements.txt
python train_guard.py                      # fine-tunes on deepset/prompt-injections -> ./guard_model
python train_guard.py --push USER/prompt-guard   # optional: publish to the Hugging Face Hub
export GROQ_API_KEY=...                    # free key from console.groq.com
python agent.py                            # try: "read poisoned.txt"
python agent.py --no-guard                 # baseline for comparison
python evaluate.py --e2e                   # detection / false-positive / canary-leak rates
```

## Results (fill in from your own runs)

| Metric | Value |
|---|---|
| Test F1 (deepset test split) | _from guard_model/test_metrics.json_ |
| Red-team attack detection rate | _from eval_results.json_ |
| False-positive rate on benign prompts | _from eval_results.json_ |
| Canary leak rate, unguarded -> guarded | _from `evaluate.py --e2e`_ |

## Design notes
- Model selection uses a validation split carved from train; the official test split is used once for final numbers.
- Long tool outputs are scored in overlapping windows (max score), since the classifier only sees 128 tokens.
- `demo_notes/poisoned.txt` demonstrates indirect injection: the guarded agent withholds the note, the baseline reads it.

## Limitations (keep these in your write-up)
- The training set is small (~550 examples), so the model will miss novel attack styles. Report the misses
  listed in `eval_results.json` rather than hiding them.
- `redteam.jsonl` is small and hand-written; it is a smoke test, not a benchmark.
- A classifier guard is one layer of defense, not a guarantee.
