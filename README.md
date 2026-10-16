# AgentEval Lab

A framework for evaluating and improving the reliability of LLM-powered agents:
accuracy, hallucination rate, evidence-groundedness, and latency, measured
before and after targeted improvements on the same dataset.

**Status:** Phases 1-3 built. Phases 4-7 are planned, not built.

## Layout
- `data/corpus/`: fictional document corpus (fictional on purpose, so answers can't come from model pretraining)
- `data/eval_set.jsonl`: test cases (question, expected answer, verbatim evidence quotes)
- `agenteval/schema.py`, `dataset.py`: schema, loading, and validation
- `agenteval/llm.py`: provider-agnostic client interface plus a mock for offline tests
- `agenteval/retrieval.py`: sentence-level BM25 retriever
- `agenteval/agent.py`: baseline RAG agent (prompt version `baseline-v1`)
- `agenteval/clients.py`: Anthropic and OpenAI-compatible (incl. local Ollama) clients
- `agenteval/run.py`: runs the agent over the eval set and saves raw results
- `agenteval/metrics.py`: deterministic scoring heuristics (correctness, abstention, groundedness)
- `agenteval/judge.py`: optional LLM judge with heuristic fallback
- `agenteval/evaluate.py`: scores a saved run and writes CSV, summary JSON, and a markdown report
- `tests/`: pytest suite

## Quick start
```bash
pip install -e ".[dev]"
pytest
```

## Running the baseline
```bash
python -m agenteval.run --provider mock --out reports/mock.jsonl          # offline smoke test
export ANTHROPIC_API_KEY=...   # pip install -e ".[anthropic]"
python -m agenteval.run --provider anthropic --model <model-name> --out reports/baseline.jsonl
# or any OpenAI-compatible endpoint (OpenAI, Ollama at http://localhost:11434/v1, ...)
OPENAI_BASE_URL=... OPENAI_API_KEY=... python -m agenteval.run --provider openai-compat --model <model-name>
```
Results are raw answers plus retrieved passages, latency, and token counts. Scoring comes in Phase 3.

## Scoring a run
```bash
python -m agenteval.evaluate --results reports/baseline.jsonl
# optional: use an LLM judge for correctness and groundedness (falls back to heuristics if a verdict can't be parsed)
python -m agenteval.evaluate --results reports/baseline.jsonl --judge-provider anthropic --judge-model <model-name>
```
Writes `<name>_scored.csv`, `<name>_summary.json`, and `<name>_report.md` next to the results file.
Metric definitions are in the generated report. Spot-check scored rows by hand before trusting any number.

## Roadmap
1. Skeleton and dataset (done)
2. Baseline agent (retrieval + LLM) (done)
3. Evaluation pipeline and metrics (done)
4. Metric tests
5. Failure analysis
6. Improvement and re-evaluation
7. FastAPI and final report
