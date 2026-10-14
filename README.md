# AgentEval Lab

A framework for evaluating and improving the reliability of LLM-powered agents:
accuracy, hallucination rate, evidence-groundedness, and latency, measured
before and after targeted improvements on the same dataset.

**Status:** Phase 1 (skeleton and dataset) complete. Phases 2-7 are planned, not built.

## Layout
- `data/corpus/`: fictional document corpus (fictional on purpose, so answers can't come from model pretraining)
- `data/eval_set.jsonl`: test cases (question, expected answer, verbatim evidence quotes)
- `agenteval/schema.py`, `dataset.py`: schema, loading, and validation
- `agenteval/llm.py`: provider-agnostic client interface plus a mock for offline tests
- `tests/`: pytest suite

## Quick start
```bash
pip install -e ".[dev]"
pytest
```

## Roadmap
1. Skeleton and dataset (done)
2. Baseline agent (retrieval + LLM)
3. Evaluation pipeline and metrics
4. Metric tests
5. Failure analysis
6. Improvement and re-evaluation
7. FastAPI and final report
