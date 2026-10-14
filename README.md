# AgentEval Lab

A framework for evaluating and improving the reliability of LLM-powered agents:
accuracy, hallucination rate, evidence-groundedness, and latency, measured
before and after targeted improvements on the same dataset.

**Status:** Phases 1-2 built. Phases 3-7 are planned, not built.

## Layout
- `data/corpus/`: fictional document corpus (fictional on purpose, so answers can't come from model pretraining)
- `data/eval_set.jsonl`: test cases (question, expected answer, verbatim evidence quotes)
- `agenteval/schema.py`, `dataset.py`: schema, loading, and validation
- `agenteval/llm.py`: provider-agnostic client interface plus a mock for offline tests
- `agenteval/retrieval.py`: sentence-level BM25 retriever
- `agenteval/agent.py`: baseline RAG agent (prompt version `baseline-v1`)
- `agenteval/clients.py`: Anthropic and OpenAI-compatible (incl. local Ollama) clients
- `agenteval/run.py`: runs the agent over the eval set and saves raw results
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

## Roadmap
1. Skeleton and dataset (done)
2. Baseline agent (retrieval + LLM) (done)
3. Evaluation pipeline and metrics
4. Metric tests
5. Failure analysis
6. Improvement and re-evaluation
7. FastAPI and final report
