# AgentEval Lab

A framework for evaluating and improving the reliability of LLM-powered agents. It measures accuracy, hallucination rate, evidence-groundedness, and latency, enabling comparisons before and after targeted improvements on the same evaluation dataset.

**Status:** Phase 1 (skeleton and dataset) complete. Phases 2–7 are planned, not yet implemented.

## Project Structure

* `data/corpus/`: Fictional document corpus designed to reduce the chance of answers being available through model pretraining.
* `data/eval_set.jsonl`: Evaluation cases containing questions, expected answers, and supporting evidence quotes.
* `agenteval/schema.py`: Evaluation data schema.
* `agenteval/dataset.py`: Dataset loading and validation.
* `agenteval/llm.py`: Provider-agnostic LLM client interface and mock client for offline testing.
* `tests/`: Automated tests using pytest.

## Getting Started

### Requirements

* Python 3.10 or newer
* pip

### Installation

Create and activate a virtual environment, then install the project and its development dependencies:

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest
```

## Project Roadmap

1. **Skeleton and dataset:** Project structure, evaluation dataset, schema, and initial tests. *(Complete)*
2. **Baseline agent:** Implement document retrieval and LLM-based answering.
3. **Evaluation pipeline:** Calculate accuracy, groundedness, hallucination-related metrics, and latency.
4. **Metric tests:** Validate metric calculations using controlled examples.
5. **Failure analysis:** Categorize errors such as poor retrieval, ignored evidence, fabricated details, and incorrect refusals.
6. **Improvement and re-evaluation:** Apply targeted changes and compare results on the same evaluation dataset.
7. **FastAPI and reporting:** Add API endpoints and generate evaluation reports.

## Testing

The initial project test suite contains seven tests. Run `python -m pytest` to execute the suite.

## License

No license has been added yet.
