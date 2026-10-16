from agenteval.judge import LLMJudge
from agenteval.llm import MockClient
from agenteval.metrics import fact_present, heuristic_correct, is_abstention, is_grounded, words

CTX = [
    "The Harbor Line opened in 2009 and runs 14 stations from Port Alder to Kestrel Heights.",
    "The Canal Line opened in 2021 and is the only line that runs 24 hours a day.",
]


def test_number_normalization_matches_2_4_and_2_40():
    assert heuristic_correct("It costs 2.4 credits.", ["2.40"])


def test_alternatives_in_key_facts():
    assert fact_present("all|three", words("covers all lines"))
    assert not fact_present("all|three", words("covers two lines"))


def test_wrong_answer_fails_and_polarity_via_key_facts():
    facts = ["no|cannot|prohibited", "peak"]
    assert heuristic_correct("No, that is during peak hours.", facts)
    assert not heuristic_correct("Yes, bikes are fine off-peak.", facts)
    assert not heuristic_correct("", facts)


def test_abstention_detection():
    assert is_abstention("I don't know based on the context.")
    assert is_abstention("The context does not mention senior fares.")
    assert not is_abstention("No, bicycles are not permitted at that time.")
    assert not is_abstention("The fare is 2.40 credits.")


def test_grounded_accepts_supported_and_derived_numbers():
    assert is_grounded("The Harbor Line has 14 stations.", CTX, "How many stations?")
    assert is_grounded("12 years", CTX, "How many years passed between the openings?")


def test_grounded_rejects_fabricated_number_and_claims():
    assert not is_grounded("The Harbor Line has 17 stations.", CTX, "How many stations?")
    assert not is_grounded("A senior fare is discounted by half for retirees.", CTX, "senior fare?")


def test_judge_parses_json_and_returns_none_on_garbage():
    assert LLMJudge(MockClient('Sure. {"correct": true}')).correct("q", "e", "a") is True
    assert LLMJudge(MockClient('{"grounded": false}')).grounded("a", ["p"]) is False
    assert LLMJudge(MockClient("no idea")).correct("q", "e", "a") is None
    assert LLMJudge(MockClient('{"correct": "yes"}')).correct("q", "e", "a") is None
