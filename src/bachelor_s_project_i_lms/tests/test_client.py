import pytest
from pydantic import BaseModel
from bachelor_s_project_i_lms.llm.client import call_llm, _extract_json

class SampleResponse(BaseModel):
    name: str
    score: int

def test_extract_json_variants():
    assert _extract_json('\n{"name": "ali", "score": 20}\n') == '{"name": "ali", "score": 20}'
    assert _extract_json('توضیحات قبل {"name": "reza", "score": 18} توضیحات بعد') == '{"name": "reza", "score": 18}'

def test_call_llm_mock_success():
    def fake_generate(messages, system):
        return 'json\n{"name": "Mohsen", "score": 100}\n', {"input_tokens": 10, "output_tokens": 15}

    result = call_llm(
        messages=[{"role": "user", "content": "hi"}],
        response_model=SampleResponse,
        generate=fake_generate,
        learner_id="test_user"
    )

    assert result.name == "Mohsen"
    assert result.score == 100

def test_call_llm_retry_on_bad_json():
    attempts = 0
    def failing_then_success(messages, system):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return '{"name": "Mohsen", "score": "not_an_int"}', {"input_tokens": 5, "output_tokens": 5}
        return '{"name": "Mohsen", "score": 95}', {"input_tokens": 15, "output_tokens": 10}

    result = call_llm(
        messages=[{"role": "user", "content": "hi"}],
        response_model=SampleResponse,
        generate=failing_then_success,
        max_retries=1
    )

    assert result.score == 95
    assert attempts == 2
