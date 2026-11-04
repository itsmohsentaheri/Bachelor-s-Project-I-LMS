import json

import pytest
from pydantic import ValidationError

from bachelor_s_project_i_lms.llm.schemas import (
    AtomMetadata,
    DecomposeOutput,
    ErrorDiagnosis,
    LocalEdge,
    McqItem,
    OpenAnswerJudgement,
)


def mcq_payload():
    return {
        "style": "conceptual",
        "stem": "Why is TCP considered connection-oriented?",
        "options": [
            {"key": key, "text": f"Option {key}"}
            for key in "ABCD"
        ],
        "correct_key": "A",
        "distractor_map": {"B": "m1", "C": "m2", "D": "m3"},
    }


def test_valid_mcq():
    item = McqItem(**mcq_payload())
    assert item.correct_key == "A"


def test_mcq_missing_distractor_rejected():
    payload = mcq_payload()
    payload["distractor_map"] = {"B": "m1"}

    with pytest.raises(ValidationError):
        McqItem(**payload)


def test_mcq_empty_misconception_id_rejected():
    payload = mcq_payload()
    payload["distractor_map"]["B"] = ""

    with pytest.raises(ValidationError):
        McqItem(**payload)


def test_misconception_needs_id():
    with pytest.raises(ValidationError):
        ErrorDiagnosis(
            error_type="MISCONCEPTION",
            confidence=0.8,
            evidence="Confused TCP with UDP",
        )


def test_prereq_gap_needs_title():
    with pytest.raises(ValidationError):
        ErrorDiagnosis(
            error_type="PREREQ_GAP",
            confidence=0.8,
            evidence="Cannot explain packet routing",
        )


def test_duplicate_subtopics_rejected():
    with pytest.raises(ValidationError):
        DecomposeOutput(
            subtopics=[
                {"title": "TCP", "summary": "TCP fundamentals", "is_atomic": True},
                {"title": " tcp ", "summary": "Another TCP topic", "is_atomic": True},
                {"title": "UDP", "summary": "UDP fundamentals", "is_atomic": True},
            ]
        )


def test_self_loop_rejected():
    with pytest.raises(ValidationError):
        LocalEdge(
            prerequisite="TCP",
            dependent=" tcp ",
            confidence=0.9,
            rationale="A topic cannot require itself",
        )


def test_atom_time_limit_enforced():
    with pytest.raises(ValidationError):
        AtomMetadata(
            learning_objective="Explain the TCP three-way handshake",
            estimated_time_min=11,
            difficulty_weight=3,
            type="concept",
            common_misconceptions=[
                {"id": "m1", "description": "Confuses SYN and ACK"},
                {"id": "m2", "description": "Assumes UDP uses handshakes"},
            ],
            source_chunk_ids=["chunk-1"],
        )


def test_open_judgement_rejects_correctness_field():
    with pytest.raises(ValidationError):
        OpenAnswerJudgement(
            scores=[
                {
                    "criterion": "Explains SYN",
                    "points": 1,
                    "evidence": "The answer mentions SYN",
                }
            ],
            confidence=0.9,
            is_correct=True,
        )


def test_llm_json_output_is_validated():
    raw = json.dumps(mcq_payload())
    item = McqItem.model_validate_json(raw)

    assert item.correct_key == "A"
