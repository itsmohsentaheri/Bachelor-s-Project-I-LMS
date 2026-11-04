# src/bachelor_s_project_i_lms/llm/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeType = Literal["concept", "practical", "test", "reading"]


# ---------- Graph building ----------

class Misconception(BaseModel):
    id: str = Field(min_length=1, description="short slug, e.g. tcp_udp_ack_confusion")
    description: str = Field(min_length=5)


class SubTopic(BaseModel):
    title: str = Field(min_length=2)
    summary: str = Field(min_length=5)
    is_atomic: bool


class DecomposeOutput(BaseModel):
    """Step 2: split a topic into 3-7 sub-topics."""
    subtopics: list[SubTopic] = Field(min_length=3, max_length=7)

    @model_validator(mode="after")
    def unique_titles(self):
        titles = [s.title.strip().lower() for s in self.subtopics]
        if len(titles) != len(set(titles)):
            raise ValueError("duplicate subtopic titles")
        return self


class LocalEdge(BaseModel):
    prerequisite: str
    dependent: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=5)

    @model_validator(mode="after")
    def not_self_loop(self):
        if self.prerequisite.strip().lower() == self.dependent.strip().lower():
            raise ValueError("self-loop edge")
        return self


class LocalPrereqOutput(BaseModel):
    """Step 3: prerequisite edges among the siblings only."""
    edges: list[LocalEdge] = Field(default_factory=list)


class CrossBranchVerdict(BaseModel):
    """Global pass: is node X a prerequisite of node Y?"""
    is_prerequisite: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=5)


class AtomMetadata(BaseModel):
    """Metadata filled for every atomic node."""
    learning_objective: str = Field(min_length=10)
    estimated_time_min: int = Field(ge=1, le=10)
    difficulty_weight: int = Field(ge=1, le=5)
    type: NodeType
    common_misconceptions: list[Misconception] = Field(min_length=2, max_length=4)
    source_chunk_ids: list[str] = Field(min_length=1)


# ---------- Question bank ----------

class McqOption(BaseModel):
    key: Literal["A", "B", "C", "D"]
    text: str = Field(min_length=1)


class McqItem(BaseModel):
    style: Literal["conceptual", "application"]
    stem: str = Field(min_length=10)
    options: list[McqOption] = Field(min_length=4, max_length=4)
    correct_key: Literal["A", "B", "C", "D"]
    distractor_map: dict[str, str]

    @model_validator(mode="after")
    def check_consistency(self):
        keys = {o.key for o in self.options}
        if keys != {"A", "B", "C", "D"}:
            raise ValueError("options must have keys A, B, C, D exactly once")
        wrong = keys - {self.correct_key}
        if set(self.distractor_map) != wrong:
            raise ValueError("distractor_map must cover exactly the wrong options")
        if any(not misconception_id.strip() for misconception_id in self.distractor_map.values()):
            raise ValueError("each distractor requires a non-empty misconception_id")
        return self


class RubricCriterion(BaseModel):
    criterion: str = Field(min_length=3)
    max_points: int = Field(ge=1, le=3)


class OpenItem(BaseModel):
    style: Literal["conceptual", "application"]
    stem: str = Field(min_length=10)
    reference_answer: str = Field(min_length=5)
    rubric: list[RubricCriterion] = Field(min_length=2, max_length=5)


class ItemVerification(BaseModel):
    """A separate solver model answers the item; used to reject bad items."""
    solver_answer: str
    is_unambiguous: bool
    matches_key: bool


# ---------- Runtime: diagnosis and judging ----------

class ErrorDiagnosis(BaseModel):
    error_type: Literal["SLIP", "MISCONCEPTION", "PREREQ_GAP", "UNKNOWN"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=5)
    misconception_id: Optional[str] = None
    suspected_prereq_title: Optional[str] = None

    @model_validator(mode="after")
    def type_specific(self):
        if self.error_type == "MISCONCEPTION" and not self.misconception_id:
            raise ValueError("MISCONCEPTION requires misconception_id")
        if self.error_type == "PREREQ_GAP" and not self.suspected_prereq_title:
            raise ValueError("PREREQ_GAP requires suspected_prereq_title")
        return self


class CriterionScore(BaseModel):
    criterion: str
    points: int = Field(ge=0)
    evidence: str


class OpenAnswerJudgement(BaseModel):
    """Judge for open answers. The mapping to correct/incorrect is done in code."""
    model_config = ConfigDict(extra="forbid")

    scores: list[CriterionScore] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
