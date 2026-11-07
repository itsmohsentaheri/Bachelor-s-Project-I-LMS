"""Tests for syllabus.py — the course .md parser.

Location:  <project root>/tests/test_syllabus.py
Run from the project root:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

# Make `src/` importable no matter where pytest is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bachelor_s_project_i_lms.graph import syllabus  # noqa: E402
from bachelor_s_project_i_lms.graph.syllabus import (  # noqa: E402
    find_course_file,
    parse_course_file,
)

SAMPLE_MD = """\
course name: computers network
Course objective: Computer engineering students

1: Computer Networks and the Internet
2: Application Layer
3: Transport Layer
4: The Network Layer: Data Plane
5: The Network Layer: Control Plane
6: The Link Layer and LANs
7: Wireless and Mobile Networks
8: Security in Computer Networks
"""


@pytest.fixture
def sample_file(tmp_path):
    """Write the standard syllabus content to a temporary .md file."""
    f = tmp_path / "computer_networks.md"
    f.write_text(SAMPLE_MD, encoding="utf-8")
    return f


# ------------------------------------------------------------- parse: basics
def test_parses_course_metadata(sample_file):
    result = parse_course_file(sample_file)
    assert result["course name"] == "computers network"
    assert result["course objective"] == "Computer engineering students"


def test_metadata_keys_are_normalized_to_lowercase(sample_file):
    result = parse_course_file(sample_file)
    assert "course objective" in result
    assert "Course objective" not in result


def test_chapter_keys_are_integers(sample_file):
    chapters = parse_course_file(sample_file)["chapters"]
    assert all(isinstance(k, int) for k in chapters)
    assert sorted(chapters) == list(range(1, 9))


def test_chapter_titles_are_correct(sample_file):
    chapters = parse_course_file(sample_file)["chapters"]
    assert chapters[1] == "Computer Networks and the Internet"
    assert chapters[8] == "Security in Computer Networks"


def test_title_keeps_inner_colon(sample_file):
    # '4: The Network Layer: Data Plane' must NOT be split at the second colon
    chapters = parse_course_file(sample_file)["chapters"]
    assert chapters[4] == "The Network Layer: Data Plane"
    assert chapters[5] == "The Network Layer: Control Plane"


# -------------------------------------------------- parse: robustness / edges
def test_markdown_decorations_are_stripped(tmp_path):
    f = tmp_path / "decorated.md"
    f.write_text(
        "# course name: networks\n"
        "- 1: Application Layer\n"
        "* 2: Transport Layer\n",
        encoding="utf-8",
    )
    result = parse_course_file(f)
    assert result["course name"] == "networks"
    assert result["chapters"] == {1: "Application Layer", 2: "Transport Layer"}


def test_dot_and_paren_numbering_also_supported(tmp_path):
    f = tmp_path / "numbering.md"
    f.write_text("1. Intro\n2) Advanced topics\n3: Wrap-up\n", encoding="utf-8")
    assert parse_course_file(f)["chapters"] == {
        1: "Intro", 2: "Advanced topics", 3: "Wrap-up",
    }


def test_blank_lines_are_skipped(tmp_path):
    f = tmp_path / "blank.md"
    f.write_text("\n\n1: Only chapter\n\n\n", encoding="utf-8")
    assert parse_course_file(f)["chapters"] == {1: "Only chapter"}


def test_empty_file_returns_empty_chapters(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("", encoding="utf-8")
    assert parse_course_file(f) == {"chapters": {}}


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        parse_course_file("no/such/file.md")


# ---------------------------------------------------- find_course_file tests
def test_find_course_file_returns_existing_path():
    found = find_course_file("computer_networks.md")
    assert isinstance(found, Path)
    assert found.exists()
    assert found.name == "computer_networks.md"


def test_find_course_file_error_lists_searched_dirs(tmp_path, monkeypatch):
    # Point every search location at an empty temp dir -> nothing findable
    monkeypatch.setattr(syllabus, "PKG_DIR", tmp_path)
    monkeypatch.setattr(syllabus, "BASE_DIR", tmp_path)
    monkeypatch.setattr(syllabus, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        find_course_file("ghost.md")
    assert "ghost.md" in str(excinfo.value)
    assert "Places I looked" in str(excinfo.value)


# ------------------------------------------------------- real-file integration
def test_real_course_file_parses():
    """If the project's real syllabus file exists, make sure it parses."""
    try:
        target = find_course_file("computer_networks.md")
    except FileNotFoundError:
        pytest.skip("real computer_networks.md not present in this checkout")
    result = parse_course_file(target)
    assert isinstance(result["chapters"], dict)
    assert len(result["chapters"]) > 0