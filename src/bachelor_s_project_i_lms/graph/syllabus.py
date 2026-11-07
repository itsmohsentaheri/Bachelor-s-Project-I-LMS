import re
from pathlib import Path

CHAPTER_RE = re.compile(r"^(\d+)\s*[:.)-]\s*(.+)$")
KV_RE = re.compile(r"^([A-Za-z][^:]*?)\s*:\s*(.+)$")

def load_roots(path: str) -> dict:
    course = {"chapters": {}}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = re.sub(r"^[#*\->\s]+", "", raw_line).strip()
        if not line:
            continue

        if (m := CHAPTER_RE.match(line)):
            num, title = int(m.group(1)), m.group(2).strip()
            course["chapters"][num] = title

        elif (m := KV_RE.match(line)):
            key = m.group(1).strip().lower()
            course[key] = m.group(2).strip()

    return course