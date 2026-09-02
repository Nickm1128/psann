#!/usr/bin/env python
"""Audit tracked outputs and long Python files for repo hygiene.

Example:
  python tools/repo_hygiene_audit.py
  python tools/repo_hygiene_audit.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = {"reports", "runs", "outputs", "logs"}
PRIVATE_PATH_PREFIXES = (".psann-dev/", "docs/internal/", "docs/archive/", "docs/backlog/")
INTERNAL_PLAN_PATHS = {
    "benchmarks/lm_plan.md",
    "docs/HISSO_logging_spec.md",
    "docs/attention_api_plan.md",
    "docs/extras_removal_inventory.md",
    "docs/geosparse_vs_relu_colab_notebook_todo.md",
    "docs/hisso_optimization_todo.md",
    "docs/lsm_robustness_todo.md",
    "docs/phase1_audit.md",
    "docs/project_cleanup_todo.md",
    "docs/repo_hygiene_audit.md",
    "docs/repo_hygiene_followups.md",
    "docs/repo_hygiene_waves.md",
}
PROVENANCE_RULES = {
    "codex": r"\bcodex\b",
    "chatgpt": r"\bchatgpt\b",
    "claude": r"\bclaude\b",
    "copilot": r"\bcopilot\b",
    "gpt-5 family": r"\bgpt[ -]?5(?:\.\d+)?\b",
    "AI-authored/generated claim": r"\bai[- ](?:authored|generated)\b",
    "agent-authored/generated claim": r"\bagent[- ](?:authored|generated)\b",
}
PROVENANCE_EXCLUSIONS = {"tools/repo_hygiene_audit.py", "tests/test_repo_hygiene_audit.py"}
TEXT_SUFFIXES = {
    "",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
INTERNAL_PROCESS_TOKENS = {
    "audit",
    "backlog",
    "followup",
    "followups",
    "instructions",
    "inventory",
    "next",
    "plan",
    "roadmap",
    "todo",
}
PUBLIC_PLAN_ALLOWLIST: set[str] = set()


@dataclass(frozen=True)
class PathIssue:
    path: str
    reason: str


@dataclass(frozen=True)
class FileLength:
    path: str
    lines: int


@dataclass(frozen=True)
class ProvenanceReference:
    path: str
    line: int
    matched_rule: str


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _filename_tokens(path: PurePosixPath) -> set[str]:
    tokens = path.stem.lower().replace("-", "_").split("_")
    if any(left == "follow" and right == "up" for left, right in zip(tokens, tokens[1:])):
        tokens.append("followup")
    return set(tokens)


def _classify_prohibited_tracked(path_text: str) -> str | None:
    path = PurePosixPath(path_text)
    if path_text.startswith(PRIVATE_PATH_PREFIXES):
        return "private planning, archive, or internal document tracked publicly"
    if path_text in INTERNAL_PLAN_PATHS:
        return "internal plan document tracked publicly"
    if (
        path_text not in PUBLIC_PLAN_ALLOWLIST
        and path.parts
        and path.parts[0] in {"docs", "benchmarks"}
        and INTERNAL_PROCESS_TOKENS.intersection(_filename_tokens(path))
    ):
        return "internal-process filename token in public docs or benchmarks"
    if path_text == "test_outputs.txt":
        return "tracked console/test output should stay local"
    if path.parts and path.parts[0] in GENERATED_DIRS:
        if path_text in {
            "reports/README.md",
            "runs/README.md",
            "outputs/README.md",
            "logs/README.md",
        }:
            return None
        return f"tracked generated artifact under {path.parts[0]}/"
    if path.parts and path.parts[0] == "benchmarks" and path.suffix.lower() == ".zip":
        return "benchmark archive bundles should live outside git"
    return None


def _collect_provenance_references(
    repo_root: Path, tracked_files: Sequence[str]
) -> list[ProvenanceReference]:
    import re

    rules = [
        (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in PROVENANCE_RULES.items()
    ]
    references: list[ProvenanceReference] = []
    for path_text in tracked_files:
        if (
            path_text in PROVENANCE_EXCLUSIONS
            or Path(path_text).suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        path = repo_root / path_text
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            for rule, pattern in rules:
                if pattern.search(line):
                    references.append(
                        ProvenanceReference(path=path_text, line=line_number, matched_rule=rule)
                    )
    return references


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _iter_python_files(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        if path.endswith(".py"):
            yield path


def collect_report(
    repo_root: Path,
    *,
    long_file_threshold: int,
    top_n: int,
) -> dict[str, object]:
    tracked_files = _git_ls_files(repo_root)
    prohibited = [
        PathIssue(path=path, reason=reason)
        for path in tracked_files
        if (reason := _classify_prohibited_tracked(path)) is not None
    ]
    provenance = _collect_provenance_references(repo_root, tracked_files)

    long_python = []
    for path in _iter_python_files(tracked_files):
        full_path = repo_root / path
        if not full_path.exists():
            continue
        long_python.append(FileLength(path=path, lines=_count_lines(full_path)))
    long_python.sort(key=lambda item: (-item.lines, item.path))

    over_threshold = [item for item in long_python if item.lines >= long_file_threshold]
    over_threshold_scripts = [item for item in over_threshold if item.path.startswith("scripts/")]

    return {
        "repo_root": str(repo_root),
        "long_file_threshold": long_file_threshold,
        "prohibited_tracked": [asdict(item) for item in prohibited],
        "provenance_references": [asdict(item) for item in provenance],
        "long_python_files": [asdict(item) for item in long_python[:top_n]],
        "over_threshold": [asdict(item) for item in over_threshold[:top_n]],
        "over_threshold_scripts": [asdict(item) for item in over_threshold_scripts[:top_n]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to inspect (default: this repo).",
    )
    parser.add_argument(
        "--long-file-threshold",
        type=int,
        default=800,
        help="Report Python files with at least this many lines as refactor targets.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum number of long-file entries to print.",
    )
    parser.add_argument(
        "--strict-long-files",
        action="store_true",
        help="Exit non-zero when Python files exceed the line threshold.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    return parser.parse_args()


def _print_text_report(report: dict[str, object]) -> None:
    prohibited = report["prohibited_tracked"]
    provenance = report["provenance_references"]
    over_threshold = report["over_threshold"]

    print("Repo hygiene audit")
    print(f"Repo root: {report['repo_root']}")
    print()

    print("Tracked output violations:")
    if prohibited:
        for item in prohibited:
            print(f"- {item['path']}: {item['reason']}")
    else:
        print("- none")

    print()
    print("Public provenance references:")
    if provenance:
        for item in provenance:
            print(f"- {item['path']}:{item['line']}: {item['matched_rule']}")
    else:
        print("- none")

    print()
    print(f"Python files >= {report['long_file_threshold']} lines:")
    if over_threshold:
        for item in over_threshold:
            print(f"- {item['path']} ({item['lines']} lines)")
    else:
        print("- none")


def main() -> int:
    args = parse_args()
    try:
        report = collect_report(
            args.repo_root.resolve(),
            long_file_threshold=args.long_file_threshold,
            top_n=args.top,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)

    has_prohibited = bool(report["prohibited_tracked"])
    has_provenance = bool(report["provenance_references"])
    has_long_files = bool(report["over_threshold"])
    if has_prohibited or has_provenance or (args.strict_long_files and has_long_files):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
