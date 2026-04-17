"""Command-line interface for the Negative Result Repository.

Primary entry point for autoresearch loop integration. Reads a proposed
experiment description (or unified diff) from a file, queries the NRR
database, and prints a JSON verdict to stdout.

Usage:
    uv run nrr check --proposal-file /tmp/proposal.txt
    uv run python -m nrr.cli check --proposal-file /tmp/proposal.txt

Output (stdout, single JSON object):
    {
      "verdict": "proceed" | "caution" | "avoid",
      "reason": "...",
      "similar_failures": [ ... ],
      "relevant_patterns": [ ... ],
      "estimated_success_probability": 0.42
    }

Exit codes:
    0  proceed
    1  caution
    2  avoid
    3  usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from nrr.repository import NegativeResultRepository


# Default DB locations searched in order. The first one that exists wins.
def _default_db_paths() -> list[Path]:
    """Candidate database paths, in priority order.

    1. $NRR_DATABASE if set
    2. ./data/nrr_database.json relative to the NRR package install
    3. <package_dir>/../../data/nrr_database.json (repo layout)
    4. ./data/results.tsv (will be parsed on the fly)
    """
    candidates: list[Path] = []
    env = os.environ.get("NRR_DATABASE")
    if env:
        candidates.append(Path(env))

    # Repo-layout fallback: src/nrr/cli.py -> repo_root/data/...
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    candidates.append(repo_root / "data" / "nrr_database.json")
    candidates.append(repo_root / "data" / "results.tsv")
    return candidates


def _load_repo() -> tuple[NegativeResultRepository, Path]:
    """Load the NRR from the first available default location."""
    last_err: Exception | None = None
    for path in _default_db_paths():
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                return NegativeResultRepository.from_json(path), path
            if path.suffix == ".tsv":
                return NegativeResultRepository.from_tsv(path), path
        except Exception as e:  # pragma: no cover - defensive
            last_err = e
            continue
    msg = (
        "No NRR database found. Set $NRR_DATABASE or place a "
        "data/nrr_database.json next to the package."
    )
    if last_err is not None:
        msg += f" Last error: {last_err!r}"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Proposal parsing
# ---------------------------------------------------------------------------

# Match `KEY = value` style python assignments. Tolerates trailing comments.
_ASSIGN_RE = re.compile(
    r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)(?:\s*#.*)?$",
    re.MULTILINE,
)

# Names worth surfacing into config_changes. Same set NRR's parser knows about,
# plus some autoresearch / nanochat-flavored extras.
_KNOWN_PARAMS = {
    "BATCH_SIZE",
    "LEARNING_RATE",
    "LR",
    "WEIGHT_DECAY",
    "NUM_EPOCHS",
    "OPTIMIZER",
    "LR_SCHEDULER",
    "DROPOUT",
    "NUM_FILTERS_1",
    "NUM_FILTERS_2",
    "NUM_FILTERS_3",
    "FC_SIZE",
    "USE_BATCHNORM",
    "ACTIVATION",
    "USE_HORIZONTAL_FLIP",
    "USE_RANDOM_CROP",
    "USE_COLOR_JITTER",
    # autoresearch (nanochat-style)
    "DEPTH",
    "WIDTH",
    "N_HEADS",
    "EMBED_DIM",
    "SEQ_LEN",
}


def _extract_config_changes(text: str) -> dict[str, str]:
    """Best-effort parse of `KEY = value` lines from a diff or plain text.

    Strips leading +/- characters from unified-diff lines first so that
    `+LEARNING_RATE = 0.1` is picked up.
    """
    changes: dict[str, str] = {}
    cleaned_lines = []
    for raw_line in text.splitlines():
        if raw_line.startswith(("+++", "---", "@@")):
            continue
        # Strip a single leading +/- (unified diff marker) so the regex sees
        # the original assignment.
        if raw_line[:1] in ("+", "-"):
            cleaned_lines.append(raw_line[1:])
        else:
            cleaned_lines.append(raw_line)
    cleaned = "\n".join(cleaned_lines)

    for m in _ASSIGN_RE.finditer(cleaned):
        key = m.group(1).strip()
        if key not in _KNOWN_PARAMS:
            continue
        value = m.group(2).strip().strip("\"'")
        # Last write wins (proposals usually only set each key once).
        changes[key] = value
    return changes


def _description_from_text(text: str, max_chars: int = 500) -> str:
    """Pull a short, human-readable description out of a proposal file.

    For a unified diff we use the file header line if present; otherwise
    we use the first non-empty line of the file.
    """
    text = text.strip()
    if not text:
        return ""

    # Prefer the first plain (non-diff-marker) line as the gist.
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        return s[:max_chars]

    return text[:max_chars]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    proposal_file = Path(args.proposal_file)
    if not proposal_file.exists():
        print(
            json.dumps({"error": f"proposal file not found: {proposal_file}"}),
            file=sys.stderr,
        )
        return 3
    try:
        text = proposal_file.read_text()
    except OSError as e:
        print(json.dumps({"error": f"cannot read proposal: {e!r}"}), file=sys.stderr)
        return 3

    description = args.description or _description_from_text(text)
    config_changes = _extract_config_changes(text) or None

    try:
        repo, db_path = _load_repo()
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 3

    result = repo.check_proposal(description=description, config_changes=config_changes)

    # Map repo's "recommendation" to "verdict" for a stable CLI contract.
    verdict = result["recommendation"]
    payload = {
        "verdict": verdict,
        "reason": result["reason"],
        "similar_failures": result["similar_failures"],
        "relevant_patterns": result["relevant_patterns"],
        "estimated_success_probability": result["estimated_success_probability"],
        "parsed_description": description,
        "parsed_config_changes": config_changes or {},
        "database": str(db_path),
    }
    print(json.dumps(payload, indent=2, default=str))

    return {"proceed": 0, "caution": 1, "avoid": 2}.get(verdict, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nrr",
        description="Negative Result Repository CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check",
        help="Check a proposed experiment against the failure database.",
    )
    p_check.add_argument(
        "--proposal-file",
        required=True,
        help="Path to a file containing a proposal description or unified diff.",
    )
    p_check.add_argument(
        "--description",
        default=None,
        help=(
            "Optional explicit description. If omitted, inferred from the "
            "proposal file."
        ),
    )
    p_check.set_defaults(func=_cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
