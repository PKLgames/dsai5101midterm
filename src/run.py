"""Command-line entry point for implemented stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset, write_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["audit", "split"])
    parser.add_argument("--config", type=Path, default=Path("decisions.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "split" and args.overwrite:
        parser.error("--overwrite is not allowed for an existing split")
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("primary_target") != "strict_G2":
            raise ValueError("Config must register strict_G2 as the primary target.")
        if args.stage == "split":
            from .split import run_split

            rows, pairs, reused = run_split(root, config)
            print(f"Split {'retained' if reused else 'created'}: {rows} rows, {pairs} pairs.")
            print("Internal-test outcomes are masked in canonical rows until evaluation.")
            return 0
        result = audit_dataset(root)
        outputs = write_audit(root, result, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"{args.stage.capitalize()} failed: {exc}", file=sys.stderr)
        return 2
    for path in outputs:
        print(f"Wrote {path}")
    errors = sum(issue["severity"] == "error" for issue in result.issues)
    warnings = sum(issue["severity"] == "warning" for issue in result.issues)
    print(f"Audit complete: {errors} blocking issues; {warnings} warnings.")
    if result.blocked:
        print("Fitting is blocked until audit errors are resolved.", file=sys.stderr)
        return 1
    print("Stage 0 governance and later-stage gates remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
