"""Command-line entry point for implemented stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset, write_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[
        "audit", "split", "features", "feature-qa", "feature-panels", "feature-freeze", "graphs", "simulations", "describe", "develop", "diagnose-core", "diagnose-curves", "diagnose-curve-support", "diagnose-representations", "diagnose-reports", "diagnose-contributions", "diagnose-finalize",
    ])
    parser.add_argument("--config", type=Path, default=Path("decisions.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--block", choices=["H", "C", "Y"])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.stage != "audit" and args.overwrite:
        parser.error("--overwrite is only available for the Stage 1 audit")
    if args.stage == "features" and args.block is None:
        parser.error("--block is required for feature extraction")
    if args.stage != "features" and args.block is not None:
        parser.error("--block is only available for feature extraction")
    if args.device is not None and not (args.stage == "features" and args.block == "Y"):
        parser.error("--device is only available for Y feature extraction")
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
        if args.stage == "features":
            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            if args.block == "Y":
                from .feature_y import run_y

                run_y(root, device=args.device or "cpu")
            else:
                from .features import run_features

                run_features(root, args.block)
            print(f"Extracted {args.block} features; admission remains pending quality review.")
            return 0
        if args.stage == "feature-qa":
            from .feature_quality import run_quality

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            run_quality(root, config)
            print("Feature QA created; reviewer decisions and admission remain pending.")
            return 0
        if args.stage == "feature-panels":
            from .review_panels import run_review_panels

            count = run_review_panels(root)
            print(f"Created {count} outcome-blind visual-review panels.")
            return 0
        if args.stage == "feature-freeze":
            from .feature_freeze import run_freeze

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_freeze(root)
            print(f"Common population frozen: {report['status']}; admitted: {report['admitted_blocks']}.")
            return 0
        if args.stage == "graphs":
            from .graphs import run_graphs

            report = run_graphs(root, config)
            print(f"Stage 4: {report['status']}; common population: {report['common_population_status']}.")
            return 0 if report["status"] == "graphs_and_masks_created" else 1
        if args.stage == "simulations":
            from .simulations import run_simulations

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_simulations(root, config["seeds"]["simulation"])
            print(f"Known-truth fits: {report['fit_count']}; failed fits: {report['failed_fits']}.")
            return 1 if report["failed_fits"] else 0
        if args.stage == "describe":
            from .descriptive import run_descriptive

            report = run_descriptive(root, config)
            print(f"Development summaries: {report['development_responses']} responses; "
                  f"{report['development_pairs']} unordered pairs.")
            return 0
        if args.stage == "develop":
            from .development import run_development

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_development(root, config)
            print(f"Development selection: {report['selected_representation']}; "
                  f"{report['candidate_count']} candidates attempted.")
            return 0
        if args.stage == "diagnose-core":
            from .diagnostics import run_core

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_core(root)
            print(f"Stage 8 core control: {report['status']}.")
            return 0 if report["status"] == "passed_operational_control" else 1
        if args.stage == "diagnose-curves":
            from .learning_curves import run_curves

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_curves(root, config)
            print(f"Learning curves: {report['status']}.")
            return 0
        if args.stage == "diagnose-curve-support":
            from .learning_curve_support import run_curve_support

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_curve_support(root)
            print(f"Learning-curve support: {report['status']}; {report['context_rows']} category records.")
            return 0
        if args.stage == "diagnose-representations":
            from .representation_diagnostics import run_representation_diagnostics

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_representation_diagnostics(root, config)
            print(f"Representation diagnostics: {report['status']}.")
            return 0
        if args.stage == "diagnose-reports":
            from .diagnostic_reports import run_reports

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_reports(root)
            print(f"Validation diagnostics: {report['status']}.")
            return 0
        if args.stage == "diagnose-contributions":
            from .contributions import run_contributions

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            count = run_contributions(root)
            print(f"Recorded {count} grouped model-logit contributions.")
            return 0
        if args.stage == "diagnose-finalize":
            from .diagnostic_finalize import run_finalize

            if not config.get("real_data_modeling_allowed"):
                raise ValueError("Stage 0 governance gate has not been opened.")
            report = run_finalize(root)
            print(f"Stage 8: {report['status']}; internal-test outcomes not read.")
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
