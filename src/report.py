"""Presentation-ready Group 19 report from saved evidence, without refitting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from .development import _hash
from .reproduce import verify_history

TITLE = "Predicting to Diagnose"
SUBTITLE = "Cat identity, visual representations, and context in cat-photo preferences"


def _csv(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _one(rows: list[dict[str, str]], **selector: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in selector.items())]
    if len(matches) != 1:
        raise ValueError(f"Report selector does not identify exactly one row: {selector}")
    return matches[0]


def _wrap(value: str, width: int = 80) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in value.splitlines())


def _page(title: str, speaker: str, source: str) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="#fbfcfd")
    ax = fig.add_axes([.055, .15, .89, .67])
    ax.set_axis_off()
    fig.text(.055, .90, title, fontsize=24, weight="bold", color="#16384c")
    fig.text(.055, .845, "Group 19 | " + speaker, fontsize=11, color="#456475")
    fig.text(.055, .07, source, fontsize=7.5, color="#516570")
    return fig, ax


def _bullets(ax: Any, entries: list[str], fontsize: int = 18, width: int = 86) -> None:
    y = .96
    for entry in entries:
        wrapped = _wrap(entry, width)
        ax.text(0, y, "- " + wrapped, fontsize=fontsize, va="top", color="#223440",
                linespacing=1.35, transform=ax.transAxes)
        y -= .092 * (wrapped.count("\n")+1) + .065


def _table(ax: Any, columns: list[str], rows: list[list[str]], widths: list[float] | None = None,
           size: int = 14) -> None:
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left",
                     colLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1, 2.4)
    for key, cell in table.get_celld().items():
        row = key[0]
        cell.set_edgecolor("#d8e2e8")
        if row == 0:
            cell.set_facecolor("#16384c")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#edf3f7" if row % 2 else "white")


def generate_report(root: Path, output: Path | None = None) -> dict[str, Any]:
    output = output or root / "Group_19_Report.pdf"
    provenance = root / "artifacts/stage11_report_provenance.json"
    if output.exists():
        raise FileExistsError(f"Report already exists; choose a different --output: {output}")
    verify_history(root)
    reproduction = json.loads((root / "artifacts/reproduction_statistical.json").read_text(encoding="utf-8"))
    if reproduction["status"] != "passed":
        raise ValueError("Fresh-process statistical reproduction must pass before report generation.")
    for block in ("H", "C", "Y"):
        check = json.loads((root / "artifacts" / f"feature_regeneration_{block}.json").read_text(encoding="utf-8"))
        if check["status"] != "passed":
            raise ValueError(f"Fixed-subset feature regeneration has not passed: {block}")
    decisions = json.loads((root / "artifacts/stage11_decisions.json").read_text(encoding="utf-8"))
    contributions = _csv(root, "tables/stage11_contributions_draft.csv")
    if {row["member"] for row in contributions} != set(decisions["presenter_allocation"]):
        raise ValueError("Contribution placeholder and confirmed presenter roster differ.")
    main = _csv(root, "tables/main_results.csv")
    primary = _csv(root, "tables/primary_contrasts.csv")
    secondary = _csv(root, "tables/secondary_results.csv")
    validation = _csv(root, "tables/validation_results.csv")
    support = _csv(root, "tables/split_coverage.csv")
    demographic = _csv(root, "tables/exploratory_demographic_semantic_rates.csv")
    cache = _csv(root, "tables/feature_quality.csv")
    selection = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))
    facts: list[dict[str, Any]] = []
    def cite(name: str, selector: dict[str, str], values: dict[str, Any]) -> None:
        facts.append({"source": name, "sha256": _hash(root / name), "selector": selector, "values": values})

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    checks: list[dict[str, Any]] = []
    page_number = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage11-report-") as directory:
        temporary = Path(directory)
        main_pdf = temporary / "main.pdf"
        def save(pdf: Any, fig: Any, topic: str) -> None:
            nonlocal page_number
            page_number += 1
            fig.text(.94, .055, str(page_number), fontsize=9, ha="right", color="#516570")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            from matplotlib.text import Text
            clipped = []
            suppressed = set()
            for axis in fig.axes:
                left, right = sorted(axis.get_xlim())
                bottom, top = sorted(axis.get_ylim())
                suppressed.update(label for value, label in zip(axis.get_xticks(), axis.get_xticklabels())
                                  if value < left or value > right)
                suppressed.update(label for value, label in zip(axis.get_yticks(), axis.get_yticklabels())
                                  if value < bottom or value > top)
            for artist in fig.findobj(Text):
                if artist.get_visible() and artist.get_text() and artist not in suppressed:
                    bounds = artist.get_window_extent(renderer)
                    if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width+2 or bounds.y1 > fig.bbox.height+2:
                        clipped.append(artist.get_text())
            checks.append({"page": page_number, "topic": topic, "outside_page_text": clipped})
            if clipped:
                raise ValueError(f"Report text is outside page bounds: {topic}: {clipped}")
            pdf.savefig(fig)
            plt.close(fig)

        with PdfPages(main_pdf) as pdf:
            fig, ax = _page(TITLE, "Chen Wanchuan", "Scope: strict G2 predictive diagnosis | Frozen internal evaluation + separate exploratory development analysis")
            ax.text(0, .92, _wrap(SUBTITLE, 57), fontsize=25, color="#16384c", va="top")
            ax.text(0, .56, "Chen Wanchuan  |  Yuen Lok Yan  |  Lu Yuqing\nChee Ching Lam  |  Chan Hoi Lam",
                    fontsize=18, va="top", linespacing=1.6)
            ax.text(0, .19, "Working presentation draft\nContribution statement and course reference details require team completion.",
                    fontsize=14, color="#9b4a1f", va="top")
            save(pdf, fig, "Title and group")

            fig, ax = _page("Question and motivation", "Chen Wanchuan", "protocol.md; claims.csv; claims_final.csv")
            _bullets(ax, [
                "Which representations predict unseen matchups between familiar photographs, and what do their gains or failures reveal?",
                "Prediction is a diagnostic instrument: compare identity, shared visual scoring, and recorded context.",
                "Primary target: unseen unordered photo pairs whose endpoints are fitting photos in the same fitting component.",
                "We do not estimate objective attractiveness, causal visual effects, or an information limit.",
            ], fontsize=18, width=84)
            save(pdf, fig, "Research question")

            fig, ax = _page("Data understanding: a response is not a respondent", "Chee Ching Lam",
                                "README.md; tables/data_audit.csv; tables/graph_summary.csv")
            audit = _csv(root, "tables/data_audit.csv")
            tr = _one(audit, partition="train", check="row_count", value="all")["count"]
            official = _one(audit, partition="test", check="row_count", value="all")["count"]
            image_count = _one(audit, partition="images", check="file_count", value="all")["count"]
            for part in ("train", "test", "images"):
                check = "file_count" if part == "images" else "row_count"
                row = _one(audit, partition=part, check=check, value="all")
                cite("tables/data_audit.csv", {"partition": part, "check": check, "value": "all"}, {"count": row["count"]})
            _bullets(ax, [
                f"{int(tr):,} labeled training responses; {int(official):,} unlabeled official-test rows; {int(image_count):,} photographs.",
                "Each row is one response to an ordered photo pair; repeated rows and repeated pairs are not independent respondents.",
                "No respondent identifier is available. Background fields record gender, pet experience/contact, future pet plans, and MBTI I/E.",
                "Raw files stay unchanged; IDs, labels, comparison types and image identities were audited and hashed.",
            ], fontsize=18)
            save(pdf, fig, "Data understanding")

            fig, ax = _page("Preparation and target-aligned split", "Chee Ching Lam",
                                "artifacts/split_manifest.parquet; tables/split_coverage.csv; tables/graph_summary.csv")
            fit_n = json.loads((root / "artifacts/model_selection.json").read_text(encoding="utf-8"))["fitting_responses"]
            graph = _one(_csv(root, "tables/graph_summary.csv"), population="common", entity="photo")
            val_row = _one(support, population="common", partition="validation", comparison_type="all")
            test_row = _one(support, population="common", partition="test", comparison_type="all")
            _table(ax, ["Partition", "Responses", "Pairs", "Purpose"], [
                ["Fitting", f"{fit_n:,}", f"{int(graph['undirected_edges']):,}", "Estimation + preprocessing"],
                ["Validation", f"{int(val_row['responses']):,}", f"{int(val_row['pairs']):,}", "Penalties and model selection"],
                ["Internal test", f"{int(test_row['responses']):,}", f"{int(test_row['pairs']):,}", "One frozen evaluation"],
            ], widths=[.16, .15, .12, .57], size=16)
            fig.text(.06, .20, "Unordered pairs stay together. Shared fitting photos are intentional for G2; no G3 transfer claim.", fontsize=12)
            fig.text(.06, .16, "Full and matched strict-G2 eligibility retain all held-out rows here; connected fitting graphs define support.", fontsize=12)
            cite("tables/split_coverage.csv", {"population": "common", "partition": "validation", "comparison_type": "all"}, val_row)
            cite("tables/split_coverage.csv", {"population": "common", "partition": "test", "comparison_type": "all"}, test_row)
            cite("tables/graph_summary.csv", {"population": "common", "entity": "photo"}, graph)
            save(pdf, fig, "Preparation and split")

            fig, ax = _page("Features: fixed extraction, fitting-only transformations", "Lu Yuqing",
                                "features/manifests/*.json; artifacts/preprocessing/*.json; tables/feature_quality.csv")
            dims = {b: len(json.loads((root / f"artifacts/preprocessing/{b}.json").read_text(encoding="utf-8"))["final_names"])
                    for b in ("H", "C", "Y")}
            _table(ax, ["Block", "Measurement", "Model dimensions"], [
                ["H", "20 low-level image statistics", str(dims["H"])],
                ["C", "Frozen MobileNetV3-Small pooled embedding", str(dims["C"])],
                ["Y", "YOLO masks + HSV measures + CLIP hard labels", str(dims["Y"])],
            ], widths=[.10, .70, .20], size=15)
            fig.text(.06, .23, "Imputation, scaling, category pooling and block normalization use unique fitting photos only.", fontsize=12)
            fig.text(.06, .19, "All blocks passed provenance/coverage gates and the team-recorded image review.", fontsize=12)
            fig.text(.06, .15, "Automated success is not ground truth. Missing measurements and no detection remain explicit.", fontsize=12)
            for b in ("H", "C", "Y"):
                cite("tables/feature_quality.csv", {"block": b}, _one(cache, block=b))
            save(pdf, fig, "Feature construction")

            fig, ax = _page("Methods and justification: a matched model ladder", "Chan Hoi Lam",
                                "protocol.md; decisions.yaml; src/estimator.py; artifacts/model_selection.json")
            _table(ax, ["Model", "Displayed-A logit / prediction", "Question"], [
                ["M0", "Smoothed fitted A frequency", "Constant baseline"],
                ["M1", "side + cat A score - cat B score", "Cat identity"],
                ["M2", "side + photo A score - photo B score", "Photo detail"],
                ["M3-H/C/Y", "side + w * (feature A - feature B)", "Shared visual scoring"],
                ["M4-H/C/Y", "side + (w + group deviation) * feature difference", "Recorded context"],
            ], widths=[.13, .59, .28], size=13)
            fig.text(.06, .15, "Stable logistic loss + ridge; analytic gradients; recorded convergence checks; exact pooled fallback.", fontsize=12)
            save(pdf, fig, "Methods")

            fig, ax = _page("Selection and leakage safeguards", "Chee Ching Lam",
                                "artifacts/model_freeze.json; artifacts/outcome_access_log.csv; artifacts/stage8_core.json")
            _bullets(ax, [
                "Tune on common strict validation; then select the shared visual block and its matched contextual extension.",
                f"Validation selected {selection['selected_representation']} (CNN). H and Y contextual procedures selected their exact pooled fallback.",
                "Saved freeze: models, mappings, preprocessing, masks, diagnostics, Git state and hashes before test outcomes were opened.",
                "Important limitation: previous analyses used these labeled data. This new internal split is not certified pristine.",
            ], fontsize=18)
            save(pdf, fig, "Model selection and safeguards")

            fig, ax = _page("Results: cat identity has the lowest point loss", "Chan Hoi Lam",
                                "tables/main_results.csv | Common frozen strict-G2 test | Lower log loss is better")
            order = ["common:M0", "common:M1", "common:M2", "M3-H", "M3-C", "M3-Y", "M4-C"]
            names = ["Constant", "Cat identity", "Photo identity", "Handcrafted", "CNN", "YOLO-CLIP", "CNN + context"]
            table = []
            for key, label in zip(order, names):
                row = _one(main, population="common", model_id=key)
                table.append([label, f"{float(row['response_log_loss']):.5f}", f"{float(row['brier']):.5f}",
                              f"{float(row['accuracy']):.1%}"])
                cite("tables/main_results.csv", {"population": "common", "model_id": key}, row)
            _table(ax, ["Frozen model", "Log loss", "Brier", "Accuracy"], table,
                   widths=[.46, .18, .18, .18], size=15)
            save(pdf, fig, "Test results")

            fig, ax = _page("Evaluation: four prespecified paired contrasts", "Chan Hoi Lam",
                                "tables/primary_contrasts.csv | Pair-cluster bootstrap, conditional on fitted models")
            ax.set_axis_on()
            ax.set_position([.23, .28, .70, .45])
            labels = ["Cat identity: M0 - M1", "Photo detail: M1 - M2", "Visual sharing: M2 - M3*", "Context: M3* - M4*"]
            ordered = [_one(primary, contrast=name) for name in
                       ("cat_identity", "photo_detail", "visual_sharing", "recorded_context")]
            for i, row in enumerate(ordered):
                point = float(row["response_loss_difference"])
                lo, hi = float(row["response_interval_low"]), float(row["response_interval_high"])
                ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", capsize=5,
                            color="#1c758a", markersize=7)
                cite("tables/primary_contrasts.csv", {"contrast": row["contrast"]}, row)
            ax.set_yticks(range(4), labels, fontsize=12)
            ax.invert_yaxis()
            ax.axvline(0, linestyle="--", color="#666666")
            ax.grid(axis="x", alpha=.2)
            ax.set_xlabel("Reference minus candidate log loss (nats/response); positive favors candidate", fontsize=11)
            fig.text(.06, .18, "Approximate 98.75% pointwise intervals for four primary contrasts; 5,000 shared pair resamples.", fontsize=12)
            fig.text(.06, .14, "Photo detail and visual sharing span no difference; context gain is small and not causal.", fontsize=12)
            save(pdf, fig, "Paired uncertainty")

            fig, ax = _page("Diagnosis: fitting gains need not generalize", "Lu Yuqing",
                                "tables/validation_results.csv; tables/temperature_diagnostic.csv; tables/secondary_results.csv")
            table = []
            for key, label in (("common:M1", "Cat identity"), ("common:M2", "Photo identity"), ("common:M3-C", "CNN")):
                row = _one(validation, model=key)
                table.append([label, f"{float(row['train_loss']):.5f}", f"{float(row['validation_loss']):.5f}"])
                cite("tables/validation_results.csv", {"model": key}, row)
            _table(ax, ["Model", "Fitting loss", "Validation loss"], table, widths=[.46, .27, .27], size=16)
            fig.text(.06, .23, "Photo-specific fitting gains did not become validation gains: consistent with estimation variance.", fontsize=12)
            temp = _one(secondary, contrast="temperature_M2")
            fig.text(.06, .19, f"Preselected temperature scaling improved photo-model test loss by {float(temp['response_loss_difference']):.5f} (secondary).", fontsize=12)
            fig.text(.06, .15, "This supports confidence adjustment as one opportunity, not a unique explanation of remaining loss.", fontsize=12)
            cite("tables/secondary_results.csv", {"contrast": "temperature_M2"}, temp)
            save(pdf, fig, "Underperformance diagnosis")

            fig, ax = _page("Recorded context: small aggregate gain, not a causal effect", "Chan Hoi Lam",
                                "tables/primary_contrasts.csv; tables/secondary_results.csv; tables/subgroup_results.csv")
            context = _one(primary, contrast="recorded_context")
            side = _one(secondary, contrast="context_with_group_side")
            _bullets(ax, [
                f"Matched CNN context gain: {float(context['response_loss_difference']):.6f} nats/response; "
                f"primary interval [{float(context['response_interval_low']):.6f}, {float(context['response_interval_high']):.6f}].",
                f"Allowing group-specific side tendencies gives a similar secondary gain: {float(side['response_loss_difference']):.6f}.",
                "The missing-context subgroup is low support and has a worse contextual point loss; aggregate improvement is not universal.",
                "Pet-care measurements, pair composition, reference coding and dependence remain alternative explanations.",
            ], fontsize=17, width=88)
            cite("tables/secondary_results.csv", {"contrast": "context_with_group_side"}, side)
            save(pdf, fig, "Context interpretation")

            fig, ax = _page("Exploratory T2: semantic choices by demographic group", "Yuen Lok Yan",
                                "tables/exploratory_demographic_semantic_rates.csv; tables/exploratory_demographic_semantic_overlap.csv")
            s1 = {"demographic": "pet_care", "group": "current_regular", "feature": "breed=british_shorthair"}
            s2 = {"demographic": "pet_care", "group": "occasionally", "feature": "breed=british_shorthair"}
            high, low = _one(demographic, **s1), _one(demographic, **s2)
            _bullets(ax, [
                "Development-only, post-evaluation description: count choices when just one endpoint carries a semantic label.",
                f"Example: CLIP-inferred British shorthair chosen {float(high['choice_rate']):.1%} in current-regular-care "
                f"comparisons versus {float(low['choice_rate']):.1%} in occasional-care comparisons.",
                f"These rates use {high['discriminating_responses']} and {low['discriminating_responses']} responses on different pair mixtures.",
                "No shared-pair group comparison passed the registered per-cell support rule. This is not an adjusted demographic preference or causal claim.",
            ], fontsize=17, width=88)
            cite("tables/exploratory_demographic_semantic_rates.csv", s1, high)
            cite("tables/exploratory_demographic_semantic_rates.csv", s2, low)
            save(pdf, fig, "Exploratory semantics")

            fig, ax = _page("Answer, interpretation, and unresolved limitations", "Chen Wanchuan",
                                "claims_final.csv; tables/diagnostic_synthesis.csv; artifacts/model_freeze.json")
            _bullets(ax, [
                "Cat identity is useful for this known-photo unseen-pair target. The photo-detail and visual-sharing primary intervals are inconclusive.",
                "A small matched CNN context gain remains under the conditional bootstrap; it is predictive, not causal.",
                "Negative extensions are evidence: H/Y context fell back to pooling; PCA did not improve the selected CNN validation loss.",
                "No respondent linkage, imperfect semantic measurements, shared-image dependence, and prior analysis exposure limit conclusions.",
            ], fontsize=17, width=88)
            save(pdf, fig, "Answer and limitations")

            fig, ax = _page("Reproducibility and next evidence", "Chen Wanchuan",
                                "artifacts/reproduction_statistical.json; artifacts/feature_regeneration_*.json; REPRODUCTION.md")
            _bullets(ax, [
                f"Fresh-process refits at frozen settings reproduced {reproduction['selected_models_reproduced']} selected models; "
                f"maximum reported test-metric difference {reproduction['max_test_metric_abs_difference']:.2g}.",
                "Fitting-only preprocessing and paired bootstrap intervals were reconstructed; no raw internal-test outcomes were reopened.",
                "Fixed image subsets were regenerated against H/C/Y caches. Observed differences and device/library versions are logged.",
                "Next evidence: independently collected pairs/respondents, stronger shared-pair context support, and validated semantic annotations.",
            ], fontsize=17, width=88)
            cite("artifacts/reproduction_statistical.json", {}, {
                "selected_models_reproduced": reproduction["selected_models_reproduced"],
                "max_test_metric_abs_difference": reproduction["max_test_metric_abs_difference"],
            })
            save(pdf, fig, "Reproduction and next evidence")

            fig, ax = _page("Contributions: TEAM MUST EDIT BEFORE SUBMISSION", "All five members",
                                "tables/stage11_contributions_draft.csv | Placeholder only; not a verified record of individual work")
            rows = [[row["member"], _wrap(row["assigned_responsibility"], 35)] for row in contributions]
            _table(ax, ["Member", "Assigned responsibility (placeholder)"], rows, widths=[.30, .70], size=14)
            fig.text(.06, .21, "Replace this slide with actual tasks and concrete outputs completed by each member.", fontsize=13, color="#9b4a1f")
            fig.text(.06, .17, "AI-assisted implementation and reproduction must be reviewed by the team; assigned ownership is not proof of contribution.", fontsize=11)
            save(pdf, fig, "Contributions placeholder")

            fig, ax = _page("Presentation and submission checklist", "All five members",
                                "Midterm_Report_Guidelines_2026.pdf; artifacts/stage11_decisions.json")
            _bullets(ax, [
                "Proposed substantive presenters: Chen (question/synthesis), Chee (data/evaluation), Lu (features), "
                "Yuen (YOLO-CLIP/semantics), Chan (models/uncertainty). Rehearse to approximately 15 minutes.",
                "Edit the contribution placeholder and complete the course Chapter 7 reference before submission.",
                "Submit Group_19_Report.pdf and Group_19_Code.zip locally prepared for course use only; no upload has been made.",
                "Official deadline: 2 November 2026, 12:00 noon HKT. Submission channel remains unverified.",
            ], fontsize=17, width=88)
            save(pdf, fig, "Submission and presentation")

            for title, lines in (
                ("References: data and methods", [
                    "Cat Image Arena dataset and course README; missing12/cat-image-arena-images, Hugging Face.",
                    "DSAI5101 Statistical Data Mining, Midterm Report Guidelines (2026).",
                    "Bradley & Terry (1952), Rank Analysis of Incomplete Block Designs. Biometrika 39, 324-345.",
                    "Wilson (1927), Probable Inference, the Law of Succession, and Statistical Inference. JASA 22, 209-212.",
                    "Efron & Tibshirani (1993), An Introduction to the Bootstrap. Chapman & Hall.",
                    "Course Chapter 7 materials: complete bibliographic details required from the team.",
                ]),
                ("References: representations and software", [
                    "Howard et al. (2019), Searching for MobileNetV3. ICCV; Torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1.",
                    "Radford et al. (2021), Learning Transferable Visual Models From Natural Language Supervision. ICML.",
                    "OpenCLIP implementation: github.com/mlfoundations/open_clip; ViT-B-32 with OpenAI weights.",
                    "Ultralytics YOLO11n segmentation: docs.ultralytics.com; actual checkpoint hashes are in features/manifests/Y.json.",
                    "NumPy, SciPy, NetworkX, Matplotlib, PyArrow, Pillow, PyTorch/Torchvision, OpenCV, OpenCLIP and pypdf: exact versions in environment locks.",
                    "Data/derived materials: education and research only. Weights and raw photographs are excluded from the code ZIP.",
                ]),
            ):
                fig, ax = _page(title, "All members", "references.csv; environment lock files; feature manifests")
                _bullets(ax, lines, fontsize=13, width=115)
                save(pdf, fig, title)

        writer = PdfWriter()
        reader = PdfReader(main_pdf)
        main_pages = len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
        appendices = [
            "figures/presentation_feature_importance_H.pdf",
            "figures/presentation_feature_importance_Y.pdf",
            "figures/presentation_context_interactions.pdf",
            "figures/simulations_ABC.pdf", "figures/graph_support.pdf",
            "figures/disagreement.pdf", "figures/support_distribution.pdf",
            "figures/penalty_paths.pdf", "figures/learning_curves.pdf",
            "figures/calibration.pdf", "figures/diagnostic_calibration.pdf",
            "figures/exploratory_demographic_semantics.pdf",
            "figures/exploratory_demographic_semantic_support.pdf",
        ]
        appendix_pages = []
        for name in appendices:
            appendix = PdfReader(root / name)
            first = len(writer.pages)+1
            for page in appendix.pages:
                writer.add_page(page)
            appendix_pages.append({"source": name, "start_page": first, "pages": len(appendix.pages)})
        writer.add_metadata({"/Title": TITLE + " - Group 19", "/Author": "Group 19",
                             "/Subject": "Draft statistical data mining report; contribution statement requires team completion"})
        with output.open("xb") as stream:
            writer.write(stream)
    record = {
        "status": "draft_pending_team_contribution_and_reference_completion",
        "report": output.name, "main_pages": main_pages, "total_pages": len(PdfReader(output).pages),
        "appendices": appendix_pages, "layout_checks": checks, "evidence": facts,
        "report_sha256": _hash(output), "source_sha256": _hash(root / "src/report.py"),
        "frozen_model_or_test_outputs_modified": False,
        "contribution_placeholder": True, "submission_channel_verified": False,
    }
    if not provenance.exists():
        provenance.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        record = generate_report(args.root.resolve(), args.output.resolve() if args.output else None)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Created {record['report']}: {record['main_pages']} main pages, {record['total_pages']} total. Contribution placeholder requires editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
