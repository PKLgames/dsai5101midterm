"""Summarize verified Cat + CNN evidence and create presentation figures."""

import csv
import json
import os
import tempfile

from model import HERE
from study import digest, load_frozen, write_csv, write_json


def read(name):
    with (HERE/name).open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def run():
    if any((HERE/name).exists() for name in ("learning_curves.csv","presentation_cat_cnn.pdf","RESULTS.md","summary.json")):
        raise FileExistsError("Final study outputs already exist.")
    verification=json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
    penalty=json.loads((HERE/"penalty_provenance.json").read_text(encoding="utf-8"))
    if verification["status"]!="passed" or penalty["status"]!="all_candidates_reproduced":
        raise ValueError("Study verification or penalty reproduction did not pass.")
    curves=[]
    for seed in (20261030,20261031):
        for size in (25,50,75,100):
            name=f"curve_{seed}_{size:03d}";model=load_frozen(name)
            curves.append({"seed":seed,"fraction":size/100,"fitting_responses":model["fit_responses"],
                           "fitting_pairs":model["fit_pairs"],"fitting_photos":model["fit_photos"],"fitting_cats":model["fit_cats"],
                           "validation_responses":model["validation_responses"],"validation_pairs":model["validation_pairs"],
                           "cat_ridge":model["selected"]["cat_ridge"],"cnn_ridge":model["selected"]["cnn_ridge"],
                           "validation_loss":model["selected"]["validation_loss"]})
    write_csv(HERE/"learning_curves.csv",curves,list(curves[0]))
    g2,g3=read("g2_results.csv"),read("g3_results.csv")
    contrast={"g2":read("g2_contrasts.csv"),"g3":read("g3_contrasts.csv")}
    util=read("g2_abstention_utility.csv"); paths=read("penalty_refits.csv")
    plot(curves,paths,{"g2":g2,"g3":g3},util)
    chosen={name:next(r for r in results if r["model"]=="Cat+CNN") for name,results in (("g2",g2),("g3",g3))}
    lines=["# Retrospective Cat + CNN additive scorer: evidence",
           "","This is a post-test, noncausal, nonconfirmatory pooled cat-identity + CNN procedure. "
           "The original G2/G3 outcomes were previously exposed; the original frozen model and official deployment are unchanged.",
           "","## Model and matching","",
           "The displayed-A logit is beta + cat(A)-cat(B) + w·(CNN(A)-CNN(B)). Separate positive ridge "
           "penalties for cat and CNN blocks were chosen from 49 fixed candidates on each split's validation rows. "
           "The original G2 and separate G3 fitting-only CNN preprocessing was refitted on unique fitting photos. "
           "Cat and CNN terms can represent overlapping information; penalty choices determine their allocation, "
           "so coefficient magnitudes are not a causal or behavioral decomposition. "
           "The frozen protocol's blanket claim of additional cat/CNN nonidentifiability is overstated: "
           "an outcome-blind G2 design check found rank 813 of 814, consistent with only the usual cat-offset gauge. "
           "See [rank check and protocol erratum](design_rank_and_protocol_erratum.json).",
           "","| Target | Responses / pairs | Resampling clusters | Cat + CNN loss | Brier | Accuracy |",
           "|---|---|---|---:|---:|---:|"]
    for name in ("g2","g3"):
        r=chosen[name];lines.append(f"| {name.upper()} | {r['responses']} / {r['pairs']} | {r['clusters']} {r['cluster_unit']} | "
                                    f"{float(r['log_loss']):.6f} | {float(r['brier']):.6f} | {100*float(r['accuracy']):.2f}% |")
    lines+=["","**G4: unsupported by explicit user choice.** No unseen-cat identity score was assigned, and no G4 predictions or outcomes were used. "
            "The CNN block could score the photo, but that is not the specified joint model. See [G4 status](g4_unsupported.json).",
            "","## Matched paired differences","",
            "Reference minus Cat+CNN: positive favors the hybrid. 95% pointwise intervals are unadjusted "
            "and conditional on fitted models/selection. G2 resamples unordered pairs; G3 resamples held-out photos.",
            "","| Target | Reference | Difference | 95% interval |","|---|---|---:|---|"]
    for target in ("g2","g3"):
        for r in contrast[target]:
            if r["reference"] in ("original:common:M1","original:M1","original:M3-C","original:M4-C"):
                lines.append(f"| {target.upper()} | {r['reference']} | {float(r['difference']):+.6f} | "
                             f"[{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}] |")
    lines+=["","The hybrid improves point loss over CNN-only on both retrospective splits, with the displayed "
            "exploratory intervals above zero. The incremental comparison against cat-only M1 is unresolved "
            "on both splits. Thus it is not established that CNN adds predictive value once cat identity is available. "
            "Comparison against contextual M4-C changes both context and model inputs. The separate "
            "[kernel/demographic comparisons](additional_paired_contrasts.csv) also use different inputs/tuning.",
            "","## Controls, reliability and comparison type","",
            "The fixed selected procedure passed 20 fair-coin fitting/validation repetitions: zero failures and 0/20 improvements "
            "over log(2). This is an operational control, not a leakage-free certificate. "
            "[Validation](validation_support_slices.csv) and [test](test_support_slices.csv) support/quality slices use "
            "original M3-C definitions; ten-bin reliability is in the corresponding support-calibration tables. "
            "Low support is flagged, and quality groups overlap.",
            "","| Target / type | Responses / pairs | Cat+CNN log loss | Accuracy |",
            "|---|---|---:|---:|"]
    for target in ("g2","g3"):
        for r in read(f"{target}_subgroups.csv"):
            lines.append(f"| {target.upper()} {r['comparison_type']} | {r['responses']} / {r['pairs']} | "
                         f"{float(r['log_loss']):.6f} | {100*float(r['accuracy']):.2f}% |")
    lines+=["","Within-cat, the cat identity score difference is exactly zero; only the CNN difference and "
            "intercept contribute. [G2 subgroup contrasts](g2_subgroup_contrasts.csv) and "
            "[G3 within-cat contrasts](g3_within_contrasts.csv) are post-test, unadjusted subgroup analyses. "
            "They do not support claiming a general causal photo effect.",
            "","## Learning curves, penalty path and temperature","",
            "Every nested subset refits unique-fitting-photo CNN preprocessing, cat keys and all 49 penalty "
            "pairs; common validation support is 1,836 responses / 589 pairs. The 100% seeds have identical "
            "fitting/validation populations and are not independent replicates. The "
            "[penalty replay](penalty_refits.csv) independently reconstructs all 49 candidates. "
            "Temperature selected on validation was T=1, so scaled and unscaled outputs coincide.",
            "","| Seed | Fraction | Fit responses / pairs / photos | Cat ridge | CNN ridge | Validation loss |",
            "|---|---:|---|---:|---:|---:|"]
    for r in curves:
        lines.append(f"| {r['seed']} | {r['fraction']:.2f} | {r['fitting_responses']} / {r['fitting_pairs']} / {r['fitting_photos']} | "
                     f"{r['cat_ridge']:g} | {r['cnn_ridge']:g} | {r['validation_loss']:.6f} |")
    lines+=["","## Fixed-grid abstention","",
            "Correct +1, incorrect -1, abstain 0; utility denominator is all 3,733 original responses. "
            "No cutoff is selected from the test.",
            "","| Confidence threshold | Retained responses / pairs | Coverage | Retained accuracy | Utility / original response |",
            "|---:|---|---:|---:|---:|"]
    for t in (.5,.6,.7):
        r=next(r for r in util if r["model"]=="Cat+CNN" and float(r["threshold"])==t)
        lines.append(f"| {t:.2f} | {r['retained_responses']} / {r['retained_pairs']} | {100*float(r['coverage']):.2f}% | "
                     f"{100*float(r['selective_accuracy']):.2f}% | {float(r['utility_per_original_response']):.6f} |")
    lines+=["","Higher retained accuracy does not establish improved total utility. See "
            "[fixed-grid results](g2_abstention_utility.csv) and [abstention chart](presentation_abstention.pdf).",
            "","## Verification and limitations","",
            "Four focused tests passed. The [verification record](verification.json) independently reconstructs "
            "10 selected G2/G3/curve fits, validates fitting-only preprocessing and gradients, replays "
            "5,000 cluster-bootstrap draws per tested target, and checks 84 abstention cells. "
            "Prior outcome exposure, post-test motivation, model-selection uncertainty, shared images/cats/respondents "
            "and conditional 95% pointwise unadjusted intervals prevent independent confirmation or causal claims.",
            ""]
    (HERE/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    names=("learning_curves.csv","presentation_cat_cnn.pdf","RESULTS.md")
    write_json(HERE/"summary.json",{"status":"complete_retrospective_exploratory",
               "verification_sha256":digest(HERE/"verification.json"),
               "outputs_sha256":{name:digest(HERE/name) for name in names},
               "g4":"unsupported by user choice","primary_claims_changed":False,"deployment_changed":False})
    print("Finalized Cat + CNN synthesis, curves and presentation figure.")


def plot(curves,paths,results,utility):
    with tempfile.TemporaryDirectory(prefix="mpl_",dir=HERE) as config:
        os.environ["MPLCONFIGDIR"]=config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(HERE/"presentation_cat_cnn.pdf") as pdf:
            fig,axes=plt.subplots(1,2,figsize=(13,7))
            for ax,target in zip(axes,("g2","g3")):
                preferred=("Cat+CNN","original:common:M1","original:M1","original:M3-C","original:M4-C")
                subset=[r for r in results[target] if r["model"] in preferred]
                for i,r in enumerate(subset):
                    ax.hlines(i,float(r["ci_low"]),float(r["ci_high"]),color="#28677c")
                    ax.plot(float(r["log_loss"]),i,"o",color="#28677c")
                ax.set_yticks(range(len(subset)),[r["model"] for r in subset]);ax.invert_yaxis()
                ax.set(title=target.upper(),xlabel="Response-weighted log loss")
            fig.suptitle("Cat identity + CNN: matched retrospective comparison",fontsize=15)
            fig.text(.05,.07,"G4 unsupported by user choice. Pointwise intervals: G2 pair, G3 held-out photo.",fontsize=9)
            fig.text(.05,.04,"Post-test exploratory, not causal or independent; comparisons with M4-C also change context.",fontsize=9)
            fig.subplots_adjust(left=.12,right=.97,top=.88,bottom=.18,wspace=.50)
            pdf.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(1,2,figsize=(13,7))
            for seed in (20261030,20261031):
                group=[r for r in curves if r["seed"]==seed]
                axes[0].plot([r["fitting_responses"] for r in group],[r["validation_loss"] for r in group],"o-",label=str(seed))
            axes[0].set(title="Nested learning curves",xlabel="Fitting responses",ylabel="Common-subset validation log loss");axes[0].legend()
            for cr in (.0001,.001,.01):
                group=sorted([r for r in paths if float(r["cat_ridge"])==cr],key=lambda r:float(r["cnn_ridge"]))
                axes[1].plot([float(r["cnn_ridge"]) for r in group],[float(r["validation_loss"]) for r in group],"o-",label=f"cat ridge {cr:g}")
                axes[1].plot([float(r["cnn_ridge"]) for r in group],[float(r["training_loss"]) for r in group],"--",alpha=.5)
            axes[1].set_xscale("log");axes[1].set(title="Penalty paths (selected cat-ridge slices)",xlabel="CNN ridge",ylabel="Log loss");axes[1].legend(fontsize=8)
            fig.suptitle("Fit-only preprocessing; 1,836 common validation responses",fontsize=14)
            fig.text(.05,.025,"Full 7×7 grid is recorded; 100% seed populations coincide.",fontsize=9)
            fig.tight_layout(rect=(0,.06,1,.95));pdf.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(1,2,figsize=(13,7));bins=read("g2_calibration.csv")
            for row in bins:
                if row["mean_probability"]:
                    axes[0].scatter(float(row["mean_probability"]),float(row["observed_a_rate"]),
                                    facecolors="#28677c" if row["support"]=="supported" else "none",edgecolors="#28677c")
            axes[0].plot([0,1],[0,1],"--",color="gray");axes[0].set(title="G2 calibration",xlabel="Mean predicted A",ylabel="Observed A",xlim=(0,1),ylim=(0,1))
            for name in ("Cat+CNN","original:common:M1","original:M3-C","original:M4-C"):
                group=[r for r in utility if r["model"]==name]
                axes[1].plot([float(r["threshold"]) for r in group],[float(r["utility_per_original_response"]) for r in group],label=name)
            axes[1].set(title="Fixed +1/-1/0 utility",xlabel="Confidence threshold",ylabel="Utility / original response");axes[1].legend(fontsize=8)
            fig.suptitle("Saved G2 outcomes: calibration and abstention",fontsize=14)
            fig.text(.05,.025,"Hollow calibration bins are low support. No threshold or deployment model selected.",fontsize=9)
            fig.tight_layout(rect=(0,.06,1,.95));pdf.savefig(fig);plt.close(fig)
            fig,ax=plt.subplots(figsize=(12,7));ax.axis("off")
            texts=["Predictive association, not causal effect.","All G2/G3 outcomes were previously exposed; no independent confirmation.",
                   "Cat and CNN coefficients depend on penalties; do not treat as a causal decomposition.",
                   "G3 uses separate fitting and validation; G4 is unsupported by explicit user choice.",
                   "95% pointwise intervals are unadjusted and omit some respondent/shared-image dependence.",
                   "No original claim, model, report, package or official prediction was changed."]
            fig.suptitle("Interpretation and safeguards",fontsize=15)
            for i,text in enumerate(texts):ax.text(.05,.85-i*.13,"- "+text,fontsize=11,transform=ax.transAxes)
            pdf.savefig(fig);plt.close(fig)


if __name__=="__main__":
    run()
