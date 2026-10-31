"""Outcome-blind rank check and explicit correction to the frozen protocol."""

import numpy as np
from threadpoolctl import threadpool_limits

from model import HERE, design, load_features
from study import digest, load_frozen, load_rows, write_json


def run():
    path=HERE/"design_rank_and_protocol_erratum.json"
    if path.exists():raise FileExistsError("Design-rank record already exists.")
    model=load_frozen("g2");rows=load_rows("fit")
    with threadpool_limits(limits=1):
        x=design(rows,load_features(),model["spec"])
        gram=(x.T@x).toarray()
        eigenvalues=np.linalg.eigvalsh(gram)
    tolerance=float(eigenvalues[-1]*1e-12)
    rank=int(np.sum(eigenvalues>tolerance))
    expected=x.shape[1]-1
    result={"status":"completed","partition":"original_common_fitting_structure_and_features_only",
            "fitting_responses":len(rows),"design_columns":x.shape[1],"cats":len(model["spec"]["cats"]),
            "cnn_dimensions":len(model["spec"]["preprocessing"]["final_names"]),
            "gram_relative_eigenvalue_tolerance":1e-12,"gram_absolute_tolerance":tolerance,
            "numerical_rank":rank,"rank_after_one_cat_offset_gauge":expected,
            "smallest_retained_gram_eigenvalue":float(eigenvalues[-rank]),
            "largest_gram_eigenvalue":float(eigenvalues[-1]),
            "interpretation":"At this tolerance the G2 design has exactly the expected one-dimensional cat-score offset gauge; after fixing that gauge no further exact rank deficiency is detected. Cat/CNN predictors can still be correlated and ridge changes fitted coefficients. This rank check is not a causal identification or cross-split guarantee.",
            "protocol_erratum":"The frozen protocol states that cat and CNN score components are not uniquely identified by the likelihood. That blanket statement overstates the evidence for the G2 design: after the single cat-offset gauge, this numerical rank check detects no additional exact deficiency. Do not claim a fundamental cat/CNN nonidentifiability beyond the gauge from that statement. Interpret coefficient allocation cautiously because penalties and correlations affect fitted values, not because an additional exact deficiency was established.",
            "g2_freeze_sha256":digest(HERE/"g2_freeze.json"),"outcomes_read_for_rank":False}
    write_json(path,result)
    print(result["protocol_erratum"])
    print(f"Rank {rank} of {x.shape[1]}; expected after cat-offset gauge {expected}.")


if __name__=="__main__":
    run()
