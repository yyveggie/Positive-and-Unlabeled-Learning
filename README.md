# Positive-Unlabeled Learning: Experiment Code

Reference implementations and the experimental pipeline for a study of
positive-unlabeled (PU) learning, covering risk estimators, class-prior
estimators, propensity models beyond SCAR, and evaluation under PU-only model
selection.

Twenty-seven method implementations are exercised across twenty-one experiment
tracks. Every track runs under a common information boundary: model selection
uses only positive and unlabeled data, and test labels stay sealed until final
scoring.

## Results

Every row records the dataset, labeling mechanism, labeled-positive
availability, seed, method, hyperparameters, the supplied and true class prior,
the PU-computable selection criterion, and the test metrics. Adverse outcomes
are retained rather than dropped: degenerate one-class predictions, non-finite
values and failed fits all appear in the tables.

Cluster-aware analysis treats the dataset as the independent unit, through
dataset-level exact tests, a hierarchical bootstrap that resamples datasets
before label frequencies and seeds, and mixed-effects sensitivity models.

## Implementation fidelity

`reproducibility/method_implementation_map.csv` is the authoritative record for
each method. It states the implementation level, from a direct transcription of
the published equation through a loss-level port to an adapter around the
authors' released code, the source file and symbol, the upstream component when
one is used, and every deviation from the original.

`reproducibility/upstream_sources.csv` pins the third-party repositories and
`reproducibility/patches/` holds the compatibility patches applied to them.
`reproducibility/compute_environment.json` and
`reproducibility/requirements-lock.txt` record the environment that produced
the results.

## Data not tracked here

Derived features and third-party checkouts are excluded because of their size.
`experiments/src/make_features.py` rebuilds the frozen encoder features and the
graph inputs, and `reproducibility/build_split_seed_manifests.py` regenerates
the manifests that fix the case-control construction.
