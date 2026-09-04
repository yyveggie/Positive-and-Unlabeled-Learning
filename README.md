# Positive-Unlabeled Learning: Experiment Code

Reference implementations and the full experimental pipeline for a study of
positive-unlabeled (PU) learning, covering risk estimators, class-prior
estimators, propensity models beyond SCAR, and evaluation under PU-only model
selection.

Twenty-seven method implementations are exercised across twenty-one experiment
tracks. Every track is run under a common information boundary: model selection
uses only positive and unlabeled data, and test labels stay sealed until final
scoring.

## Layout

    experiments/src/         method implementations, runners, and summarizers
    experiments/results/     result tables, one CSV per track and shard
    reproducibility/         experiment registry, method map, environment, tests

## Setup

    python -m venv .venv
    .venv/bin/pip install -r reproducibility/requirements-lock.txt

The lock file pins the numerical stack that produced the recorded results.
`reproducibility/compute_environment.json` records the hardware, operating
system and interpreter used, along with the boundary beyond which results are
not expected to reproduce bit for bit.

## Running experiments

`reproducibility/experiment_registry.json` is the authoritative map from track
to runner. Each entry gives the purpose, the runner path and the exact command,
including seed loops where a track needs them. Examples:

    python experiments/src/run_p1.py       # classical and risk-estimator columns
    python experiments/src/run_ext.py      # modern loss-level methods
    python experiments/src/run_dcpu.py     # DC-PU on the shared tabular protocol
    python experiments/src/run_gaps.py prior     # prior sensitivity, five objectives
    python experiments/src/run_gaps.py balance   # classwise error rates

Runners append to `experiments/results/`. The matching `summarize_*.py` module
reduces a track to the tables it reports. `experiments/src/hierarchical_reanalysis.py`
performs the cluster-aware analysis: dataset-level exact tests, a hierarchical
bootstrap that resamples datasets before label frequencies and seeds, and
mixed-effects sensitivity models.

## Results

Every row records the dataset, labeling mechanism, labeled-positive
availability, seed, method, hyperparameters, the supplied and true class prior,
the PU-computable selection criterion, and the test metrics. Adverse outcomes
are retained rather than dropped: degenerate one-class predictions, non-finite
values and failed fits all appear in the tables.

## Implementation fidelity

`reproducibility/method_implementation_map.csv` is the authoritative record for
each method. It states the implementation level, from a direct transcription of
the published equation through a loss-level port to an adapter around the
authors' released code, the source file and symbol, the upstream component when
one is used, and every deviation from the original.

`reproducibility/upstream_sources.csv` pins the third-party repositories.
`reproducibility/patches/` holds the compatibility patches applied to them.

## Data not tracked here

Derived features and third-party checkouts are excluded because of their size.
`experiments/src/make_features.py` rebuilds the frozen encoder features and the
graph inputs; `reproducibility/build_split_seed_manifests.py` regenerates the
split and seed manifests that fix the case-control construction.

## Tests

    .venv/bin/python -m unittest discover -s reproducibility/tests
