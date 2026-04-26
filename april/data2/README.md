# data2

This folder contains a real curated analysis dataset for `model2`.

It mirrors the `data/` layout used for model1, but it is a targeted union of mechanism-relevant slices rather than an attempt to approximate the full `100^5` prompt space.

## Files

- `manifest.json`
  Dataset metadata and family counts.
- `final_dataset.jsonl`
  The deduplicated analysis corpus.
- `canonical_probe_dataset.jsonl`
  A compact probe set covering the main mechanism branches.
- `families/`
  Flat family index files with `row_ids` into `final_dataset.jsonl`.
- `pairs/`
  Matched clean/corrupt prompts for patching and causal interventions.
- `manifest.template.json`
  The earlier packaging proposal kept as a reference record.

## What This Dataset Is For

The families were chosen to separate the parts of the model2 computation that matter:

1. tens-stage winner selection
2. ones-stage tie-breaking among the max-tens candidates
3. exact repeated maxima
4. positional invariance
5. digit binding inside each two-digit number
6. decade-boundary behavior such as `09 < 10`

## Core Families

- `all_ones_equal_dataset`
  Pure tens-comparison regime with ones fixed.
- `all_tens_equal_dataset`
  Pure ones-comparison regime with tens fixed.
- `unique_max_tens_dataset` and `unique_max_tens_pos_0..4_dataset`
  Clean step-1 isolation and position checks.
- `max_tens_tie_unique_winner_dataset`
  Step-2 tie-breaking when multiple numbers share the top tens digit.
- `repeated_max_tens_2..5_dataset`
  Tens ties of different sizes.
- `repeated_full_max_2..5_dataset`
  Exact repeated winning numbers.
- `cross_decade_boundary_dataset` and `decade_boundary_09_10..89_90_dataset`
  Lexicographic boundary tests.
- `binding_collision_dataset`
  Same tens bag and ones bag, different within-number pairing.
- `permutation_orbit_dataset`
  Same numbers under selected permutations.

## Notes

- The dataset is intentionally metadata-heavy so later notebook work can filter new slices without regenerating prompts.
- Pair files live separately because the matching structure matters for patching experiments.
- For second-digit analysis, teacher-forcing the first answer digit is still the recommended evaluation setup.
