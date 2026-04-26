# families

This directory contains generated family index files for `data2`.

Each file matches the model1 family-index pattern:

- `name`
- `tag`
- `description`
- `count`
- `row_ids`

## Highest-Signal Families

- `all_ones_equal_dataset.json`
- `all_tens_equal_dataset.json`
- `unique_max_tens_dataset.json`
- `unique_max_tens_pos_0_dataset.json` through `unique_max_tens_pos_4_dataset.json`
- `max_tens_tie_unique_winner_dataset.json`
- `repeated_max_tens_2_dataset.json` through `repeated_max_tens_5_dataset.json`
- `repeated_full_max_2_dataset.json` through `repeated_full_max_5_dataset.json`
- `cross_decade_boundary_dataset.json`
- `decade_boundary_09_10_dataset.json` through `decade_boundary_89_90_dataset.json`
- `binding_collision_dataset.json`
- `permutation_orbit_dataset.json`

## Derived Convenience Families

The directory also includes precomputed slices that are handy for quick plots and ablation tables:

- `winner_tens_*`
- `winner_ones_*`
- `tens_margin_*`
- `ones_margin_within_max_tens_*`
- `first_*_pos_*`
- `all_tens_equal_*`
- `max_tens_tie_*`
