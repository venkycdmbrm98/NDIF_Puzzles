# pairs

This directory contains generated matched prompt pairs for model2 causal work.

## Available Files

- `pairs_tens_switch.jsonl`
  Only one tens digit changes and the winner flips.
- `pairs_ones_switch.jsonl`
  The max-tens set stays fixed and one relevant ones digit changes.
- `pairs_binding_swap.jsonl`
  The tens bag and ones bag stay fixed, but the internal pairing changes.
- `pairs_position_swap.jsonl`
  The numbers stay fixed and only position changes.
- `pairs_boundary_cross.jsonl`
  Clean boundary transitions such as `09 -> 10`.

These files are separate from `final_dataset.jsonl` because the pairing structure matters for matched interventions and activation patching.
