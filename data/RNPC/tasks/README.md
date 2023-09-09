This directory contains the data for each RNPC task.

Common columns:
- `id`: Example ID.
- `combo`: The combination of modifier semantic categories in the current NP. For example, `pri-sub` means that the first modifier (M1) is privative and the second (M2) is subsective.
- `source NP`: The source NP of the current example.
- `label`: The label of the current example. The meaning of the labels should be self-explanatory.


Each file:
- `SPTE`: The Single-Premise Textual Entailment task. The label means whether the `premise` entails (`1`) or does not entail (`0`) the `hypothesis`.
- `MPTE`: The Mingle-Premise Textual Entailment task. Premise 1 and Premise 2 are concatenated in the `premise` column, separated by the a period and a whitespace (`". "`). The label means whether they collectively entail (`1`) or do not entail (`0`) the `hypothesis`.
- `EPC`: The Event Plausibility Comparison task. The label means whether the `second_event` is less (`0`)/equally (`1`)/more (`2`) likely than the `first_event`.
