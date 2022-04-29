This directory contains the sample data for each RNPC task.

Common columns:
- `id`: Example ID.
- `combo`: The combination of modifier semantic categories in the current NP. For example, `pri-sub` means that the first modifier (M1) is privative and the second (M2) is subsective.
- `source NP`: The source NP of the current example.
- `label`: The label of the current example. The meaning of the labels should be self-explanatory.


Each file:
- `SPTE`: 30 examples of the Single-Premise Textual Entailment task.
- `MPTE`: 30 examples of the Mingle-Premise Textual Entailment task. Premise 1 and Premise 2 are concatenated in the `premise` column, separated by the a period and a whitespace.
- `EPC`: 30 examples of the Event Plausibility Comparison task.