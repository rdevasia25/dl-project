# Gate dataset — datasheet

Source: Hugging Face `ZhiqiEliWang/conpact` (CONPACT, Wang et al. 2026).

## Source pools

- **wildchat**: 50 histories, 9779 unique user turns (3082 duplicates removed, 161 non-English, 1874 keyword-flagged)
- **hermes**: 50 histories, 94 unique user turns (358 duplicates removed, 0 non-English, 2 keyword-flagged)

## Labeled splits

- **val**: 100 items, 28 positive / 72 negative; categories {'keyword_flagged_turn': 22, 'injected_sc': 20, 'random_turn': 50, 'natural_sc_swe': 5, 'natural_sc_wildchat': 3}; positive SC types {'Action': 5, 'Preference': 5, 'Process': 6, 'Output': 8, 'Information': 4}
- **test**: 200 items, 83 positive / 117 negative; categories {'natural_sc_wildchat': 33, 'random_turn': 93, 'injected_sc': 40, 'keyword_flagged_turn': 24, 'natural_sc_swe': 10}; positive SC types {'Output': 37, 'Process': 15, 'Preference': 11, 'Action': 11, 'Information': 9}

## Stream set (realistic prevalence)

- 3277 consecutive user turns from 10 test histories; 98 natural constraints found; 100 injected-SC instances.

## Label quality

- Not double-labeled; no inter-annotator agreement measured.
- Sampled turns labeled by: Claude (225 items), following LABELING_GUIDE.md. Items labeled by an LLM are marked label_source = "llm".

## Split rule

Splits are by history (no conversation appears in both). Validation uses SC ids 1, 4, 7, 10, 13 (one per type); test uses the other ten, so threshold tuning never sees the test constraints' wording. Turn texts are de-duplicated within and across splits.
