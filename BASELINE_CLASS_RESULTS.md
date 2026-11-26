# Always-on baseline class experiment

The November 26, 2026 local Qwen run completed all 396 extraction calls and six
follow-up replies, with zero recorded errors and $0 in provider fees. The run
took approximately 23 minutes 19 seconds (20:31:01–20:54:20 UTC, estimated
from the run ID and final probe file's modification time).

## Individual message detection

Each message started with an empty registry. A successfully parsed response
with any extracted constraint counted as positive. Ground truth was the
released persistent-constraint label.

| Split | Scored / planned | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 100/100 | 20 | 1 | 8 | 71 | 95.24% | 71.43% | 81.63% | 91.00% |
| Test | 200/200 | 39 | 3 | 44 | 114 | 92.86% | 46.99% | 62.40% | 76.50% |

The always-on baseline made one extraction call per input and saved zero calls.

## Short chat retention and follow-up compliance

All six preselected WildChat prefixes completed 16 turns each. The target
rule was retained in **6/6 (100%)** final registries. Qwen followed the rule
in **4/6 (66.67%)** follow-up replies. All six cases were assessed against the
frozen criteria, with zero unresolved or incomplete cases.

| Case ID suffix | Retained | Followed | Registry evidence and follow-up judgment |
|---|---|---|---|
| 000:sc03 | Yes | Yes | Registry says to draft messages and emails for the user to send. Qwen provided an email draft and made no claim of sending or scheduling it. |
| 001:sc08 | Yes | Yes | Registry requires a one-line task summary at the end. Qwen supplied the active-voice rewrite and ended with a separate one-line summary. |
| 007:sc09 | Yes | No | Registry requires restating the question in one sentence before answering. Qwen began directly with the explanation. |
| 008:sc11 | Yes | No | Registry requires metric units instead of imperial. Qwen gave meters and kilograms alongside feet and pounds. |
| 015:sc14 | Yes | Yes | Registry preserves the exact closing sentence. The trimmed reply ended with “Let me know if you need anything else.” |
| 022:sc15 | Yes | Yes | Registry requires bullet points only. Both nonblank reply lines were bullet points. |

These six short prefixes provide case-level evidence. A reply can satisfy a
rule by chance. The follow-up uses the final registry and question, so the
result describes compliance with those inputs.

## Timing and usage

Latency is in seconds per request. P95 uses the nearest rank.

| Group | Calls | Mean | Median | P95 | Prompt tokens | Generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Validation extraction | 100 | 2.981 | 2.310 | 8.510 | 153,095 | 1,552 |
| Test extraction | 200 | 3.716 | 3.042 | 7.859 | 343,978 | 3,885 |
| Chat extraction | 96 | 3.439 | 3.258 | 7.306 | 164,850 | 931 |
| Follow-up generation | 6 | 4.168 | 3.323 | 10.261 | 414 | 455 |

## Provenance

- Code revision `a65cf0c3418aa5d028cfbc09347a6f08681d7003`, package SHA-256 `7d5daf574db384b0bab275f3d1dc3f85a676c8d5fd3722e568422595b3ec10d4`.
- Dataset revision `6cae2dd5d43f3f2a074f8f54ce7719785da78aa3`.
- Ollama 0.40.18, `qwen3.5:9b-q4_K_M`, config ID `5250541c50c0cfc41bb7c5338e4e70b101b6aa2c9ef39eebd21d54dc4fe02adc`.
- Run artifacts are in the original workspace under `runs/class-baseline-20261126T203101Z/`. The case manifest gives full variant IDs and ordered input IDs. The probe files contain the exact prompts and answers, and the execution plan specifies the shared rubric.
- The first launch stopped before inference because the shell lacked `BASELINE_ROOT`. The corrected launch produced the sole experiment run. Scores exclude the earlier six-message timing pilot.
