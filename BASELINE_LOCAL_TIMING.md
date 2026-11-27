# Local Qwen timing pilot · November 26, 2026

Six independent validation messages ran through the real baseline on Rahul's
M1 Pro MacBook Pro (16 GB). All six requests completed and parsed. The six calls
took **34.05 seconds** in total, including **6.47 seconds** to load the model on
the first call. The next five averaged **4.10 seconds**. Provider fees were $0.
Electricity and hardware costs were not measured.

| Validation ID | Input characters¹ | Input tokens² | Output tokens | Seconds | Parsed constraints |
|---|---:|---:|---:|---:|---:|
| val-0084 | 102 | 1,238 | 7 | 13.57 | 0 |
| val-0034 | 209 | 1,281 | 7 | 3.44 | 0 |
| val-0070 | 340 | 1,310 | 7 | 3.12 | 0 |
| val-0012 | 638 | 1,349 | 7 | 3.11 | 0 |
| val-0099 | 1,613 | 1,618 | 7 | 4.16 | 0 |
| val-0027 | 4,800 | 2,378 | 7 | 6.65 | 0 |

¹ Combined user message and preceding assistant text. ² Ollama's prompt count
includes the shared extraction instructions. Across all calls Ollama reported
9,174 prompt tokens and 42 generated tokens. Every response contained an empty
constraint list. A post-run label check found five negatives and one positive
(val-0027). The extractor missed that positive. The sample was chosen by length
for timing and has only one labeled positive. It cannot establish overall recall
or registry retention.

## Estimated time for the next runs

| Run | Sequential calls | Estimated elapsed time |
|---|---:|---:|
| 100 validation + 200 test examples | 300 | 20 minutes |
| 10 ordinary conversations | 3,277 | 3 hours 44 minutes |
| 100 full injected conversation replays | 32,770 | 37 hours 16 minutes |

These estimates assume continuous operation on the same Mac, with similarly
short outputs. We used the five warm (input character count, seconds) pairs in
the table, interpolated linearly between them, used the shortest warm time below
209 characters, and extended the last slope above 4,800. We added the observed
cold-start premium once per run. The 300-example estimate uses the validation
length distribution for both splits. The two stream estimates use actual
released input lengths, substituting the changed turn in each replay. No test
example or stream turn was sent to the model. There were 264 clean turns longer
than the longest pilot input. Longer generated answers, growing registries,
memory pressure, and interruptions can increase the elapsed time. Every extra
second per turn adds roughly nine hours to the full injected replay.

## Reproducibility and handoff

- Model `qwen3.5:9b-q4_K_M`, Ollama 0.40.18, model digest
  `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
  Ollama reported the llama.cpp runner, a 16,384-token active context and
  6,056,386,231 bytes of model allocation in GPU memory.
- Temperature 0, top-p 1, maximum output 4,096 tokens, thinking disabled,
  truncation and context shifting disabled. Sampling settings and full digest
  are checked or specified by the then-current `sc_baseline.ollama_runtime` (now in `sc_common.py`). The model started
  unloaded. The same server handled all six sequential calls.
- Dataset revision `6cae2dd5d43f3f2a074f8f54ce7719785da78aa3`.
  Validation SHA-256 `2fd44af8e328bf25b45d156774d8d82cbc28cb7ba0bbc3a3d9b528fae7722953`.
  Sort the 100 rows by combined character count, breaking ties by ID. Take
  one-based ranks 8, 25, 42, 59, 76, 93. Only observed fields and source IDs
  were copied. Each example started with an empty registry.
- Runtime code in commit `b472a63`. At launch, the manifest recorded its parent
  `f042602` with the adapter present in the working tree. The package SHA-256
  identifies the code actually run:
  `7d5daf574db384b0bab275f3d1dc3f85a676c8d5fd3722e568422595b3ec10d4`.
  Configuration ID `5250541c50c0cfc41bb7c5338e4e70b101b6aa2c9ef39eebd21d54dc4fe02adc`.
- On the pilot machine, ignored `runs/qwen-timing-20261126/` contains the
  selected inputs, IDs, results, manifest and before/after server metadata.
  Jev can use the same `OllamaRuntime` with `extract_turn` for matched model
  settings. The [class baseline run](BASELINE_CLASS_RESULTS.md) is complete.

The team experiment scope was reduced after this pilot. See the
[revised plan](BASELINE_INTEGRATION_PLAN.md#revised-experiment-scope) for the
300 labeled messages and 6–8 short follow-up cases under a two-hour model
runtime budget.
