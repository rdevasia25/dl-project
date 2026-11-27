# Baseline provenance and third-party notices

Research reference: Zhiqi Wang, Yichi Zhang, Dongwon Lee, Nathalie Baracaldo,
and Yuchen Yang, *Lost in Context Compaction: When LLM Systems Drop Side
Constraints*, [arXiv:2608.11242v2](https://arxiv.org/abs/2608.11242v2),
especially §5.4 and Appendix F.

Upstream code: <https://github.com/ZhiqiEliWang/compaction-integrity>.
Inspected revision: `e38e2987870111ed33784484a5efe0b4de07a2d0`.
Links below are pinned to that revision.

| Source | Reuse / current status |
| --- | --- |
| [prompts.py](https://github.com/ZhiqiEliWang/compaction-integrity/blob/e38e2987870111ed33784484a5efe0b4de07a2d0/src/compaction_integrity/prompts.py) | `build_sc_extraction_prompts`, `_format_existing_scs`; copied verbatim into the shared prompt section of `sc_common.py` and the pinned test reference |
| [sc_extractor.py](https://github.com/ZhiqiEliWang/compaction-integrity/blob/e38e2987870111ed33784484a5efe0b4de07a2d0/src/compaction_integrity/sc_extractor.py) | Parser copied to the parsing section of `sc_common.py` and the pinned test reference; rendering and `SCExtractor.observe` / `_observe` append-only behavior adapted in the shared extraction section of `sc_common.py`; evidence helpers unused (verification disabled) |
| [eval_sc_extractor.py](https://github.com/ZhiqiEliWang/compaction-integrity/blob/e38e2987870111ed33784484a5efe0b4de07a2d0/src/compaction_integrity/scripts/eval_sc_extractor.py) | Sequential state, source provenance, `_prev_assistant`; evaluator itself will not be imported |
| [runtime/base.py](https://github.com/ZhiqiEliWang/compaction-integrity/blob/e38e2987870111ed33784484a5efe0b4de07a2d0/src/compaction_integrity/runtime/base.py) | Stage 2's `ModelResponse` / `ModelRuntime` interface is informed by this pattern; our protocol and fields are adapted |
| [WildChat extraction configuration](https://github.com/ZhiqiEliWang/compaction-integrity/blob/e38e2987870111ed33784484a5efe0b4de07a2d0/config/tasks/sc_extractor/sc_extractor_wildchat100k.yaml) | Model, whole-turn mode, thinking disabled, output ceiling; reference configuration; hosted adapter uses the approved model, whole-turn mode and output ceiling |

The added contract, serialization, telemetry fields, and integration documentation
are project implementation work. Stages 3–6 copy the prompt/parser and adapt registry logic. The test references
contain only the relevant upstream functions, retrieved at the pinned revision.
Hosted serving, typed provenance/state, validation, error diagnostics and the CLI
are project adaptations; registry lifecycle
extensions and model training are excluded. No dataset or evaluation implementation is reused.

Upstream is MIT licensed. Preserve this notice with subsequent copied or adapted
portions. This notice does not assign a license to unrelated team contributions.

## Upstream MIT license

Copyright (c) 2026 Zhiqi Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
