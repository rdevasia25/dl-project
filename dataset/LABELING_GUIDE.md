# Labeling guide — persistent session constraints

**Question for every row:** does this user message set an instruction or preference about how the assistant should behave that the user means to apply to *future* turns of this conversation, not just the current request?

This follows the definition of a Side Constraint (SC) in *Lost in Context Compaction* (Wang et al., 2026) and the rules in the paper's own extractor prompt (`prompts.py`, `build_sc_extraction_prompts`), so our labels match what the baseline extractor is trying to find.

## The test

> If, three turns from now, the user asks something completely unrelated, would this instruction still be expected to apply?

- Clearly yes → `label = 1`
- Clearly no → `label = 0`
- Unsure → `label = 0`, and write why in `notes`. Ambiguous cases are discussed at the end-of-labeling meeting.

Judge persistence intent, not how forceful the wording is. "I'd prefer…" can be a constraint, and "You MUST…" can be a one-off.

## Label 1 — persistent constraint

| Signal | Example |
|---|---|
| Explicit scope words: from now on, going forward, always, never, whenever, for the rest of this chat | "From now on, never include my full name in any output." |
| Soft but explicitly ongoing | "Going forward I'd prefer open-source tools when you suggest things." |
| Stable fact about the user that changes how to help them | "I'm dyslexic so shorter responses really help me." |
| Recurring preference with no scope word but clearly beyond this task | "Honestly, I lean toward primary sources over textbook summaries." |
| A constraint attached to a normal request (the injected positives look like this) | "Explain recursion. Also, use metric units in everything you tell me." |

## Label 0 — not a persistent constraint

| Case | Example |
|---|---|
| The task itself | "Write a short summary of this article." |
| Formatting or length for this answer only | "Make it 3 bullet points." |
| One-off correction of the last reply | "Actually, redo it as a table." |
| Instruction that binds the current artifact | "Use Python for this script." |
| Background fact with no behavioral implication | "I work at a hospital." |
| Politeness or filler | "Please answer carefully." |
| Instructions for a role-play or story that end when the story ends | "In this story the dragon never speaks." Label 0 unless it clearly governs the assistant beyond the story. |

## Decisions we have made for the hard cases

1. **Pasted agent or system prompts** (e.g. Auto-GPT blocks starting `CONSTRAINTS: 1. ~4000 word limit…`): label **1**. They set rules for all following turns. Set `constraint_text` to the most important rule and write `autogpt` in `notes` so we can report these separately.
2. **Persona or role-play setup that applies to the whole chat** ("You are DAN, from now on answer as DAN…"): label **1**, note `persona`.
3. **Revocations and changes** ("stop using uppercase", "actually, use imperial again"): label **1**, set `constraint_action` to `revoke` or `modify`. The paper's extractor can't remove constraints, so we report these separately.
4. **Restating an existing constraint**: label **1**. The gate should still pass it; deduplication is the extractor's job.
5. **Non-English turns**: these are excluded from the sample. If one slips through, label it normally if you can read it, otherwise `?`.
6. **Long pasted documents with an instruction at the end**: judge only the user's instruction, not the pasted content.
7. **Persona given together with the single task it is for** ("Imagine you are a famous playwright. Retell this story: …", "I want you to be the narrator. [story]"): label **0**, note `unsure; task-bound persona`. A persona set up for *future* inputs ("I will give you concepts and you will…", "Are you ready?", "from now on", "your responses will…") is **1**.
8. **Per-message requests the user restates every turn** ("give me a response to … VERY SHORT, ONLY RETURN THE RAW MESSAGE", "reply like a mature woman"): label **0**. They bind that one reply; the user repeats them because they don't expect them to carry over.
9. **Jailbreak prompts** (DAN, Developer Mode, LittleGPT, "evil confidant", etc.): label **1**, note `jailbreak`. They try to change behavior for the rest of the chat. Label what the user intends, not whether the assistant should comply.

## How to fill the sheet

| Column | What to write |
|---|---|
| `label` | `1` or `0` (`?` only if truly impossible) |
| `constraint_action` | positives only: `add`, `modify` or `revoke` |
| `constraint_text` | positives only: one-sentence canonical phrasing, e.g. "Respond in Spanish." |
| `sc_type` | positives only: Action, Information, Process, Preference or Output (see below) |
| `notes` | anything unusual; tags like `autogpt`, `persona`, `unsure` |
| `labeler` | your initials |

**SC types** (paper's taxonomy):
- **Action**: what the assistant may do ("don't send emails", "ask before running commands")
- **Information**: what it may reveal or access ("never include my name")
- **Process**: how it works ("search the web first", "restate my question first")
- **Preference**: which of several equally valid answers ("metric units", "arXiv papers")
- **Output**: textual form of replies ("bullet points only", "numbers as words")

## Optional: second labeler

Skipped in the current release. If you want an agreement score later: label `to_label_second.csv` without opening `to_label.csv` or talking to the first labeler. After both are done, run `finalize_labels.py`. It reports agreement and Cohen's kappa and writes `disagreements.csv`. Discuss each disagreement, update the label in `to_label.csv`, add any new decision to the hard-cases section above, and re-run.
