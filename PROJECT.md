# Project: Efficient Persistent Constraint Extraction Using Jev

## 1. Project Overview

This is a Cornell Tech Deep Learning course project based on the research paper:

**Lost in Context Compaction: When LLM Systems Drop Side Constraints**

Paper: https://arxiv.org/abs/2608.11242

GitHub: https://github.com/ZhiqiEliWang/compaction-integrity

### Problem

LLM assistants need to remember persistent user instructions throughout a conversation, even when conversation history is compressed.

Example:

User: "From now on, always respond in uppercase."

The assistant should continue following this instruction in subsequent messages, including after context compaction.

The original paper proposes extracting persistent constraints from user messages and maintaining them in a separate registry.

However, extracting constraints from every user message requires expensive LLM calls, even though many user messages do not contain persistent instructions.

### Our Proposed Solution

We introduce **Jev as a lightweight decision gate** before the existing LLM constraint extractor.

Instead of invoking the LLM extractor on every message:

1. Receive a user message.
2. Send the message and relevant conversational context to Jev.
3. Jev estimates whether the message contains a persistent session constraint.
4. If the score exceeds a configurable threshold, invoke the original LLM extractor.
5. Otherwise, skip extraction.
6. Preserve the original constraint registry behavior.

The objective is to reduce expensive LLM calls and overall system costs while maintaining high constraint retention and downstream instruction compliance.

---

## 2. Project Scope

We will compare exactly TWO approaches.

### Approach A: Baseline

Use the original paper's approach:

- Process every user message using the LLM constraint extractor.
- Extract persistent constraints when present.
- Maintain the constraint registry.
- Record extraction calls, token usage, latency, and costs.

### Approach B: Jev-Gated Extraction

Our proposed improvement:

- Run Jev on each user message.
- Obtain a probability or decision score for whether the message contains a persistent constraint.
- Compare the score against a configurable threshold.
- Invoke the same LLM extractor only when the score exceeds the threshold.
- Maintain the same constraint registry.
- Measure extraction-call reduction, accuracy, latency, and costs.

### Out of Scope

We are NOT implementing:

- A custom neural classifier.
- A separately trained PyTorch classification model.
- LLM fine-tuning.
- A complete production memory system.
- A frontend or web application.
- Every experiment from the original paper.

**Important:** Keep the project as simple as possible while maintaining scientific validity and meeting the course requirements.

Do not introduce unnecessary functionality or additional models.

---

## 3. Research Question

Can a lightweight decision model like Jev reduce unnecessary LLM constraint-extraction calls while preserving persistent session instructions and downstream compliance?

### Hypothesis

Jev can filter out messages that do not contain persistent instructions, substantially reducing extractor invocations and overall costs.

We hypothesize that a suitably chosen threshold can preserve high constraint recall while significantly reducing unnecessary LLM extraction calls.

---

## 4. Team Responsibilities

The project will be completed by three team members working on separate components.

### Person 1: Dataset and Baseline Implementation

**Main Responsibility:** Prepare the experimental dataset and reproduce the original constraint-extraction baseline.

Tasks:

1. Clone and understand the original paper's GitHub repository.
2. Identify the existing constraint extraction functionality.
3. Understand its inputs, outputs, and dependencies.
4. Prepare an evaluation dataset of approximately 300 labeled user messages.
5. Include positive and negative examples of persistent constraints.
6. Implement or adapt the baseline to process every user message.
7. Record extracted constraints, LLM calls, token usage, latency, and costs.
8. Export results in a consistent format for evaluation.

Deliverables:

- Labeled dataset.
- Working baseline implementation.
- Baseline experimental results.

### Person 2: Jev Gate Implementation

**Main Responsibility:** Build the proposed Jev-based decision gate.

Tasks:

1. Study the current Jev API documentation.
2. Establish API connectivity.
3. Develop a decision question that identifies persistent session constraints.
4. Implement a function that receives a message and relevant context.
5. Obtain Jev's predicted likelihood of a persistent constraint.
6. Implement configurable threshold-based routing.
7. Integrate Jev with the baseline extractor.
8. Record Jev predictions, execution times, token usage, and costs.
9. Test different thresholds.

Deliverables:

- Working Jev integration.
- Configurable decision gate.
- Jev-gated extraction pipeline.
- Experimental results at different thresholds.

### Person 3: Evaluation and Experiments

**Main Responsibility:** Build the evaluation framework and analyze experimental results.

Tasks:

1. Develop evaluation scripts.
2. Compare Jev predictions against labeled ground truth.
3. Calculate precision, recall, false positives, and false negatives.
4. Measure extractor-call reduction.
5. Compare baseline and Jev-gated total costs.
6. Compare end-to-end latency.
7. Generate graphs showing threshold tradeoffs.
8. Evaluate a smaller subset of conversations for constraint retention and downstream compliance.
9. Analyze failure cases.
10. Prepare reproducible experimental tables and figures.

Deliverables:

- Evaluation framework.
- Comparison metrics.
- Graphs and visualizations.
- Experimental findings.

All three team members will collaborate on integration, debugging, and final submission.

---

## 5. Dataset

We aim to construct an evaluation dataset of approximately 300 labeled user messages.

### Suggested Dataset Categories

| Category | Approximate Count |
|---|---:|
| Persistent session instructions | 60 |
| Ordinary user messages | 120 |
| One-turn instructions | 60 |
| Ambiguous messages, updates, or revocations | 60 |
| Total | 300 |

These counts are initial design targets.

The dataset should use the original paper's publicly available examples where appropriate, supplemented by additional labeled examples if needed.

### Example Positive Messages

- "From now on, always respond in uppercase."
- "For the rest of this conversation, use bullet points."
- "Always include Python examples in your answers."
- "Stop using uppercase in future responses."

### Example Negative Messages

- "What is machine learning?"
- "Summarize this article."
- "Translate this sentence into French."
- "Write this particular answer in uppercase."

One-turn instructions should be considered negative examples if they do not create persistent requirements.

### Data Format

Each example should follow a shared JSON structure.

```json
{
  "id": "example_001",
  "user_message": "From now on, always respond in uppercase.",
  "previous_assistant_message": null,
  "is_persistent_constraint": true,
  "category": "persistent_instruction"
}
```

The team should agree on annotation rules before labeling.

Use separate validation and held-out test data so that threshold selection does not rely on final test results.

---

## 6. System Architecture

### Baseline Architecture

```text
User Message
     |
     v
LLM Constraint Extractor
     |
     v
Constraint Registry
```

The extractor runs on every user message.

### Proposed Jev Architecture

```text
User Message
     |
     v
Jev Decision Gate
     |
     v
Constraint Probability
     |
     v
Threshold Check
     |
     +----------------------+
     |                      |
     v                      v
Above Threshold        Below Threshold
     |                      |
     v                      v
LLM Extractor          Skip Extraction
     |
     v
Constraint Registry
```

### Important Experimental Control

Both approaches must use:

- The same LLM constraint extractor.
- The same extractor prompt.
- The same extraction model.
- The same constraint registry procedure.
- The same evaluation dataset.
- The same downstream LLM for compliance experiments.

The primary experimental difference should be whether Jev controls extractor invocation.

---

## 7. Jev Gate

The gate determines whether a user message introduces, modifies, or revokes a persistent instruction.

### Proposed Decision Question

"Does this user message establish, modify, or revoke an instruction that should continue to govern future assistant behavior in the current conversation, rather than applying only to the present request?"

Use Jev's supported API and verify the current request and response formats.

### Threshold Logic

If Jev returns a probability score:

```python
should_extract = jev_score >= threshold
```

Suggested thresholds:

- 0.1
- 0.3
- 0.5
- 0.7
- 0.9

Test additional thresholds if appropriate.

A lower threshold will generally invoke the extractor more often while reducing the likelihood of missing actual constraints.

Because missing persistent instructions can affect future responses, recall is particularly important.

---

## 8. Experimental Evaluation

### Experiment 1: Gate Classification Performance

Evaluate how accurately Jev identifies persistent constraints.

Metrics:

- Precision.
- Recall.
- False positives.
- False negatives.
- Confusion matrix.

Formulas:

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)
```

### Experiment 2: Extractor-Call Reduction

Compare how frequently each system invokes the expensive LLM extractor.

Formula:

```text
Call Reduction =
1 - (Jev-Gated Extractor Calls / Total Messages)
```

Example:

```text
Total messages = 300
Baseline extractor calls = 300
Jev-gated extractor calls = 90

Call reduction = 70%
```

This example is hypothetical, not an actual experimental result.

### Experiment 3: Cost Comparison

Compare the complete cost of both approaches.

```text
Baseline Cost =
Total Cost of All LLM Extraction Calls

Jev-Gated Cost =
Total Jev Gate Cost + Total Selected LLM Extraction Cost
```

Report percentage savings.

Include the Jev gate's own cost. Do not assume that fewer extraction calls automatically means lower total cost.

### Experiment 4: Latency Comparison

Measure:

- Average Jev decision time.
- Average LLM extraction time.
- Total baseline processing time.
- Total Jev-gated processing time.

Specify whether processing is sequential or parallel.

### Experiment 5: Constraint Retention and Compliance

Use a smaller subset of conversations.

For each approach:

1. Process the conversation history.
2. Maintain the extracted constraint registry.
3. Compare stored constraints against known ground truth.
4. Test downstream LLM responses requiring those constraints.
5. Determine whether the responses obey the persistent instructions.

This evaluates whether reducing extractor calls affects actual system behavior.

---

## 9. Suggested Project Structure

```text
project/
├── PROJECT.md
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── baseline.py
│   ├── extractor.py
│   ├── jev_gate.py
│   └── evaluate.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_jev.py
│   └── compare.py
│
├── results/
│   ├── metrics/
│   └── figures/
│
└── tests/
```

This is only a suggested structure.

Reuse original code where possible rather than unnecessarily rebuilding existing components.

---

## 10. Development Order

### Step 1: Initial Setup

- Clone the reference repository.
- Create the shared project repository.
- Install required dependencies.
- Understand the original extractor.
- Agree on input/output data formats.

### Step 2: Independent Development

Person 1:
- Dataset construction.
- Baseline implementation.

Person 2:
- Jev API integration.
- Gate implementation.

Person 3:
- Evaluation scripts.
- Metric calculations using sample data.

These tasks should proceed in parallel wherever possible.

### Step 3: Integration

- Connect Jev to the original extractor.
- Use identical messages for both systems.
- Verify that threshold routing behaves correctly.
- Confirm that both pipelines generate compatible outputs.

### Step 4: Preliminary Experiments

- Run the baseline.
- Collect Jev scores.
- Test multiple thresholds.
- Calculate recall, costs, and extractor-call reduction.
- Create preliminary figures.

### Step 5: Final Experiments

- Verify experimental results.
- Conduct the smaller downstream compliance experiment.
- Analyze missed constraints.
- Prepare final graphs and reproducible outputs.

---

## 11. Course Milestones

### November 16, 2026: Project Milestone

Target deliverables:

- Working baseline.
- Working Jev gate.
- Labeled experimental dataset.
- Preliminary comparison.
- Initial graphs and findings.

Course milestone report:

At least 2–3 pages using the ACM template, including:

- Research problem.
- Relevant paper.
- Formal method.
- Dataset.
- Evaluation metrics.
- Preliminary experimental results.

### November 30, 2026: Final Submission

Target deliverables:

- Complete implementation.
- Reproducible experiments.
- Final experimental comparisons.
- Graphs and tables.
- Approximately 10-page final research report.
- Code submission.
- Presentation preparation.

Follow the course's academic integrity requirements. The provided instructions prohibit using generative AI to write the final report and require disclosure of permitted AI-assisted coding.

---

## 12. Success Criteria

The project aims to demonstrate that selective extraction can reduce cost while maintaining high constraint recall.

Suggested research targets:

- At least 95% gate recall.
- At least 50% fewer extractor calls.
- Lower overall processing costs.
- Minimal downstream compliance degradation.

These are targets, not guaranteed outcomes.

If Jev fails to achieve these results, analyze why. Negative findings and threshold tradeoffs remain valid experimental contributions.

---

## 13. Instructions for AI Coding Assistants

This project will involve AI coding assistance.

When working on this project:

1. Read this PROJECT.md before making changes.
2. Focus on the assigned team member's responsibilities.
3. Do not implement a custom neural classifier.
4. Do not add extra models unless explicitly requested.
5. Do not create unnecessary frontend components.
6. Avoid overengineering.
7. Keep code simple and understandable.
8. Prefer using the original paper's implementation.
9. Verify the actual Jev API documentation rather than guessing.
10. Maintain a consistent dataset and output format.
11. Make experimental results reproducible.
12. Record API costs and latency accurately.
13. Do not fabricate results or experimental measurements.
14. Add basic tests where appropriate.
15. Explain assumptions and implementation decisions.
16. Do not write the course's final report.

### Core Priority

The objective is NOT to build a large software system.

The objective is to conduct a scientifically valid experiment comparing:

**Baseline: Always-on LLM Constraint Extraction**

versus

**Proposed Method: Jev-Gated Selective LLM Constraint Extraction**

Keep the implementation focused on this comparison.
