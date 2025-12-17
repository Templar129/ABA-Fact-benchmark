# ABA-Facts Benchmark on LoCoMo

This repository extends the original **LoCoMo** benchmark by introducing an **ABA-facts evaluation setting**, designed to test large language models’ ability to track evolving, contradictory, or corrected facts across long conversations.

This project is **based on and built upon** the official LoCoMo repository:
- https://github.com/snap-research/locomo

---

## Getting Started

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_NAME>
```

---

### 2. Set Up the Python Environment

We recommend creating a virtual environment in the **project root directory**.

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# or
venv\Scripts\activate      # Windows
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. OpenAI API Setup

This project relies on OpenAI GPT models for answer generation and evaluation.

Before running any scripts, make sure your OpenAI API key is exported:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

On Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

---

## Running ABA Evaluation

To run the ABA-facts evaluation on LoCoMo-style test data, use the provided evaluation script.

### Example Command

```bash
python path/to/run_aba_eval.py \
  --input locomo/TestData \
  --output-dir locomo/aba_results \
  --answer-model gpt-5-mini \
  --judge-model gpt-5-mini
```

### Arguments

- `--input`: Directory containing LoCoMo-style JSON test files
- `--output-dir`: Directory where evaluation results will be saved
- `--answer-model`: LLM used to generate answers
- `--judge-model`: LLM used to grade the answers

The output directory will contain per-file results as well as aggregated statistics.

---

## Fact Insertion and QA Generation (Work in Progress)

This repository also includes scripts for:

- Inserting **ABA-style facts** into long conversations
- Generating **category-specific QA pairs** (e.g., categories 91–94)
- Injecting generated QA pairs back into LoCoMo-format JSON files

These scripts are currently **under active development**. Existing, runnable components are included in the repository, and this section will be updated as the pipeline is finalized.

---

## Project Scope and Notes

- This work focuses on **long-context reasoning**, **recency bias**, and **temporal fact tracking** in LLMs.
- The ABA setting introduces structured fact transitions (A → B → A′) across conversation timelines.
- Evaluation is fully automated using LLM-based grading.

---

## Acknowledgements

This project is built on top of the LoCoMo benchmark:

> Snap Research. *LoCoMo: Long-Context Multi-Turn Reasoning Benchmark.*

We thank the original authors for releasing their code and data.

---

## License

Please refer to the original LoCoMo license and the license file in this repository for usage terms.
