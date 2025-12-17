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

## Data Preparation: ABA Fact Insertion and QA Generation

### Raw Data

We include the **raw LoCoMo JSON conversation files** (extracted directly from the original LoCoMo repository) in:

```
raw_data/
```

These files serve as the starting point for ABA-fact insertion and QA generation.

---

### Step 1: ABA′ Fact Insertion (Manual, ChatGPT-Based)

To insert new ABA′ facts into a LoCoMo conversation file:

1. Open a ChatGPT interface.
2. Upload **one JSON file** from the `raw_data/` folder that you want to modify.
3. Navigate to:
   ```
   data_generation_prompts/Data_insertion_prompt.txt
   ```
4. Copy **the entire prompt** from this file and paste it into the ChatGPT input window.
5. Send the message.

ChatGPT may ask follow-up questions (e.g., choosing between Option A or Option B).  
Simply follow the instructions provided.

After completion, ChatGPT will provide:
- A **modified JSON file** with inserted ABA′ facts
- A **separate list of inserted facts**

Download both outputs.

---

### Step 2: QA Generation and Insertion

1. In the **same ChatGPT conversation**, upload:
   - The modified JSON file
   - The corresponding list of inserted facts
2. Navigate to:
   ```
   data_generation_prompts/QA_insertion_prompt.txt
   ```
3. Copy **the entire prompt** and paste it into the ChatGPT input window.
4. Send the message.

ChatGPT will return a **final downloadable JSON file** containing:
- The original conversation
- Inserted ABA′ facts
- Automatically generated and inserted QA pairs (categories 91–94)

This final JSON file is ready for evaluation.

---

## Project Scope and Notes

- Focuses on **long-context reasoning**, **recency bias**, and **temporal fact tracking**
- Introduces structured fact transitions (A → B → A′)
- Fact insertion and QA generation are performed via controlled, prompt-driven ChatGPT interactions
- Evaluation is fully automated using LLM-based grading

---

## Acknowledgements

This project is built on top of the LoCoMo benchmark:

> Snap Research. *LoCoMo: Long-Context Multi-Turn Reasoning Benchmark.*

We thank the original authors for releasing their code and data.

---

## License

Please refer to the original LoCoMo license and the license file in this repository for usage terms.
