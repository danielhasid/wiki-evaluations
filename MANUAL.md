# Wiki Retrieval Eval — Manual Guide

## Overview

This project is a retrieval evaluation framework for Wikipedia-based RAG (Retrieval-Augmented Generation) systems. It provides:

- **Wikipedia data collection** — download, parse, and process Wikipedia pages
- **QA dataset generation** — generate complex Q&A pairs from Wikipedia content via LLM
- **Retrieval evaluation** — measure recall/precision of vector-DB and BM25 retrieval strategies against a ground-truth dataset (FRAMES)
- **Final answer correctness** — compare LLM-generated answers to ground-truth answers using an LLM-as-judge approach
- **Data augmentation** — rephrase FRAMES prompts via LLM for robustness testing

---

## Project Structure

```
wiki-retrieval-eval/
├── config.py                          # Central config (reads from env vars)
├── requirements.txt                   # Python dependencies
│
├── adapters/
│   ├── db_adapter.py                  # MongoDB interface
│   ├── llm_adapter.py                 # OpenAI / LLM interface
│   └── retrieval_adapter.py           # Milvus & retrieval interface (stubs to implement)
│
├── correctness/
│   ├── retreival_correctness.py       # Retrieval recall/precision evaluation
│   └── final_answer_correctness.py    # LLM answer correctness evaluation
│
├── frames_augmentation/
│   └── reframes.py                    # Rephrase FRAMES prompts via LLM
│
├── wikipedia/
│   ├── generate_wiki_qa.py            # Generate Q&A from Wikipedia files
│   ├── hebrew_frames.py               # Load Hebrew FRAMES ground truth
│   ├── download_temporal_reasoning_pages.py
│   ├── download_from_sublinks.py
│   ├── extract_references_content.py
│   ├── extract_by_categories.py
│   ├── wiki_html_markdown.py
│   ├── create_ft_dataset.py
│   ├── response_function_utils.py
│   └── json_utils.py
│
├── tests/
│   └── retrieval_test.py              # Ad-hoc retrieval test script
│
└── utils/
    └── logger.py                      # WiseryLogger wrapper
```

---

## Prerequisites

### 1. Python

Python **3.11** is required.

### 2. External Services

| Service | Default address | Purpose |
|---------|----------------|---------|
| **MongoDB** | `localhost:27017` | Stores ground-truth datasets, evaluation results, and messages |
| **Milvus** | `localhost:19530` | Vector database for semantic retrieval |

Make sure both services are running before executing evaluation scripts.

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set Environment Variables

All sensitive configuration is read from environment variables. Set them in PowerShell before running any script:

```powershell
$env:OPENAI_API_KEY   = "sk-..."           # Required for LLM features
$env:LLM_MODEL        = "gpt-4o"           # Optional, default: gpt-4o
$env:LLM_TEMPERATURE  = "0.0"              # Optional, default: 0.0

$env:MONGODB_URI      = "mongodb://localhost:27017"   # Optional, default shown
$env:MONGODB_DEFAULT_DB = "wiki_retrieval"            # Optional, default shown

$env:MILVUS_HOST      = "localhost"        # Optional, default shown
$env:MILVUS_PORT      = "19530"            # Optional, default shown
```

---

## Running the Scripts

All scripts are run from the **project root** directory.

### Ad-hoc Retrieval Test

`tests/retrieval_test.py` runs a single query against a named Milvus collection and prints the retrieved documents.

```powershell
python tests/retrieval_test.py
```

Edit the `query` and `agent_name` variables at the top of `__main__` before running:

```python
query = "What product did Binh Nguyen advertise on 10/02/2019 15:52:40?"
agent_name = "A06_OSINT"
```

> **Note:** The underlying retrieval functions in `adapters/retrieval_adapter.py` are currently stubs. Implement them before this script will return real results.

---

### Retrieval Correctness Evaluation

`correctness/retreival_correctness.py` evaluates retrieval recall and precision against the FRAMES ground-truth dataset stored in MongoDB.

```powershell
python correctness/retreival_correctness.py
```

**What it does:**

1. Connects to MongoDB and loads the FRAMES ground-truth (`correctness_frames` DB, `full_frames_with_entities` collection).
2. For each channel/DB, runs the configured retrievers and collects retrieved document filenames.
3. Computes **recall@k** and **precision@k** for k ∈ {10, 20, 30, 50, 100, 200}.
4. Stores aggregated results back to MongoDB.
5. Optionally exports results to an Excel file (`retrievers_<ranking_func>.xlsx`).

**Key parameters** (edit in `__main__` block):

| Parameter | Description |
|-----------|-------------|
| `db_name` | MongoDB channel ID to evaluate (or `None` to sample all channels) |
| `ranking_func` | `"reciprocal_rank_fusion"` or `"default"` |
| `calculate_only` | If `True`, skip writing results to MongoDB |
| `write_to_excel` | If `True`, export results to an `.xlsx` file |

**Example:**

```python
enrich_sources_with_correctness_info(
    db_name='ys46fwnt1frkxrfnnajztz1t4x',
    ranking_func='reciprocal_rank_fusion',
    calculate_only=True,
    write_to_excel=True,
)
```

---

### Final Answer Correctness Evaluation

`correctness/final_answer_correctness.py` uses an LLM-as-judge approach to score generated answers against ground-truth answers.

```powershell
python correctness/final_answer_correctness.py
```

**What it does:**

1. Loads ground-truth answers from MongoDB (`correctness_frames` / `full_frames_with_entities`).
2. Loads model responses from the `messages` collection of a given conversation DB.
3. For each query, calls the OpenAI LLM twice (bidirectional comparison) and averages the scores.
4. Scores are on a 0–1 scale (internally 0/2/4, normalized to 0/0.25/0.5 per direction).
5. Exports results to MongoDB and an Excel file (`<channel_id>_final_answer_correctness.xlsx`).

**Edit the conversation DB** in the `__main__` block:

```python
test_compute_final_answer_correctness_metrics("p51dx4qju7d78miksie3j3tmpe")
```

---

### Generate Q&A from Wikipedia Files

`wikipedia/generate_wiki_qa.py` reads Wikipedia article files from a local directory and generates complex Q&A pairs using the LLM.

```powershell
python wikipedia/generate_wiki_qa.py
```

**Edit the `path` variable** in `__main__` to point to your directory of Wikipedia text files:

```python
path = 'C:/path/to/your/wiki-articles-folder'
```

Output is saved as `<path>.json` containing a list of generated question objects with fields:
- `question` — the generated question
- `type` — question type (`fact`, `aggregation`, `comparative`, `summarization`, `other`)
- `answer` — full answer
- `supported_sentences` — list of evidence sentences with source file references

---

### Rephrase FRAMES Prompts

`frames_augmentation/reframes.py` reads the FRAMES ground-truth from MongoDB and creates a rephrased version of each prompt using the LLM, storing results back to MongoDB.

```powershell
python frames_augmentation/reframes.py
```

**Edit the `channel_id`** in `__main__` to match your MongoDB channel containing the FRAMES data:

```python
channel_id = 'h477og5xnpnnbj8id7wkpffgfy'
```

Results are written to the `correctness_frames_rephrased` collection in the same channel.

---

## Implementing the Retrieval Stubs

The retrieval functions in `adapters/retrieval_adapter.py` are currently stubs that raise `NotImplementedError`. You must implement them before running any retrieval-dependent script:

| Function | Description |
|----------|-------------|
| `MilvusConnector.get_milvus_client()` | Return a real Milvus / LangChain VectorStore client |
| `retrieve_documents_and_scores_with_vectordb_semantic()` | Semantic vector search |
| `retrieve_documents_and_scores_with_llm_keywords()` | LLM-generated keyword search |
| `retrieve_documents_and_scores_with_bm25_keywords()` | BM25 keyword search |
| `retrieve_relevant_docs()` | Hybrid retrieval combining the above |
| `call_retrieval_function()` | Dispatcher that calls the appropriate retriever(s) |
| `store_to_mongo_aggregated_results()` | Aggregate and store results in MongoDB |
| `evaluate_single_db()` | Evaluation loop over a single database |

---

## MongoDB Data Schema

### Ground-Truth Collection

- **DB:** `correctness_frames`
- **Collection:** `full_frames_with_entities`
- **Key fields:** `Prompt`, `Answer`, `wiki_links_entities`

### Evaluation Results Collection

- **DB:** `<channel_id>`
- **Collection:** `final_answer_correctness`
- **Key fields:** `query`, `response`, `ground_truth`, `score`

### Conversation Messages Collection

- **DB:** `<channel_id>`
- **Collection:** `messages`
- **Key fields:** `query`, `response`, `step_id`

---

## Logging

All modules use the `WiseryLogger` wrapper (`utils/logger.py`), which writes structured logs to stdout:

```
2026-02-26 10:00:00 | INFO     | wiki-retrieval | start
```

Log level defaults to `INFO`. To enable `DEBUG` logs, set the level programmatically:

```python
import logging
logging.getLogger("wiki-retrieval").setLevel(logging.DEBUG)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `NotImplementedError` on retrieval functions | Implement the stubs in `adapters/retrieval_adapter.py` |
| `Exception: Ground truth not found in DB` | Ensure the FRAMES dataset is loaded into MongoDB (`correctness_frames` / `full_frames_with_entities`) |
| `openai.AuthenticationError` | Set the `OPENAI_API_KEY` environment variable |
| `pymongo.errors.ConnectionFailure` | Ensure MongoDB is running at the configured `MONGODB_URI` |
| `No module named '...'` | Run `pip install -r requirements.txt` |
| Milvus connection error | Ensure Milvus is running at `MILVUS_HOST:MILVUS_PORT` |
