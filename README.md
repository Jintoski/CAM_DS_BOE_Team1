# Bank of England employer project, Team 1

Code and data for Team 1's Assignment 3 submission (Cambridge Data Science, employer project). The report and the presentation slides are submitted separately as PDFs.

The project reads the analyst Q&A sections of HSBC and Barclays earnings call transcripts from 2021 to 2026, and turns them into signals a PRA supervisor can use. These signals cover what analysts ask about, how management sounds, and how directly the question was answered.

## Contents

| Path | Workstream | What it is |
|---|---|---|
| `Team1_CAM_EP_Assignment3_notebook_1_corpus_build.ipynb` | WS1 | Transcript PDFs to `turns`, `exchanges` and `documents` tables, then FinBERT sentiment and BERTopic topics |
| `Team1_CAM_EP_Assignment3_notebook_2_qa_signals.ipynb` | WS1 | Answer-quality signals per exchange and per quarter, and grounded extraction with Gemma 3 |
| `Team1_CAM_EP_Assignment3_notebook_3_llm_signals.ipynb` | WS2 | LLM assessment of each Q&A exchange, with validation and dashboard |
| `ws3_reporting/` | WS3 | GenAI prompts, the supervisory report template, the generated reports and the comparison of the two reporting methods |
| `data/raw/` | | The 104 transcript PDFs and their manifest |
| `data/corpus/` | | Every table notebooks 1 and 2 write, as `.parquet` and `.csv`, plus the fitted topic model and the Gemma output cache. Column definitions are in `DATA_DICTIONARY.md` |
| `data/ws2/` | | WS2's submitted run, `run_20261009T163647_497637Z/`, with its reviewed extensions in `extensions/` and its dashboards in `dashboard/`, and the embedding and Phi caches it was built from |

## How to run this project

Every path in the notebooks is relative to the repository root, so open and run them from there. Run them in order: each notebook reads what the previous one wrote.

| Notebook | Reads | Writes |
|---|---|---|
| 1 corpus build | `data/raw/manifest_all.csv` and the PDFs it lists | `data/corpus/`: `documents`, `turns`, `exchanges`, `sentiment`, `sentiment_exchanges`, `topics`, `topic_info`, `topic_model/` |
| 2 Q&A signals | the tables notebook 1 wrote | `data/corpus/`: `signals`, `signals_quarter`, `signals_topic`, `signals_matched`, `signals_attention`, `signals_review`, `signals_llm.jsonl` |
| 3 LLM signals | the tables in `data/corpus/`, and `data/ws2/cache/` | a timestamped run folder under `data/ws2/`, and new entries in `data/ws2/cache/` |

Every table is written both as `.parquet` and as `.csv`.

### Notebooks 1 and 2 (WS1)

Locally, in a Python 3.10 environment:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"
jupyter lab
```

On Google Colab, clone the repository and change into it before running anything else:

```python
!git clone https://github.com/Jintoski/CAM_DS_BOE_Team1.git
%cd CAM_DS_BOE_Team1
```

Each notebook's first code cell installs what Colab is missing.

Every expensive step loads its saved result from `data/corpus/` when one exists, so both notebooks run end to end on a CPU. The flags in each notebook's settings cell force a step to run again:

| Flag | Notebook | Set it to | Effect |
|---|---|---|---|
| `REBUILD_CORPUS` | 1 | `True` | Parse the 104 PDFs in `data/raw/` again instead of loading the tables |
| `FORCE_REBUILD` | 1 | `True` | Ignore the per-document page cache and extract the text from the PDFs again |
| `RESCORE_SENTIMENT` | 1 | `True` | Score every turn and exchange with FinBERT again |
| `REFIT_BERTOPIC` | 1 | `True` | Fit a new topic model instead of loading `data/corpus/topic_model/` |
| `RUN_FINBERT`, `RUN_BERTOPIC` | 1 | `False` | Skip the sentiment or the topic stage |
| `RUN_EXTRACTION` | 2 | `False` | Use only the Gemma results already in `signals_llm.jsonl` |
| `RUN_ALIGNMENT` | 2 | `False` | Skip question-answer alignment, so no embedding model is loaded |

A rebuild from the PDFs, rescoring and refitting all benefit from a GPU. Notebook 2 calls Gemma 3 only for exchanges missing from `signals_llm.jsonl`, and the file in the repository covers all of them. To extract again, delete the file. On Colab the notebook then loads `google/gemma-3-4b-it` from Hugging Face, which requires accepting its licence there. Run locally, it expects `gemma3:4b` served by Ollama.

### Notebook 3 (WS2)

Notebook 3 was built for Google Colab with an A100 GPU. Run it in its own runtime rather than after notebooks 1 and 2, because its first cell installs `transformers` 4.51, while notebooks 1 and 2 pin 5.12.

By default the notebook reviews the submitted run rather than running Phi again, so it needs no GPU:

1. Clone the repository and change into it, as above.
2. Run the notebook top to bottom. Three cells mount Google Drive: subsection 1.2, the first cell of 7.2 and the cell before 7.4. The repository's data does not need Drive. On Colab, approve the prompt or skip the cell. Outside Colab, skip them, because `google.colab` is not available there.
3. Subsections 7.2, 8.6 and 8.7 load the saved run from `data/ws2/run_20261009T163647_497637Z/` and print its coverage. Subsection 7.4 refreshes the reviewed extensions in `extensions/`. Section 9 shows the coverage, the failure categories and the dashboards.
4. Subsection 9.1 fetches the St. Louis Fed Financial Stress Index (STLFSI4) from FRED, so it needs an internet connection. Each run writes a new timestamped set of `Bank_directness_FSI_*` and `FSI_*` files into `dashboard/`.

To run the model again, set `RUN_FULL_PIPELINE = True` in subsection 8.5. Sections 2 to 8 then load the WS1 tables from `data/corpus/`, run `microsoft/Phi-4-mini-instruct` over every eligible exchange, and write a new run folder named `run_<UTC timestamp>` under `data/ws2/`. To review that run instead, replace `run_20261009T163647_497637Z` with the folder it printed, wherever the notebook names the run: subsections 7.2, 7.4, 8.5, 8.6 and 9.1.

The submitted run folder holds every table the run exported. These include `LLMSignals.csv`, the 32-field file WS3 consumes, with its data dictionary next to it. `RUN_SUMMARY.md` gives the run's coverage: 753 of the 1,122 eligible exchanges have an accepted assessment.

`data/ws2/cache/` holds the sentence embeddings and the Phi answers from the original run. A rerun reuses them while the model revisions it resolves are unchanged, so Phi is only called for exchanges without an accepted answer. The submitted run used `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083` and `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

### WS3 reporting

WS3 has no notebook. Each folder in `ws3_reporting/` holds a prompt file, the files given to the model with it, and the report it produced:

- `Model Metrics GenAI Instructions/` builds the supervisory report from the WS1 and WS2 signal tables.
- `Transcript Based Gen AI Instructions/` builds it from the transcripts directly.
- `Comparison Report/` compares the two methods.

### Reading notebook 1

Three markdown cells near the top of notebook 1 (the introduction, "How to run this notebook" and "Common Variables") say that `RUN_FINBERT` and `RUN_BERTOPIC` are False and that the run covers preprocessing only. Both flags are True in the submitted notebook, and the sentiment and topic sections run, loading their saved results from `data/corpus/`. The Google Drive steps in "How to run this notebook" are not needed after cloning: the paths are already relative to the repository root, as described above.

## Limitations

- Notebook 3's labels have not yet been checked against human annotation. The template and the metrics are in place (5.2), so the accepted labels are known to be well formed and grounded in the text, but not yet known to be correct.
- Component 7 holds one reviewed claim. A match on one number shows consistency for that claim only, not for the rest of the call.
- Component 9's packets are the first 12 keyword matches in period order, so they cover 2021 quarters only and are not representative. Its three findings were drafted with OpenAI Codex, whose version was not recorded, so they cannot be regenerated exactly. Each was reviewed by a team member before it was saved.
- 33% of WS2 assessments failed validation, 43% of those on the schema. Constrained decoding, which forces valid JSON during generation, is the next step.
- The directness weights in notebook 3 are set by design, not fitted. They can be calibrated once human labels exist.
- The WS3 reports were generated in hosted chat models and cannot be rerun exactly. The transcript-route report checked its quotations against the transcripts; the signal-route report states that its source links were not independently verified.
- Summaries grouped by topic, metric and speaker across the whole corpus need a retrieval step over the exchanges and prepared remarks, which was not built. The `topic_id`, `role`, `bank`, `period` and `audience` columns are the filters it would use.
- Topic labels are names given by reading each topic. The model was fitted on HSBC and Barclays only, so another firm would need its own fit.

## Data sources

The transcripts are first-party publications of HSBC Holdings plc and Barclays PLC, downloaded from their investor relations websites. `data/raw/manifest_all.csv` records for each file the URL it was downloaded from, the page the link was found on, the download time and the SHA-256 of the file. Every row in the derived tables carries the `doc_sha256` of its source document, so any figure can be traced back to the transcript and page it came from.
