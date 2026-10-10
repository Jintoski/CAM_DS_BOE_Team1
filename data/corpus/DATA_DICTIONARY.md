# Data dictionary

The three tables produced by stage 1 of the corpus build: what every column means,
what it may contain, and where it comes from. Sample versions of all three are in
[`_sample/`](_sample/), built from two real transcripts.

> ## About the sample
>
> `_sample/` holds a **slice of real output**: the two development transcripts,
> taken from the tables the pipeline built over all 104. It is there so the shape
> can be reviewed without opening a 3,500-row file.
>
> The numbers in it are real, but two documents are not a sample anything can be
> concluded from. For corpus-wide figures read `build_report.md`, which also lists
> the acceptance gates and their results.
>
> **Three gates currently fail** and their thresholds have deliberately not been
> adjusted. See `REVIEW_QUEUE.md`.

---

## `turns.parquet` -- one row per speaker turn

A turn is one continuous stretch of speech by one person, from the moment they are
named to the moment somebody else is.

| column | type | example | meaning |
|---|---|---|---|
| `turn_id` | str | `0f3a106b51bf-0000` | Primary key, and the citation key. First 12 characters of `doc_sha256`, then the turn number within the document, zero padded to 4. Stable across rebuilds. |
| `doc_sha256` | str | `0f3a106b51bf294f...` | SHA-256 of the source PDF. Joins to `sha256` in `manifest_all.csv` and to `documents.parquet`. |
| `bank` | str | `hsbc` | `hsbc` or `barclays` in the MVP. |
| `fiscal_year` | int | `2026` | Fiscal year the document reports on, not the year it was published. A Q4 call held in January belongs to the previous year. |
| `fiscal_quarter` | int | `2` | 1 to 4. |
| `audience` | category | `equity` | Who the call was for. `equity` or `fixed_income`. Fixed income calls are a different audience asking different questions, so filter deliberately rather than mixing them. |
| `doc_format` | category | `prepared_and_qa` | What the document contains. `qa_only`, `prepared_and_qa` or `prepared_only`. Barclays publishes the Q&A separately from the management speech; HSBC publishes one document with both. |
| `section` | category | `prepared` | `prepared` for the scripted opening, `qa` for the question and answer session. The boundary is the first turn taken by someone from an outside firm. |
| `speaker` | str | `Georges Elhedery` | Name as printed in the transcript. |
| `speaker_norm` | str | `georges_elhedery` | Lower case with underscores, for joining the same person across quarters where the printed name varies. |
| `affiliation` | str | `GROUP CEO` | The firm or job title printed next to the name. An outside firm means an analyst; a group role means management. |
| `role` | category | `management` | `analyst`, `management`, `operator`, `ir` or `unknown`. Derived from `affiliation`, not guessed from what was said. |
| `page_start` | int | `1` | Page of the PDF the turn starts on, 1 based. This is what makes a quote checkable against the source. |
| `turn_idx` | int | `0` | Position of the turn within its document, from 0. Sorting by `doc_sha256` then `turn_idx` reconstructs the call in order. |
| `n_words` | int | `1610` | Whitespace separated words in `text`. Use it to judge what will exceed a model's input limit. |
| `n_chars` | int | `10155` | Characters in `text`. |
| `text` | str | `Welcome, all, to today's call...` | The spoken text, faithful. No lower casing, no number stripping, no stopword removal, no stemming. `CET1`, `11.5%` and `£900m` survive exactly. Any cleaning your model needs is yours to do. |
| `repaired` | bool | `False` | True if ligature repair changed this turn. The banks' PDF fonts drop `ff`, `fi` and `fl`, which arrives either as a visible replacement character or as a silently missing letter. Useful as a quality flag, not as a warning. |

---

## `exchanges.parquet` -- one analyst question and the answers to it

Built by pairing each analyst turn in the Q&A with every management turn that
follows, up to the next analyst turn. This is the table to point topic modelling at.

| column | type | example | meaning |
|---|---|---|---|
| `exchange_id` | str | `0f3a106b51bf-x000` | Primary key. Document prefix, then `x` and the exchange number. |
| `bank` | str | `hsbc` | As above. |
| `period` | str | `2026_Q2` | Fiscal year and quarter as one sortable string. |
| `audience` | category | `equity` | As above. |
| `analyst` | str | `Guy Stebbings` | Who asked. |
| `analyst_firm` | str | `BNP PARIBAS` | The firm they asked from. |
| `question_text` | str | `I have a couple of questions around growth...` | The analyst's turn, whole. Analysts routinely ask two or three questions in one turn, so this is not always a single question. |
| `question_turn_id` | str | `0f3a106b51bf-0002` | Joins back to `turns.turn_id`. |
| `question_page` | int | `5` | Page of the PDF the question starts on, 1 based. With `documents.source_url` it cites an exchange to a first-party page. |
| `answer_text` | str | `I'm going to take your first question...` | Every answering turn joined with a space, in order. |
| `answer_turn_ids` | list of str | `['...-0003', '...-0004']` | The turns that make up `answer_text`, in order. Joins to `turns.turn_id`. |
| `responder_names` | list of str | `['Georges Elhedery', 'Pam Kaur']` | Who answered, in order. More than one name means the question was handed on. |
| `responder_roles` | list of str | `['management', 'management']` | Their roles, same order. |
| `n_words_q` | int | `187` | Words in `question_text`. |
| `n_words_a` | int | `570` | Words in `answer_text`. |

---

## `documents.parquet` -- one row per PDF

The quality view. Every transcript in scope gets a row whether it parsed or not,
which is what makes a silent failure visible.

| column | type | example | meaning |
|---|---|---|---|
| `doc_sha256` | str | `0f3a106b51bf294f...` | Primary key. Joins to `manifest_all.csv` and to the other two tables. |
| `relpath` | str | `hsbc/2026_Q2/transcript__...pdf` | Path under `datasets/`. |
| `bank` | str | `hsbc` | As above. |
| `fiscal_year` | int | `2026` | As above. |
| `fiscal_quarter` | int | `2` | As above. |
| `audience` | category | `equity` | As above. |
| `doc_format` | category | `prepared_and_qa` | As above. |
| `source_url` | str | `https://www.hsbc.com/-/files/...pdf` | Where the PDF was downloaded from, copied from `manifest_all.csv`. |
| `source_page` | str | `https://www.hsbc.com/investors/results-and-announcements/all` | The investor relations page the link was found on. |
| `fetched_at` | str | `2026-08-31T15:03:16+00:00` | When the PDF was fetched, UTC. |
| `n_pages` | int | `12` | Pages in the PDF. |
| `n_turns` | int | `29` | Rows this document contributes to `turns.parquet`. |
| `n_exchanges` | int | `10` | Rows it contributes to `exchanges.parquet`. Zero is normal for a management speech with no Q&A. |
| `n_speakers` | int | `11` | Distinct speakers found. |
| `fffd_count` | int | `0` | Replacement characters found before repair. A rough measure of how badly the font was subset. |
| `silent_repairs` | int | `0` | Words repaired. Counts both the visible and the silent form of the same defect. |
| `parse_status` | str | `ok` | `ok`, `low_turns`, `no_qa` or `failed`. Anything other than `ok` is named individually in the build report rather than tolerated in aggregate. |
| `warnings` | str | | Free text. Anything unusual about this document. |

---

## `sentiment.parquet` -- one row per turn

FinBERT reads at most 512 wordpieces. A quarter of the turns in the sample are
longer than that, so scoring a turn whole would silently throw its tail away.
Sentences are scored instead and averaged back to the turn, weighted by sentence
length.

| column | type | example | meaning |
|---|---|---|---|
| `turn_id` | str | `0f3a106b51bf-0002` | Joins to `turns.parquet`. One row per turn. |
| `n_sentences` | int | `10` | Sentences the turn was split into. |
| `n_tokens_max` | int | `461` | Wordpieces in the longest single sentence. If this ever exceeds 512 the splitter has failed, not the model. |
| `n_sentences_truncated` | int | `0` | Sentences that still had to be cut. Should be zero. |
| `finbert_positive` | float | `0.286` | Length-weighted mean of the positive score across the turn's sentences. |
| `finbert_negative` | float | `0.079` | As above. |
| `finbert_neutral` | float | `0.635` | As above. Most turns land here, which is what you would expect from prepared corporate speech. |
| `finbert_label` | str | `neutral` | Whichever of the three is highest. |
| `baseline_label` | str | `positive` | `distilbert-sst2`, a general-domain model, on the same sentences. It exists so you can see where a finance model and a general one disagree. |
| `models_agree` | bool or null | `true` | **Null when `finbert_label` is neutral.** The baseline is binary and has no neutral class, so a neutral turn has nothing to agree with. Comparing them anyway reports a disagreement rate that is really just the missing class. |

## `sentiment_exchanges.parquet` -- one row per exchange

Same scores, one level up. The turn is the right grain for scoring, the exchange is
the right grain for asking whether management answered more warmly than it was
asked.

These are **aggregated from `sentiment.parquet`, not re-scored**. A turn score is
already a length-weighted mean over its sentences, so weighting turns by word count
gives the same answer as scoring every sentence of the exchange would. The two
tables therefore cannot disagree.

| column | type | example | meaning |
|---|---|---|---|
| `exchange_id` | str | `0f3a106b51bf-x000` | Joins to `exchanges.parquet` and `topics.parquet`. |
| `bank`, `period`, `audience`, `analyst` | str | | As in the other tables. |
| `n_answer_turns` | int | `2` | How many management turns answered. More than one means the question was handed on. |
| `q_n_words` | int | `187` | Words in the question. |
| `q_positive`, `q_negative`, `q_neutral` | float | `0.191` | The analyst's turn. |
| `q_label` | str | `neutral` | Highest of the three. |
| `a_n_words` | int | `570` | Words in the answer. |
| `a_positive`, `a_negative`, `a_neutral` | float | `0.533` | The answering turns, weighted by length. |
| `a_label` | str | `positive` | Highest of the three. |
| `tone_gap` | float | `+0.342` | `a_positive - q_positive`. How much more positive the answer was than the question. Negative means management was the more downbeat of the two. |

## `topics.parquet` -- one row per analyst question

BERTopic is fitted over **analyst questions only**. Management answers would drag
the clusters toward prepared-remarks boilerplate, and the brief asks what analysts
raise.

| column | type | example | meaning |
|---|---|---|---|
| `exchange_id` | str | `0f3a106b51bf-x000` | Joins to `exchanges.parquet`. |
| `bank`, `period`, `audience` | str | `hsbc`, `2026_Q2`, `equity` | As in the other tables. |
| `analyst` | str | `Guy Stebbings` | Who asked. |
| `topic_id` | int | `0` | The topic. **`-1` is the outlier bucket**, not a topic. It is reported, never dropped. A question the model was fitted on keeps the topic it was given after outlier reduction, so the fitted corpus has none; only a question added later without a refit can land in `-1`. |
| `topic_label` | str | `0_nii_hedge_growth` | BERTopic's generated name. |
| `probability` | float | `NaN` | Confidence of the assignment, for a question assigned by `transform` only. `NaN` for a fitted question, because no per-question score survives the save. |
| `question_cleaned` | str | | The text actually fed to the model, after politeness boilerplate was stripped. Kept so you can see what the model saw. |

## `topic_info.parquet` -- one row per topic

| column | type | example | meaning |
|---|---|---|---|
| `topic_id` | int | `0` | Including `-1`. |
| `label` | str | `0_nii_hedge_growth` | |
| `size` | int | `6` | Questions assigned to it. |
| `top_words` | str | `nii; hedge; growth; second half` | Its defining terms, most important first. English stopwords are removed before this is computed; without that every topic is labelled "the, to, you, and". |

---

## Things worth knowing before you model

- **The text is faithful, not cleaned.** Numbers, percentages, currency symbols and
  capitalisation are all intact. This is deliberate: stripping digits would remove
  `CET1`, `IFRS 9` and `Basel 3.1`, which is most of the vocabulary that matters.
- **Filter `audience` and `doc_format` explicitly.** Fixed income calls and
  management speeches are in the corpus on purpose, as a comparison group. Treating
  them as the same population as equity Q&A will quietly skew any result.
- **Long turns exceed model input limits.** Median management turns run to several
  hundred words and the longest run past a thousand. Stage 1 does not split them,
  because where to split depends on what you are doing.
- **Transcript coverage is uneven, and that is a finding.** Some banks publish no
  first-party transcripts at all. Gaps are recorded rather than filled from third
  party vendors, which would break the provenance rule the project rests on.
