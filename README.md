# docqa: one chat agent for your documents, code and SQL

Drop business documents (PDF, DOCX, Markdown, text, CSV), source code in any
language and `.sql` files into one folder, optionally connect a database, and ask:

* **business questions** answered from the documents, with file / page / section citations,
* **code questions** answered from the code, split along functions and classes, with file and line citations,
* **SQL requests** turned into a checked, read-only query that follows your data dictionary
  and your own golden examples, run against the database (or just validated, if none is connected),
  and fixed automatically when the database returns an error.

Ingestion and answering are separate: `docqa sync` (or `docqa watch`) keeps a local
vector index in step with the folder, and `docqa chat` or `docqa ui` answers from it.

Everything runs locally by default (Ollama + Chroma). Switching the chat model or
the embeddings to a cloud API is one line in `.env`; your files and the vector DB
stay on your machine either way, and in cloud mode only the retrieved snippets are
sent to the API.

## Try it in two minutes

```bash
pip install -e ".[ui]"
ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text
set DOCQA_LLM_MODEL=qwen2.5-coder:7b        # macOS/Linux: export DOCQA_LLM_MODEL=qwen2.5-coder:7b

docqa sample                               # sample docs, Python + TypeScript code, schema, golden SQL, SQLite DB
docqa --docs sample_workspace/workspace_files --db sqlite:///sample_workspace/my_company.db ui
```

Press **Sync folder** in the sidebar, then ask:

| Ask | Goes to | What to look for |
|---|---|---|
| What uptime do Enterprise customers get, and what if we miss it? | docs | 99.99%, 25% credit, cites `company_policies.md` |
| How does validate_request_token detect an expired token? | code | cites `code/auth_service.py, lines ...` |
| Rank customers by net revenue | SQL | excludes refunded and pending orders, because `data_dictionary.md` says so |
| Only Enterprise customers | SQL | the follow-up is rewritten using the previous query |
| Month-over-month net revenue growth | SQL | copies the `LAG()` pattern from `golden_queries.sql` |
| Does the API client retry as many times as the policy allows? | code + docs | two partial answers merged into one |

The same works in the terminal: replace `ui` with `chat --sync`.

## Quick start with your own files (local, Ollama)

```bash
# 1. Ollama (https://ollama.com/download), then:
ollama pull llama3.1              # or qwen2.5-coder:7b, better at SQL and code
ollama pull nomic-embed-text

# 2. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# 3. Point it at your files
cp .env.example .env          # set DOCQA_DOCS_DIR=/path/to/your/files
docqa sync                    # first run embeds everything; later runs only changes
docqa chat                    # interactive, remembers follow-ups
docqa ui                      # the same in a browser (pip install -e ".[ui]")
```

One-off question: `docqa ask "What is the refund policy?"`, or force a route with
`docqa ask --route sql "Top 5 customers by revenue"`.
Keep the index fresh while you edit files: `docqa watch --interval 10`.
Inside chat: `/sql`, `/code` or `/doc` before a question forces that route, `/golden`
saves the last working query as a golden example, `/sync` re-indexes, `/reset`
clears history, `/quit` exits. `docqa -v chat` also prints how each follow-up was
rewritten and every failed SQL attempt.

Example:

```
you> What is the refund policy?
bot> Customers may request a refund within 30 days of purchase [1].
Sources:
  [1] policies/refunds.pdf, page 1

you> How do I apply for it?
bot> Email billing@example.com with your order number [2].
Sources:
  [2] policies/refunds.pdf, page 2
```

## How a question is answered

1. **Follow-ups are rewritten** into a standalone question using the recent chat.
2. **A plan is made**: which specialists the question needs, most important first:
   `doc` (business documents), `code` (source code), `sql` (write and run a query).
   The planner is skipped when only one kind of content is indexed or you force a route.
3. **Each planned specialist runs once**, searching only its own chunks (every chunk is
   tagged `doc`, `code`, `sql` or `example` at sync time). If that finds nothing it
   searches everything. Search is **hybrid**: vector similarity finds passages that mean
   the same thing, BM25 keyword search finds exact names like `validate_request_token`,
   `cust_stat_cd` or an error message, and the two lists are merged with reciprocal rank
   fusion. Identifiers are split (`validateRequestToken` also matches "request token")
   and file paths count three times, so naming a file finds it.
   `DOCQA_HYBRID_SEARCH=0` uses vectors only.
4. **With two or three specialists, one more call merges** their partial answers,
   keeping citations and SQL.

The plan is fixed up front, so a question never loops between agents: at most five
model calls plus SQL retries. A LangGraph supervisor that re-decides after each step
was considered; it is slower on local models and needs a step cap to stop it picking
the same worker forever, and the SQL specialist already reads the business documents,
which covers the main reason to combine agents.

## Code: any language, no per-language setup

Code is split with LangChain's language-aware splitters, which cut at `class`, `def`,
`func`, `{}` blocks and so on, for Python, JavaScript, TypeScript, Java, Kotlin, Scala,
C, C++, C#, Go, Rust, Swift, Ruby, PHP, Perl, Lua, Haskell, Elixir, R, Solidity,
COBOL, PowerShell, Visual Basic, Protobuf, HTML and LaTeX. Shell, YAML, TOML, CSS,
Vue, Svelte, Dart, Terraform, GraphQL, XML and a few more are indexed too, split on
blank lines. To add another extension, add it to `LANGUAGE_BY_EXT` or
`GENERIC_CODE_EXTENSIONS` in `docqa/loaders.py`; nothing else changes.

* Citations include line ranges: `src/auth.py, lines 12-48`.
* `node_modules`, `venv`, `dist`, `build`, `target`, `vendor`, `bin`, `obj`, hidden
  folders and `*.min.*` files are skipped, as are code and text files over
  `DOCQA_MAX_FILE_KB` (1024).
* Files that usually hold secrets are never indexed, so they can't end up in a prompt:
  `.env*`, `id_rsa` and other SSH keys, names containing `secret`, `credential` or
  `password`, and anything inside `secrets/`, `keys/`, `certs/`, `private/` or
  `credentials/` folders.
* `DOCQA_CODE_CHUNK_SIZE` (1500) and `DOCQA_CODE_CHUNK_OVERLAP` (150) are separate
  from the document chunk size.
* `DOCQA_INCLUDE_CODE=0` indexes documents only.

## SQL: your documents are the business layer

A schema knows `cust_stat_cd`; your data dictionary knows that "churned" means
`cust_stat_cd = 4`. Before writing SQL the agent collects:

| Input | Where it comes from |
|---|---|
| Schema | the connected database (`DOCQA_DATABASE_URL` or `--db`; `CREATE TABLE` plus 3 sample rows per table), DDL pasted in the app or passed with `--schema tables.sql`, and `.sql` files in the folder |
| Business rules | the closest passages from your PDFs, DOCX and Markdown: status codes, KPI formulas, join paths |
| Golden examples | verified question-to-SQL pairs in any `.sql` file (below) |

Then a LangGraph loop runs **link** (map each term in the question to tables, joins and
rules, citing the passages) -> **generate** -> **check** -> **execute**, and on an error
goes back to generate with the exact error, up to `DOCQA_SQL_MAX_ATTEMPTS` (3) times.
The answer shows a short summary of the result, the mapping, the final query and the
result table, plus any failed attempts in verbose mode. `DOCQA_SQL_SCHEMA_LINKING=0`
skips the mapping step (one model call fewer).

**No database?** Leave `DOCQA_DATABASE_URL` empty and set `DOCQA_SQL_DIALECT`
(`postgres`, `mysql`, `tsql`, `snowflake`, `bigquery`, `oracle`, `sqlite`, ...) or pick it
in the app. The query is still parsed in that dialect, so syntax errors are fixed, but
it is not run and the answer says so. With a database, the dialect comes from the URL.

**Golden examples** are written into any `.sql` file in the folder:

```sql
-- Question: Month-over-month net revenue growth
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(amount_usd) AS net_revenue
    FROM orders WHERE status = 'C' GROUP BY 1
)
SELECT month, net_revenue,
       LAG(net_revenue) OVER (ORDER BY month) AS prev_revenue
FROM monthly;
```

The three most similar ones go into every SQL prompt. Save a good answer as an
example with the app's **Save as golden example** button or `/golden` in chat: it is
appended to `golden_queries.sql` in the folder and indexed on the next sync, so it
survives an index rebuild and can be edited like any other file.

**Safety.** Generated SQL is never trusted:

* sqlglot must parse exactly one statement whose root is a `SELECT` / `UNION` /
  `INTERSECT` / `EXCEPT`, with no `INSERT`, `UPDATE`, `DELETE`, `MERGE`, DDL, `PRAGMA`,
  `ATTACH`, `SELECT INTO` or `FOR UPDATE` anywhere inside it, CTEs included;
* functions that read files, reach the network or stall the server are refused even
  inside a `SELECT`: `pg_read_file`, `lo_import`, `dblink`, `pg_sleep`, `load_file`,
  `load_extension`, `read_csv`, `openrowset`, `xp_cmdshell`, `utl_http` and similar
  (a denylist modelled on WrenAI's);
* the connection is read-only: SQLite is opened with `mode=ro`, PostgreSQL and MySQL
  sessions are set to read-only transactions, and every query is rolled back;
* at most `DOCQA_SQL_MAX_ROWS` (50) rows are fetched.

For SQL Server, Snowflake, BigQuery and others, connect with a **read-only database
user**: that is the only guarantee the database itself enforces. Large databases:
`DOCQA_SQL_TABLES=orders,customers` limits the schema sent to the model.

**Run your own query.** `/run SELECT ...` in chat or the app checks and runs a query you
wrote, with the same read-only rules, and explains the result.

**Start a data dictionary from the database.** `docqa describe-db --out docs/data_dictionary.md`
asks the model to describe each table and column from the schema and sample rows. Every
line is a guess marked for review: correct it, then sync, and it becomes the business
layer for later questions. It refuses to overwrite a file unless you pass `--force`. Install the
driver for your database yourself (`psycopg`, `pymysql`, `pyodbc`, `snowflake-sqlalchemy`, ...).

## Using it for several organisations

Nothing about a company's schema is hard-coded. Each organisation needs three things, and
all of them live in its own folder and settings:

| Layer | How docqa gets it | Per organisation |
|---|---|---|
| Schema | read from the live database at start-up (SQLAlchemy reflection: `CREATE TABLE`, foreign keys, 3 sample rows per table). Big databases: only the most relevant `DOCQA_SQL_MAX_TABLES` (20) tables are shown per question, ranked by tables named in the question or in the retrieved business rules, then embedding similarity, plus their foreign-key neighbours, within `DOCQA_SQL_SCHEMA_CHARS` (12000). | `DOCQA_DATABASE_URL` |
| Data dictionary | the organisation's PDFs, DOCX, Markdown, wiki exports in its folder | `DOCQA_DOCS_DIR` |
| Golden SQL | 5-10 of the team's own queries as `-- Question:` blocks in a `.sql` file in that folder, shown to the model as worked examples | same folder |

Keep one `.env` per organisation and pick it with `--env-file`:

```bash
docqa --env-file acme.env sync
docqa --env-file acme.env ui
```

Each docs folder gets its own index, so organisations never see each other's documents.

## Many databases: one question, the right database

Point the agent at a folder of SQLite files instead of one database URL, and every
SQL question is routed to the database that can answer it:

```bash
docqa --db-dir benchmarks/spider/spider_data/database chat
# or DOCQA_DATABASE_DIR=... in .env; in the app, put the folder path in the database field
```

For each question the agent ranks the databases by a short card per database (name,
tables, columns, foreign keys, a few example values, and BIRD's column descriptions
from `database_description/*.csv`). Two rankings are fused: keyword search (BM25) over
the cards, for exact names like `singer`, and embedding similarity, for meaning
("vocalist" finds `singer`). The chat model then picks one of the top
`DOCQA_DB_CANDIDATES` (3) by reading their cards side by side (`DOCQA_DB_ROUTER=llm`),
or the top hit is used with no model call (`retrieval`). The chosen database then goes
through the normal SQL loop, opened read-only. Answers say which database was used and
which others were considered. Cards and embeddings are cached in the data folder and
only rebuilt for files that change. In chat, `/run @car_1 SELECT ...` runs your own
query on a named database.

## Testing the SQL agent on Spider and BIRD

Two public text-to-SQL benchmarks ship one SQLite file per database, with questions and
verified gold SQL:

| | Databases | Dev questions | What it tests | License |
|---|---|---|---|---|
| [Spider 1.0](https://yale-lily.github.io/spider) | 166 (20 in dev) | 1,034 | joins, nesting, grouping across 138 domains | CC BY-SA 4.0 |
| [BIRD](https://bird-bench.github.io/) | 11 in dev (95 total) | 1,534 | big, messy real data; needs the per-question "evidence" hints and column descriptions | CC BY-SA 4.0 |

Both licenses allow use with attribution (BIRD's README says it moved to CC BY-SA 4.0).
The data is not in this repo: it downloads into `./benchmarks/`, which git ignores.

```bash
docqa get-benchmark spider        # official Google Drive file from the Spider page, ~1 GB
docqa get-benchmark bird          # official BIRD dev.zip, unpacks the nested dev_databases.zip
# if a download link has moved: download the zip in a browser, then
docqa get-benchmark spider --zip ~/Downloads/spider_data.zip

docqa eval spider --route-only --limit 200   # routing only, no chat model needed, seconds
docqa eval spider --limit 50                 # route + write SQL + run it: execution accuracy
docqa eval spider --limit 50 --oracle-db     # give it the right database (leaderboard setting)
docqa eval bird --limit 50                   # BIRD: evidence hints are passed as context
docqa eval spider --resume data/eval/<file>.jsonl   # continue an interrupted run
```

`eval` prints, and saves next to a per-question `.jsonl` under `data/eval/`:

* **Execution accuracy**: the share of questions whose generated query returns the same
  rows as the gold query. BIRD compares rows as sets, exactly like BIRD's own
  `evaluation.py`; Spider compares them as multisets, in order when the gold query has a
  top-level `ORDER BY`. (That is Spider's classic execution match, not its stricter
  "test-suite" accuracy.) Floats are rounded to 6 places.
* **Routing**: how often the right database was picked, and how often it was among the
  candidates, plus accuracy when routed right, so you can tell routing misses from SQL misses.
* Attempts, failures after retries, seconds per question, accuracy by BIRD difficulty, and
  the databases with the most misses.

`--limit N` takes a reproducible random sample (`--seed`), because dev files are sorted
by database and the first 50 would cover one or two. Without `--oracle-db` the agent
routes across every database in the folder (all 166 for Spider), which is harder than
the leaderboard setting. A full Spider dev run is ~3 model calls per question; on a
local 7B model start with `--limit 50 --no-linking` and compare.

### Getting more out of a small local model

Measured on Kiran's PC with `qwen2.5-coder:7b` (2026-11-03, before these features):
Spider 54% (50 questions, routed across 166 databases), BIRD dev 50% (30 questions).
The misses were wrong databases, made-up column names, filter values spelled differently
from the data, and (Spider only) correct answers scored wrong because of column order.
Each fix below comes from a published text-to-SQL agent:

| Setting | What it does | Idea from | Cost |
|---|---|---|---|
| on by default (`DOCQA_SQL_VALUE_LOOKUP=0` or `--no-values` to turn off) | Reads the distinct short text values of every column once per database and tells the model which ones the question mentions: "Commander" becomes `cards.type = 'commander'`, "1:54" becomes `'01:54'` | CHESS (entity and value retrieval), Vanna | a few seconds per database, once |
| always on | When a query fails with `no such column` / `no such table` / `ambiguous column`, the retry is told the aliased table's real columns, the closest-named columns anywhere, and that names with spaces need quotes | DIN-SQL self-correction, MAC-SQL refiner, CHESS revisor | none |
| `--shots 4` (eval) | Shows the 4 most similar solved training questions (numbers and values masked before matching) as earlier chat turns. Spider's `train_spider.json` comes with the download; BIRD needs its separate train set | DAIL-SQL | a longer prompt |
| `DOCQA_SQL_CANDIDATES=3` or `--sql-candidates 3` | Writes 3 queries with different instructions (JOINs, CTEs, re-check columns, keep it simple) and keeps the result most of them return; a non-empty result wins a tie | MCS-SQL, CHASE-SQL (self-consistency) | ~3x time per question |
| scoring fix | Spider now accepts the same columns in a different order, like Spider's official `exec_eval.py` | test-suite-sql-eval | none |

For your own database the same features apply in `chat`, `ask` and the app; your golden
SQL file plays the role of the training set.

**Round 2, from the per-question analysis of those runs.** Half of the Spider misses were
routing (11 of 22), mostly between look-alike databases (flight_1/2/4, three "documents"
databases); the rest were the model adding columns or filters nobody asked for, grouping
by name instead of id, skipping a bridge table, and repeating a wrong column name even
after the hint. So:

* **Routing by fit.** The router now considers 5 candidates (`DOCQA_DB_CANDIDATES`), and
  each is checked for which question words are its table/column names and which stored
  values the question mentions. Candidates are reordered by that fit, and the model sees
  it on each card ("Matches the question: names: airport, city; values:
  airports.City='Aberdeen '"). `docqa eval spider --route-only` measures it without a chat
  model; `--no-fit` shows retrieval alone.
* **Automatic name repair.** When a query fails on a column or table that differs from a
  real one only in case, underscores or spaces (`indep_year` vs `IndepYear`), or uses the
  wrong table alias, the name is fixed and the query re-run without another model call.
* **Stricter rules** in the SQL prompt: only the asked-for columns, no extra filters,
  group by key not name, join through bridge tables, NOT IN / EXCEPT for negation,
  ORDER BY ... LIMIT 1 for "the most recent", parentheses around mixed AND/OR.
* **BIRD evidence with the question.** The per-question hint ("youngest refers to
  MAX(Birthday)") is now appended to the question, as BIRD's own baselines do, instead of
  sitting in the context where the 7B model ignored it.

```bash
docqa eval spider --limit 50 --shots 4                    # values + error hints + 4 examples
docqa eval spider --limit 50 --shots 4 --sql-candidates 3 # plus voting (slow on a laptop)
docqa eval bird --limit 30 --no-values                    # measure one feature at a time
```

**Round 3, from the round-2 Spider run.** The shortlist improved (gold database in the
top 5 for 94% of questions, schema fit alone puts it first 83.5% of the time), yet the
final pick got slightly worse: given five look-alike cards, the 7B model overrode a
correct #1 (employee_hire_evaluation → shop_membership, course_teach → gymnast). So:

* **Trust a clear fit winner.** If #1 leads #2 on schema fit by `DOCQA_ROUTE_MARGIN`
  (0.25 since round 4), it is used without a model call. Otherwise the model chooses among only the top
  `DOCQA_ROUTE_CHOICES` (2). `--route-margin 99` restores always asking.
  `docqa eval spider --route-only` now prints, per margin, how many questions have a clear
  lead and how often that #1 is right, so the margin can be tuned without a chat model.
* **"All the information about X"** now means `SELECT * FROM X`, with no joins. This hint is
  added only to questions that say "all the information / details / data", because as a
  general rule the model also applied it to single-column questions.

**Round 4, from the round-3 runs** (Spider 66%, BIRD 53.3%). On BIRD the 0.15 margin
skipped the model on two student_club questions that schema fit had wrong, one of which
names "Student_Club" outright. Schema fit now counts the database's own name (a full
multi-word name +0.3, a single word +0.1), and the default margin is 0.25: on 200 Spider
questions a lead that large covered 54% of questions with #1 right every time.

**SQL pattern library.** `docqa/sql_patterns/` holds 22 SQL techniques, each with five worked
examples on one small demo shop schema (`demo_schema.sql`): filtering and AND/OR, NULLs, top N,
aggregation, GROUP BY / HAVING, joins through bridge tables, LEFT JOIN, negation (NOT EXISTS /
NOT IN / EXCEPT), subqueries, CTEs, ranking per group, running totals and shares, LAG / LEAD,
set operations, CASE pivots, percentages, dates (SQLite, Postgres and MySQL notes), text
functions, "the row with the highest", self-joins, recursive hierarchies and DISTINCT. Each
file says when to use the technique and the usual mistake. For every question the best one or
two techniques are retrieved by their cue words ("never", "running total", "top ... each") and
shown to the model with two examples, labelled as a demo schema so it copies the technique and
not the names. Nothing in it depends on your databases, so it does not go stale when your SQL
changes. The tests run all 110 examples against the demo database.

Add your own techniques or house rules as more `.md` files in the same format in a folder of
your choice and set `DOCQA_SQL_PATTERNS_DIR`. Turn the library off with `DOCQA_SQL_PATTERNS=0`
(`docqa eval ... --no-patterns` to measure it).

**Documents in the SQL eval.** Until now the benchmark passed no documents to the SQL agent.
`docqa eval bird --docs <folder>` indexes a folder with the app's normal sync (PDF, DOCX, CSV, MD...)
and gives each question the passages from `<folder>/<database>/` once the agent has picked the
database (a data dictionary, business rules), plus one page per SQL technique from any other
folder (a SQL manual), searched by the technique's `manual:` words because a business question
shares no words with a manual. Pattern examples are now translated into the connected database's
dialect with sqlglot (Oracle `FETCH FIRST n ROWS ONLY`, SQL Server `TOP n`, ...), and the 30 s query
limit also applies on Postgres, MySQL and Oracle.

Measured on BIRD 30 with qwen2.5-coder:7b: patterns 66.7%, patterns + whole passages 53.3%. The
passages came from the right database every time, but four to six of them made the model add
columns and joins it read about. So `--docs` now gives only the lines that explain values (codes,
units, meanings) from the top 2 passages, the SQL manual only when a query fails on a function,
date or syntax error, and a rule that documents explain values and never add columns or joins
(`--docs-full` restores the old behaviour). The app also shows manual pages only on such errors.

**Answer checks.** Syntax, safety and database errors were already checked and sent back to the
model. But every wrong answer of the best BIRD run was valid SQL that ran. Now each query is
`EXPLAIN`ed first (names and syntax, before a heavy query starts), and a result that looks wrong
is sent back once with what is suspicious: no rows, a column that is NULL in every row, rows
multiplied by a join, a percentage outside 0-100 or a negative count. Each kind fires at most once
per question; if the model returns the same query it is accepted, and if the re-ask fails to run
the first query is kept. `DOCQA_SQL_REVIEW=1` (`--review`) adds a self-review: the model sees its
query and first rows and answers OK or WRONG with a reason. Turn the checks off with
`DOCQA_SQL_CHECKS=0` (`--no-checks`).

**What the model sees (round 7).** With BIRD's column descriptions attached, one wide table
(card_games `cards`, 14,438 characters) filled the 12,000-character schema budget and the other
five tables were never shown; 3 of the 10 misses of the best run needed a hidden table. The schema
is now built in layers: every chosen table is named with its primary key and joins first, then its
CREATE TABLE (or a one-line column list), then column notes (trimmed to the lines about the
question) and sample rows, most relevant table first. Each result row of `docqa eval` records what
every table got (`"schema"`), and the summary counts questions where a table was shortened.
In chat, with one document folder per database, the documents are now fetched for the chosen
database after routing, as in the eval.

Two mistakes the database accepts are now caught: a condition that compares a column with itself
(`ON st.setCode = st.setCode`) is sent back with the real join keys, and a text filter on a value
the column never holds (`Country = 'Slovakia'` where the column stores `SVK`) is sent back with the
stored values. The planning step now states what one result row is about, the measure, grouping,
denominator and tie rule ("the score of the oldest player" is one player's score, not a MAX over
everyone). `--sql-candidates N --independent` plans each candidate afresh, and the summary shows
how often any candidate was right and how often the agent picked it. The "first row" rule follows
the dialect (Oracle `FETCH FIRST 1 ROWS ONLY`, SQL Server `TOP 1`), Oracle queries are checked
with an empty wrapped query (`WHERE 1 = 0`) before they run, and `DOCQA_SCHEMA_TTL` re-reads a
changing schema during a chat.

For a fair comparison, freeze question sets: `--save-ids` writes the chosen ids, `--ids` runs
exactly those, and `--exclude-ids` keeps the regression questions out of an unseen set:
`docqa eval bird --exclude-ids regression30.txt --limit 60 --seed 1 --save-ids unseen60.txt`.

**Changes from the pasted scripts.** The Dropbox Spider link is the old release; the
downloader uses the Google Drive file linked from the Spider page and, if that moves,
tells you to download in a browser and pass `--zip`. The schema indexer built `PRAGMA table_info('{t}')`
with string formatting, which breaks on a quote in a table name; here identifiers are
quoted. One whole-database document per schema in Chroma with top-1 retrieval has no
keyword half and no second look at close candidates. The Streamlit script had its prompt,
session state and chat loop outside the code (broken indentation and a stray fence), never
ran the query, and could not tell a right answer from a wrong one. Routing lives in the
existing agent, CLI and app instead of a second app.

## Writing documents the SQL route can use

Plain sentences work. Three patterns help most:

* **Codes and enums**: "In `fact_transactions`, `txn_type_cd = '01'` means Purchase, `'02'` Chargeback, `'03'` Refund."
* **Metric formulas**: "Net ARR is `SUM(mrr_amount * 12)` from `subscriptions` where `is_active = 1` and `trial_flag <> 'Y'`."
* **Join paths**: "Join `customers.zip_code` to `zip_territory_map.zip_code`, then join on `territory_id`."

## Using a cloud model instead

```bash
pip install -e ".[openai]"     # or .[anthropic] / .[google]
```

```ini
# .env
DOCQA_LLM_PROVIDER=openai      # openai | anthropic | google | ollama
DOCQA_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Embeddings can stay local (no document text leaves your machine during sync)
DOCQA_EMBED_PROVIDER=ollama
```

Anthropic has no embedding model, so pair `DOCQA_LLM_PROVIDER=anthropic` with
Ollama or OpenAI embeddings. Each embedding model and docs folder gets its own
collection, so switching `DOCQA_EMBED_*` or `--docs` builds a fresh index rather
than mixing vectors or deleting the other folder's chunks.

With Ollama, `DOCQA_NUM_CTX` (default 8192) sets the model's context window.
Ollama's own default is small and it drops overflowing text without an error.

## How it handles the five "missing pieces"

| Problem | What docqa does | Where |
|---|---|---|
| Changing files | LangChain indexing API with a SQLite record manager and `cleanup="full"`: unchanged chunks are skipped, edited files are replaced, deleted files are removed. A file that fails to load keeps its previous chunks, and a sync that finds no files is refused unless you pass `--allow-empty`. | `docqa/indexing.py` |
| Multi-format parsing | pypdf (per page), docx2txt, Markdown split by headings. Optional Docling (`DOCQA_USE_DOCLING=1`, `pip install -e ".[docling]"`) for tables and scanned PDFs. | `docqa/loaders.py` |
| Chunking with overlap | `RecursiveCharacterTextSplitter`, 800 chars / 150 overlap by default (`DOCQA_CHUNK_SIZE`, `DOCQA_CHUNK_OVERLAP`). | `docqa/loaders.py` |
| Follow-up questions | With history, the LLM first rewrites the question into a standalone one, and that is what gets searched and answered. Earlier answers are not passed to the answer step, so their old citation numbers can't leak in. | `docqa/chain.py` |
| Grounding and citations | Context is numbered and labelled with file, page and section; the prompt allows only that context, requires `[n]` citations, and has a fixed "not found" reply. The app prints only the sources the answer cited (`[1]`, `[1, 2]`, `[1-3]`) and flags an answer that cites none. | `docqa/chain.py` |

## Changes from the pasted code-and-SQL designs

* `from langchain.indexes import ...` and `from langchain.chains import create_sql_query_chain`
  fail on LangChain 1.x (they moved to `langchain_classic`). The SQL chain is not used here:
  the LangGraph loop does more (linking, checking, retries).
* The DOC route's context dropped the file names while the prompt asked the model to cite them.
* The regex guardrail blocked `WHERE note = 'please update'` and let `PRAGMA`, `ATTACH`, `SELECT INTO`
  or a second statement through; it is replaced by a parser check plus a read-only connection.
* The app's sync ran full cleanup with no guards and one unreadable file aborted it; it now uses
  the same sync as the CLI (failed files keep their chunks, an empty folder is refused), and
  `.sql` files are indexed, which the app version skipped.
* The router's `"SQL" in route` check matched words like "NoSQL" and fell back to documents
  only; routes are now parsed as whole words, and an unclear plan searches everything.
* `rglob("*")` indexed `node_modules`, virtualenvs and minified bundles.
* The golden-SQL snippet had a syntax error (`Document(` was never closed) and stored examples
  only in a vector store filled from the UI, so a rebuilt index lost them. They are files now.
* The multi-agent supervisor could pick the same worker forever; the plan here is fixed and
  every specialist runs once.
* `langchain-community` (for `SQLDatabase`) is being sunset; schema introspection is done
  directly with SQLAlchemy, and that dependency is gone.

## Changes from the original document blueprint

* `format_docs` dropped all metadata, so the model could not cite anything. Each
  passage now carries `[n] (source: file, page N, section "...")`.
* `from langchain.indexes import SQLRecordManager` fails on LangChain 1.x; it now
  lives in `langchain_classic.indexes` and needs `sqlalchemy[asyncio]`.
* `source` is the path relative to the docs folder, and `page` is 1-based (pypdf
  counts from 0).
* Loader metadata such as PDF producer and dates is dropped. The record manager
  hashes metadata too, so volatile fields would make every sync re-embed.
* Empty pages, Office lock files (`~$...`), hidden files and unreadable files are
  skipped and reported instead of crashing the sync.
* If nothing is retrieved, the LLM is not called at all.
* Chunk hashes use SHA-256 instead of the default SHA-1.

## Limits worth knowing

* Generated SQL is checked and run read-only, but read the query before you reuse it
  elsewhere: a valid query can still answer a slightly different question.
* Statement timeouts are not set in chat; a very heavy query runs until the database stops it.
  (`docqa eval` stops each benchmark query after `--timeout` seconds, 30 by default.)

* Scanned PDFs have no text layer; sync warns about them. Use Docling for OCR.
* In Docling mode citations show file and section but not page numbers.
* DOCX files have no page numbers to cite (Word computes pages at render time).
* A file that fails to parse keeps its last indexed version; sync lists it under
  `error`. A file that has never parsed is not in the index at all.
* Run one writer at a time: `watch` and `chat` with `/sync` in two terminals both
  write the same Chroma and SQLite files.
* Chroma's anonymous telemetry is turned off.
* Small local models follow the citation rules less reliably than large ones.
  `qwen2.5:7b` or `llama3.1:8b` are a reasonable floor.

## Using it from Python

```python
from docqa import Settings, build_agent, sync_documents

s = Settings.from_env()
print(sync_documents(s).summary())
agent = build_agent(s)
ans = agent.ask("What is the refund policy?")
print(ans.route, ans.answer, ans.sources_text(), sep="\n")

ans = agent.ask("Top 5 customers by net revenue", route="sql")
print(ans.sql.sql, ans.sql.columns, ans.sql.rows)
```

## Open-source projects worth knowing

All ten projects below were read for this build. No code is copied from any of them; the
ideas listed were reimplemented here and are credited.

| Project | License | Idea used in docqa | Use it instead when |
|---|---|---|---|
| Vanna | MIT | related tables per question; golden examples as earlier chat turns | SQL is all you need and you want its UI and integrations |
| WrenAI | Apache-2.0 (docs CC BY 4.0) | denylist of dangerous SQL functions; draft table and column descriptions for review (`describe-db`) | many people need governed metrics and charts |
| SQLAIAgent-Ollama | no license file | running a query you wrote (`/run`) | (cannot be reused: no license) |
| SmolSQLAgents | MIT | confirmed the retry-with-error loop and keeping the schema small | you want a smolagents-based agent |
| AnythingLLM | MIT | uploading files from the app | you want a full multi-user document workspace |
| RAGFlow | Apache-2.0 | hybrid keyword + vector search; OCR for images | you need its deep PDF layout and table parsing |
| Verba | BSD-3-Clause | hybrid search with weighted fusion | you want a Weaviate-based document app |
| Continue | Apache-2.0 | BM25 over code with file paths weighted up; skipping secret files | editing code in VS Code or JetBrains |
| Aider | Apache-2.0 | matching identifiers in the question to files | editing code from the terminal |
| Cody (public snapshot; the main repo is now private) | Apache-2.0 | keyword search split on symbol names | editing code with Sourcegraph |

Ideas looked at and not taken: a LangGraph supervisor that re-decides after every step
(see "How a question is answered"), WrenAI's full semantic-layer model (MDL), which is a
separate product to maintain, and Aider's tree-sitter repo map, which needs a native
parser per language where the LangChain splitters need none.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use fake embeddings and a scripted chat model, so they need neither Ollama
nor an API key. They cover loading all formats with citation metadata, overlap,
incremental add/skip/edit/delete, persistence across runs, the empty-folder and
failed-file guards, follow-up rewriting, grounded prompts, citation parsing, code
splitting along functions with line numbers, golden examples, the SQL read-only
check and read-only connection, self-healing retries against a real SQLite file,
routing and multi-route answers, routing across a folder of databases, the Spider/BIRD
loaders, row comparison rules and `eval` (on tiny fake benchmarks), the CLI and the Streamlit app. None of them run a
real model, so try the sample questions with your model before relying on it.
