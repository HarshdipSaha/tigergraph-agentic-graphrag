# TigerGraph Agentic GraphRAG Hackathon — Full Brief

Compiled 2026-09-05 from the Unstop listing, the official Notion guidebook, the dataset on Google Drive (downloaded and analysed), the TigerGraph repos, and write-ups from the previous TigerGraph hackathon. Sources are at the bottom.

---

## 1. TL;DR

- **What:** Build three question-answering pipelines over one corpus on TigerGraph: plain **RAG**, **GraphRAG**, and **Agentic GraphRAG**. Benchmark all three on accuracy, completeness, and token cost. The research question the organisers want answered is *"when does agentic reasoning actually help, and when is it overkill?"*
- **Who:** Open globally. Students and professionals. Solo or teams of up to 5. Free.
- **Prize pool:** ₹70,000 (₹35k / ₹20k / ₹15k). Certificates for all valid submissions. Top teams featured on TigerGraph channels.
- **Format:** Two rounds. Round 1 open to all. Top 15 advance to Round 2 ("Reasoning Over Time": conflicting / evolving / uncertain facts on a harder dataset), then a live presentation.
- **Hard deadlines (IST):** register by **Sep 12–13**, Round 1 submission by **Sep 24**, Round 2 by **Oct 1**, results **Oct 7**.
- **Dataset domain:** 2,951 English Wikipedia articles, of which 2,162 are **Olympic event pages (1988–2022)** with a structured infobox. The other ~740 docs are distractors (films, politicians, companies). All 150 benchmark questions are about Olympic events and are answerable from infobox fields alone. This shapes the whole build (see §7 and §10).
- **Registered so far:** 132 (as of Sep 5).

---

## 2. Key dates and deadlines

The Unstop stage cards, the Unstop "Details" prose, and the Notion guidebook disagree on the registration close date. Use the earliest.

| Milestone | Unstop stage cards (authoritative for the platform) | Unstop details prose | Notion guidebook |
|---|---|---|---|
| Registration opens / guidebook + dataset live | 02 Sep 2026 00:00 IST | Sep 1 | Sep 2 |
| **Registration closes** | **14 Sep 2026 00:00 IST** (i.e. end of Sep 13) | **Sep 12** | Sep 14 |
| Round 1 build window | 14 Sep → 24 Sep 2026 00:00 IST | — | — |
| **Round 1 submission deadline** | 24 Sep 2026 00:00 IST | Sep 24 | Sep 24 |
| Round 2 window (top 15 only) | 25 Sep → 05 Oct 2026 00:00 IST | Oct 1 final deadline | Sep 25 → Oct 1 |
| Judging | — | Oct 2–5 | Oct 2–5 |
| Results announced | Oct 7 | Oct 7 | Oct 7 |

Notes:
- Unstop stage "end" times are 12:00 AM IST, which in practice means the previous calendar day is the last full day. Treat Round 1 as due **end of Sep 23 IST** to be safe.
- The Unstop page showed "8 Days Left" on Sep 5, consistent with a Sep 13 registration close.
- The guidebook's day-of-week labels are wrong for 2026 (Sep 14 is a Monday, not Sunday). Ignore them.
- Nothing stops you building before Sep 14. The dataset and guidebook are already live.
- Page last updated on Unstop: 03 Sep 2026 21:24 IST.

---

## 3. Eligibility, team, registration

- Eligible: undergraduate, postgraduate, engineering, management, arts/commerce/science, law, medical, freshers, experienced professionals. Effectively anyone.
- Team size 1–5. Register on Unstop (organiser: TigerGraph). Free.
- Tags on Unstop: Data Science, AI Engineering. Mode: Online.
- Unstop listing: https://unstop.com/hackathons/agentic-graphrag-hackathon-tigergraph-1747871

---

## 4. Prizes

| Place | Cash |
|---|---|
| 1st | ₹35,000 |
| 2nd | ₹20,000 |
| 3rd | ₹15,000 |

Plus certificates for every valid submission and community-channel features for top teams. Free Savanna credits, Vector DB, MCP, and GSQL access for all participants.

---

## 5. The challenge

### Core (everyone must build this)

An Agentic GraphRAG system that plans and executes a multi-step investigation. Required components, quoted from the organisers:

1. **An agent harness** managing state, tools, context, evidence, and stopping criteria.
2. **An orchestrator agent** that decides what needs investigating and picks the next action. Explicitly *not* a fixed retrieval sequence.
3. **Specialised agents** for: entity linking, graph traversal, similarity search, document retrieval, aggregation, multi-hop reasoning, evidence evaluation.

The orchestrator's next move must depend on: the original question, the graph and available entities, evidence returned so far, and what is still missing.

Illustrative paths given by the organisers:
- `Entity Linking → Graph Traversal → Answer`
- `Similarity Search → Identify Entity → Graph Traversal → Retrieve Supporting Documents → Answer`
- Harder questions: several iterations before enough evidence exists.

### The benchmark (mandatory)

Every question is answered **three ways**: RAG, GraphRAG, Agentic GraphRAG. Compare on **accuracy, completeness, and token efficiency**. The organisers say outright: "That comparison is the point."

### Stretch (Round 2 theme, optional in Round 1)

**Reasoning over time.** Detect conflicting versions of a fact, determine what supersedes what, decide which sources are more authoritative, handle uncertainty. Round 2 finalists get a harder dataset for this. "This is where strong teams pull ahead."

### Ground rules

1. Build the three-way benchmark. Non-negotiable.
2. Add-ons are welcome and count under engineering and innovation.
3. You may bring your own dataset as a bonus, but the provided one is the common benchmark everyone is scored on.

---

## 6. Required stack and resources

| Resource | Link | Notes |
|---|---|---|
| TigerGraph Savanna (cloud, recommended) | https://tgcloud.io | "Credits will be provided." Free tier: 1 read/write + 2 read-only workspaces, up to 32 vCPU / 256 GB per workspace, manual backups only. |
| TigerGraph Community Edition (local) | https://dl.tigergraph.com | Free, single server, up to 300 GB graph+vector. Needs 4 CPU / 16 GB min, 8 CPU / 20 GB recommended. TigerGraph 4.2+ for vector search. |
| TigerGraph GraphRAG repo | https://github.com/tigergraph/graphrag | The starter. AGPL-3.0. Has an "agentic chat engine" with *planned* (upfront DAG) and *reactive* (step-by-step) styles, MCP tool calling, entity extraction, community detection, trace logging. Supports OpenAI, Azure, Google, Bedrock, Ollama, HF, Groq. Docker one-liner setup. |
| TigerGraph MCP server | https://github.com/tigergraph/tigergraph-mcp | `pip install tigergraph-mcp`. 50+ tools: schema, vertices/edges, run/install queries, NL→GSQL/Cypher, loading jobs, vector upsert + similarity search. stdio or streamable-http. Set `TG_TGCLOUD=true` for Savanna. Guides for LangGraph (recommended), CrewAI, VS Code Copilot. |
| TigerGraph docs | https://www.tigergraph.com/docs/home/ | |
| GSQL vector search tutorial | https://github.com/tigergraph/ecosys/blob/master/tutorials/VectorSearch.md | `ALTER VERTEX X ADD VECTOR ATTRIBUTE emb(dimension=N)`; `vectorSearch({X.emb}, qvec, K, {candidate_set: c})` for hybrid graph-filtered vector search. |
| Dataset | https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9 | Downloaded to `data/` in this repo. |
| Guidebook | https://alluring-beryllium-491.notion.site/Agentic-GraphRAG-Hackathon-Guidebook-34fc2cb129c08146998af3568d7d2594 | Single source of truth per organisers. |
| Discord | https://discord.gg/eKWm3mbkw2 | Updates land here first. |
| WhatsApp group | https://chat.whatsapp.com/GN0x6yajuroDJTzLMEZvBR | |
| 1:1 with organiser (Devanshu Saxena, TigerGraph) | https://calendly.com/devanshu-saxena-tigergraph/20min, WhatsApp wa.me/917404313376 | Office hours offered throughout. |
| LLM | Any provider | Organisers say free tiers are enough. |

**Not yet published:** the guidebook advertises two sub-guides, "Build Faster with MCP" and "Accuracy Evaluation Guide" (evaluation approach, code snippets, scoring methodology). As of Sep 5 these are plain text blocks in Notion with no linked page. Watch Discord for them. The evaluation guide matters because it will define how "accuracy" is scored.

---

## 7. Dataset deep dive (from the actual files)

Files (all JSONL, downloaded to `data/`):

| File | Contents | Size |
|---|---|---|
| `corpus.jsonl` | 2,951 documents, ~5.47M tokens | 22 MB |
| `eval_public.jsonl` | 100 questions **with** answers and gold doc IDs | 35 KB |
| `eval_hidden.jsonl` | 50 questions, no answers. You submit raw outputs for these. | 8 KB |

Dataset README rules: **"The corpus is the only source of truth."** If Wikipedia today disagrees with the corpus, the corpus wins. Docs are English Wikipedia converted to plain text, coverage 1987–2023, subject matter "narrow". Licensed CC BY-SA 4.0, each doc carries its source URL.

### Corpus document schema

```json
{"doc_id": "Q303623", "title": "...", "url": "...", "wikidata_qid": "Q303623",
 "wikipedia_pageid": 35771859, "approx_tokens": 704,
 "text": "[Infobox Olympic event]\n  event: ...\n  games: 2012 Summer\n  venue: Eton Dorney\n  date: 6 to 8 August\n  competitors: 24\n  nations: 12\n  gold: Rudolf DombiRoland Kökény\n  goldNOC: HUN\n  silver: ...\n  bronze: ...\n  prev: 2008\n  next: 2016\n\n<prose>\n<results tables as pipe-delimited rows>"}
```

Token length: min 111, median 1,032, max 26,874.

### What is actually in the corpus

| Infobox type | Docs |
|---|---|
| `[Infobox Olympic event]` | **2,162** |
| `[Infobox film]` | 549 |
| officeholder / person / writer / scientist | ~152 |
| tennis tournament event, football/handball competition, company, misc | ~88 |

Olympic event docs by Games (Summer: 1988=165, 1992=179, 1996=170, 2000=215, 2004=194, 2008=210, 2012=245, 2016=281, 2020=44; Winter: 1988=34, 1992=41, 1994=54, 1998=52, 2002=65, 2006=67, 2010=76, 2014=74, 2018=24, 2022=19). Top sports: Athletics 295, Swimming 202, Wrestling 145, Shooting 115, Judo 111, Rowing 103, Cycling 102, Weightlifting 98, Boxing 83, Alpine skiing 80, Sailing 78, Cross-country skiing 78, Biathlon 78, Speed skating 74, Fencing 68.

The ~740 non-Olympic docs (films like *Forrest Gump*, people like Vladimir Putin, companies like BHP) are **distractors**. No benchmark question touches them. They exist to punish naive vector retrieval and to let you show the agent ignoring noise.

Infobox field coverage among the 2,162 event docs: gold 100%, competitors 98.5%, nations 98.4%, next 97.6%, prev 93.4%, venue 96.3%, date 57.9%. 55 docs have empty or non-numeric `competitors` (e.g. "23 teams"). 49 docs have "Olympics" in the title but no event infobox (mostly tennis tournaments and team sports with a different infobox).

### The 150 questions: five templates

Every question in both files fits one of five templates. Public vs hidden distribution:

| qtype | Public (100) | Hidden (50) | Template | Gold docs needed | Infobox fields that answer it |
|---|---|---|---|---|---|
| `lookup` | 19 | 7 | "How many nations competed in *{exact event title}*?" | 1 | `nations` |
| `multi_hop` | 28 | 10 | "Who won the gold medal in the event held at *{venue}* on *{date}*?" | 1 | `venue` + `date` → `gold` |
| `temporal` | 22 | 8 | "Who won gold in *{event}* at the *{Summer/Winter}* Olympics held immediately before *{year}*?" | 2 | resolve event at year → `prev` → same event at previous Games → `gold` |
| `aggregation` | 21 | 15 | "How many *{sport}* events at the *{Games}* had more than *{N}* competitors?" | 8–43 (avg 15) | filter by sport+games, count where `competitors > N` |
| `superlative` | 10 | 10 | "Which *{sport}* event at the *{Games}* had the highest number of competitors?" | 8–43 (avg 14) | filter by sport+games, argmax `competitors` |

Public-set metadata fields: `answer_named_in_question` is false for all 100, `guess_baseline` is 0.0 for all 100, `answer_verified` is true for all 100. `gold_doc_ids` are Wikidata Q-IDs equal to `doc_id`. Every answer is a single string (no multi-answer lists), but team golds are concatenated with no separator (e.g. `"Dani KingLaura TrottJoanna Rowsell"`, `"Erik LesserDaniel BöhmArnd PeifferSimon Schempp"`), which matters for answer normalisation.

Recomputed check: pub-001 ("biathlon events at 2018 Winter with >73 competitors") from infobox `competitors` fields gives 5, matching the gold answer. The benchmark is consistent with the infobox.

**Note the hidden set is weighted toward the hard types:** aggregation + superlative are 50% of hidden questions vs 31% of public. Those are exactly the types where plain RAG fails (you need 8–43 docs and a numeric comparison). Tune for them.

### Traps observed in the data

- **Venue ambiguity:** 317 distinct (venue, games) pairs, but Sydney Convention and Exhibition Centre hosted 57 events at 2000, Georgia World Congress Center 54 at 1996, Beijing National Stadium 37 at 2008. Venue alone is not enough; the date must disambiguate, and date strings are messy ("22 September 2000 (heats)25 September 2000 (final)", "11–19 August", "February 20–21, 1994").
- **Date field missing in 42% of event docs.** Some multi_hop questions may need the prose or results tables to recover the date.
- **Sport name in aggregation questions must map to the title prefix** ("cross-country skiing" → "Cross-country skiing at the …", "short-track speed skating" → "Short-track speed skating at the …").
- **Temporal questions rely on `prev`** but 6.6% of docs lack it, and the previous Games' event may have a slightly different title (weight classes change). Matching on sport + Games year + normalised event name is safer than title equality.
- **Non-numeric competitors** ("23 teams", empty) must be handled without crashing aggregation.
- **2020 Summer and 2022 Winter coverage is sparse** (44 and 19 docs), so "immediately before 2022/2020" temporal questions hit small sets.

---

## 8. Evaluation and judging

### What you must measure, per question, per pipeline

- **Accuracy:** correctness, completeness, grounding in available evidence. Scored against a held-out ground truth (the 50 hidden questions). Quality assessed via automated evaluation, LLM-as-judge, human review, and repository review.
- **Token efficiency:** context tokens, LLM input tokens, LLM output tokens, total tokens per answer.
- **Trace and agentic behaviour (Agentic pipeline only):** number of retrieval and reasoning steps; retrieval methods selected; specialised agents invoked; tools called; time per operation; tokens per operation; total tokens; number of chunks and citations; whether strategy changed mid-investigation; when and why it stopped.

Organisers' stated objective: "not simply to measure whether Agentic GraphRAG produces a better answer. It is to determine whether the additional reasoning and retrieval steps are worth the additional complexity and token cost."

For the hidden 50 you submit **raw outputs: tokens used, answers generated, and the agentic trace.** Exact submission format is not yet specified. Ask on Discord.

### Judging weights

| Criterion | Weight | What they look for |
|---|---|---|
| Investigation accuracy | **30%** | Correct, complete answers using the right evidence across steps |
| Evidence quality and explainability | 15% | Grounded answers, clear citations, visible investigation path |
| Agentic effectiveness and efficiency | 15% | Picks the right retrieval method, uses agentic steps only where they add value, balances accuracy vs token cost |
| Agentic design, engineering, code quality | 15% | Architecture, tool use, reliability, reproducibility, repo quality |
| Innovation | 15% | Novel investigation methods, graph reasoning, or UX |
| Final presentation and Q&A | 10% | Demo quality, technical clarity, answers to judges |

---

## 9. Deliverables

**Round 1 (everyone):**
1. Working Agentic GraphRAG system
2. GitHub repository
3. Architecture diagram
4. Demo video
5. Metrics dashboard comparing the three pipelines (tokens, accuracy, completeness) on the 100 public questions
6. Raw outputs for the 50 hidden questions (answers, tokens, agentic trace)
7. Optional: social media post tagging @TigerGraph ("counts in your favour")

**Round 2 (top 15):**
1. Refined system on the harder Round 2 dataset
2. Updated repo
3. Architecture diagram
4. 3–5 minute demo video
5. Metrics dashboard
6. Short write-up: what you built, how it works, key results, limitations, what you would add with more time
7. Live presentation to judges (top teams)

---

## 10. Strategic read: what the data implies for the build

This section is analysis, not organiser text.

1. **The benchmark is a structured-data problem wearing a text costume.** All five question types resolve from infobox fields. The natural graph schema is obvious: `Games(year, season)`, `Sport`, `Event(title, competitors, nations, date, win_value)`, `Venue`, `Athlete`, `NOC`, `Document(chunks, embedding)`, with edges `Event-HELD_AT->Venue`, `Event-PART_OF->Games`, `Event-IN_SPORT->Sport`, `Event-GOLD->Athlete`, `Event-PREV->Event`, `Event-NEXT->Event`, `Event-DESCRIBED_BY->Document`. Aggregation and superlative become one GSQL query. Temporal becomes a `PREV` hop. Multi-hop becomes venue+date pattern match.

2. **This is exactly the setup where plain RAG loses and GraphRAG wins, and the organisers know it.** Aggregation needs 8–43 documents; a top-k chunk retriever will not see them all, and the distractor films will pollute similarity search. Expect RAG to score well on `lookup` and poorly on `aggregation`/`superlative`. Your dashboard should show this per qtype, because "where each approach succeeds or fails" is the explicit ask.

3. **Where agentic genuinely earns its keep on this dataset:** (a) venue disambiguation when 57 events share a venue and the date string is messy, so the agent must retrieve, inspect, and re-query; (b) temporal questions where `prev` is missing or the event was renamed, requiring a fallback strategy; (c) events with empty `competitors` where the agent must decide to parse the results table or the prose; (d) recognising a `lookup` question and *not* spending agentic steps on it (this is scored under "agentic effectiveness"). A router that classifies qtype and picks the cheapest sufficient pipeline is directly rewarded by the rubric.

4. **Extraction quality decides accuracy.** Because infobox parsing is deterministic, a careful regex/structured loader beats LLM entity extraction here on both accuracy and tokens. Use the LLM only where structure is missing (date recovery, results-table parsing). Keep `doc_id`/`wikidata_qid` on every node so citations map straight to gold doc IDs.

5. **Answer normalisation is a hidden scoring risk.** Gold answers are exact strings: event titles with en-dashes, athlete names with diacritics, concatenated team names. Output the canonical corpus string, and record the doc ID beside it.

6. **Round 2 is about conflicting facts over time.** The `prev`/`next` chain and per-Games versions of the same event are the natural "evolving fact" structure. Design nodes with validity intervals and source authority now so the extension is cheap. Think: the same athlete's result across Games, or venue/date changes between sources.

7. **Metrics dashboard should be per-qtype, per-pipeline:** accuracy (exact and normalised), completeness, context/input/output/total tokens, steps, tools called, stop reason. That table alone hits three rubric rows.

---

## 11. What the previous TigerGraph hackathon looked like

The "GraphRAG Inference Hackathon by TigerGraph" (Unstop, earlier in 2026) asked for LLM-only vs basic RAG vs GraphRAG comparisons. Public write-ups from participants:

- **CyberGraph RAG** (MITRE ATT&CK / CISA KEV / NVD, 21k docs, 3.5M tokens): reported GraphRAG at 685 avg tokens / 3.8 s / 100% accuracy vs basic RAG 1,280 tokens / 6.45 s / 60% vs LLM-only 950 tokens / 10.15 s / 20%. Used Gemini for generation, multi-hop GSQL traversal, interactive graph visualisation.
- **MediGraph** (Disease → Symptom → Drug): Savanna + Groq + FAISS + Streamlit dashboard; GraphRAG used 125 tokens vs 813 for LLM-only (70–85% savings).

Pattern: the write-ups all led with a side-by-side metrics table, token savings, and a visual of the traversal. This hackathon adds the agentic layer and a fixed common benchmark, so the differentiator shifts from "graphs save tokens" (already proven) to "the agent chooses well and shows its trace".

---

## 12. Open questions to resolve with organisers

1. Exact submission format for the 50 hidden-question raw outputs (JSONL schema? trace format?).
2. How Savanna credits are issued (form? code in Discord?).
3. When the "Accuracy Evaluation Guide" and "Build Faster with MCP" pages will be published, and whether accuracy scoring is exact-match, normalised, or LLM-judged.
4. Whether registration closes Sep 12 or Sep 13/14. Register before Sep 12 regardless.
5. Whether Round 1 submission happens on Unstop or via a separate form.
6. Whether the GraphRAG repo (AGPL-3.0) must be used or is just a starter.

---

## 13. Sources

- Unstop listing: https://unstop.com/hackathons/agentic-graphrag-hackathon-tigergraph-1747871 (fetched via Playwright, Sep 5 2026)
- Notion guidebook: https://alluring-beryllium-491.notion.site/Agentic-GraphRAG-Hackathon-Guidebook-34fc2cb129c08146998af3568d7d2594
- Dataset folder: https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9 (README, corpus, both question files downloaded and parsed)
- TigerGraph GraphRAG repo: https://github.com/tigergraph/graphrag
- TigerGraph MCP: https://github.com/tigergraph/tigergraph-mcp
- GSQL vector search tutorial: https://github.com/tigergraph/ecosys/blob/master/tutorials/VectorSearch.md
- TigerGraph pricing / Savanna free tier: https://www.tigergraph.com/pricing/
- TigerGraph blog, "Agentic GraphRAG Gives AI a Playbook for Smarter Retrieval": https://tigergraph.com/blog/agentic-graphrag-gives-ai-a-playbook-for-smarter-retrieval/
- Previous hackathon listing: https://unstop.com/hackathons/graphrag-inference-hackathon-by-tigergraph-tigergraph-1678762
- CyberGraph RAG write-up: https://dev.to/bhuvi_d/how-we-built-cybergraph-rag-a-35m-token-cybersecurity-graphrag-system-with-tigergraph-5eon
- MediGraph write-up: https://dev.to/chinmayirhegde/how-i-built-a-graphrag-system-that-saves-70-85-llm-tokens-using-tigergraph-4i63
