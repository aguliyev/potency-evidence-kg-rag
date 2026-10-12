Agents stop: This file is for humans only.


------------------------


Make Docker Compose and two Docker files. 
One Docker container will be Jupiter notebook. 
Another Docker container will be Postgress database 
with PG vector and Apache Age, which allows for a graph database.

make bin/ with Shell Scripps to `install`, `start` and `stop` our system.
And make a lib/ directory for code which will be used in the notebook. 

this lib/ and notebooks/ should be mapped local directories to the Jupiter container 
so that I can change the code and Jupiter sees the updated files.

Starting the Postgres container should also run migrations. Which Python package do you suggest for that? Should we use SQL alchemy and alambic or something else? 

----------------

you have a table for `alias_candidate`, 
but probably it should be `aliases` with a status pending/approved/rejected, 
so that candidates can be easily made functioning aliases.
and you have some aliases in python code in project_profiles.py ,  let us make it a json file instead, 
with aliases we know for all tenants/projects, and it be loaded into the `aliases` table.

also, suggest if we can use a "corpus-wide" (probably per document) pass to identify more aliases, and load them into the `aliases` table for review.
This could be with LLM, but also some deterministic algorithms?

so, check below and make implementation plan.


Here is a high-level plan and instruction blueprint for adding the **Global Alias Management and Pre-Processing Discovery** stage to your knowledge graph system, positioning it right before the main KG-building workflow.

# High-Level Implementation Plan: Global Alias Management & Pre-Processing Discovery

## 1. Objective

Introduce a dedicated pre-processing phase that establishes a centralized, database-backed `aliases` table (supporting `pending`, `approved`, and `rejected` statuses) populated via two primary sources:

1. **Static Seed Data:** A version-controlled JSON configuration file containing known domain aliases (products, standard assay abbreviations).
2. **Dynamic Discovery:** A lightweight corpus-wide / document-level pre-processing pass combining deterministic pattern matching (e.g., acronyms, parentheses definitions) and a targeted LLM/heuristic extraction pass to mine unmapped synonym candidates before full semantic extraction begins.

---

## 2. Architecture & Data Flow Changes

```text
[Supplied Raw Corpus] 
       │
       ▼
[Pre-Processing Alias Discovery Pass] ◄── [Static JSON Alias Config]
       │
       ▼
[Unified `aliases` Table (`pending` / `approved` / `rejected`)]
       │
       ▼
[KG Extraction & Entity Resolution Engine (Consumes Approved Aliases)]

```

* **Storage Evolution:** Replace the isolated `alias_candidate` table and Python hardcoded dictionaries with a unified relational `aliases` table.
* **Execution Phase:** Insert this phase immediately *after* the initial structural import (Stage 1) and *before* Jev classification and Instructor extraction (Stage 3).

---

## 3. Implementation Steps (High-Level Task Breakdown)

### Step 1: Database Schema Migration (`aliases` table)

* Create an Alembic migration to establish the unified `aliases` table.
* **Key Fields:** `canonical_name`, `alias_value`, `status` (`pending`, `approved`, `rejected`), `source_type` (`static_config`, `deterministic_rule`, `llm_mined`), provenance references, and timestamps.

### Step 2: Externalize Static Aliases to JSON

* Create a structured JSON configuration file (e.g., `lib/potency_kg/config_data/global_aliases.json`) containing core domain synonyms.
* Write a bootstrapping loader function that imports these entries into the `aliases` table with an initial status of `approved`.

### Step 3: Implement Pre-Processing Document-Level Discovery

* **Deterministic Rules:** Write lightweight text-processing utilities to scan document passages for high-confidence synonym patterns (e.g., matching acronym definitions like *"exagamglogene autotemcel (exa-cel)"*).
* **LLM / Heuristic Pass (Optional/Configurable):** Implement a batch-processing step over document chunks to surface candidate synonyms.
* **Auto-Routing:** Automatically insert mined entries into the `aliases` table with a `pending` status for future review, or immediately leverage `approved` aliases during entity resolution.

### Step 4: Integrate into the Resolution & Extraction Pipeline

* Update `resolution.py` to query the `aliases` table (filtering for `status = 'approved'`) when normalizing entity mentions, ensuring all downstream extractions cleanly map synonyms to their canonical parent nodes.


----------------



