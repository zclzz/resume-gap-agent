# resume-gap-agent

A project for the Advance NLP course by RDAI.

An agentic system that ingests an applicant's resume (PDF or text) and a target
role, sources live job postings for that role at the requested seniority level,
extracts required skills across the postings, and produces a **prioritized gap
report**.

## The demo (`demo.ipynb`)

A sample workflow can be viewed in [`demo.ipynb`](demo.ipynb). It runs the full agentic pipeline with live Adzuna postings, an OpenAI-driven sourcing agent and
critic over MCP tools, real LLM extraction — on three resumes at three
different seniority levels, each producing its own gap report in its own cell:

| Resume | Target role | Level |
|---|---|---|
| `senior-accountant-elegant-resume-example.pdf` | Accountant | senior |
| `software_engineer_resume.pdf` | Software Engineer | mid |
| `urban_design_resume.pdf` | Urban Designer | entry |

Resumes are loaded from `data/` as PDFs (`pypdf` text-layer extraction; scanned
image-only PDFs are rejected with a clear error). Each cell prints the
structured log stream as the graph runs — watch `sourcing_agent.done` and
`critic_agent.done` for the LLM's decisions and whether the pipeline looped —
followed by the postings table, an emoji severity summary, and a gaps
DataFrame with remediation suggestions.

## Architecture

Two layers:

1. **Sourcing layer** (`sources/`) — retrieves and normalizes external postings
   behind the `JobSource` ABC. `MockJobSource` (fixtures) and `AdzunaJobSource`
   (live). A deterministic, level-aware title filter (`matches_level`) enforces
   the requested seniority (entry / mid / senior) on whatever the source returns.
2. **Analysis engine** (`graph/`) — a role- and seniority-agnostic LangGraph
   pipeline that consumes normalized postings + resume and emits a `GapReport`.

### The agentic loop

The pipeline has **two nested agentic loops** around an otherwise deterministic
LangGraph topology:

```text
                                     ┌───── outer loop: critic wants more coverage ────┐
                                     ▼                                                 │
parse_resume → parse_target_role → source_postings → extract → aggregate → gap_diff → critic ──(ok)──► report
                                     │         ▲
                                     └─────────┘  inner ReAct loop: the agent calls MCP tools
                                                  (search_jobs, get_posting_details, ...) until satisfied
```

- **Inner loop — the sourcing agent** (`graph/agents.py`). A LangGraph ReAct
  agent (`create_react_agent`) that decides how to phrase the search query,
  whether to reword and re-search when results look off-target, and which
  postings to keep — by calling MCP tools. Bounded by `agent_recursion_limit`,
  with a plain-search fallback if its final answer is unparseable.
- **Outer loop — the LLM critic** (`graph/agents.py`). Judges via structured
  output whether the analysis rests on enough posting coverage to be
  trustworthy (count vs. target/minimum, coherence of the top skills, whether
  sourced roles match the target). If not, a conditional edge routes the graph
  back to sourcing with a concrete refinement hint. Bounded by
  `max_critic_loops` — the code ANDs the LLM's vote with the hard cap, so the
  critic can be outvoted (visible in the logs as `llm_needs_more=True,
  capped_needs_more=False`).

The pattern throughout: **the LLM proposes, deterministic code enforces.**
Extraction, aggregation, the gap diff, the seniority filter, and every loop
bound are plain Python.

### MCP services (`mcp_server.py`)

In agentic mode the sourcing agent's tools are served by an MCP server running
as a **separate process over stdio** (`FastMCP`). Tools:

- `search_jobs(role, level, location, limit)` — explore postings (compact results)
- `get_posting_details(posting_ids)` — full text of specific postings
- `normalize_skill_name(name)` — canonicalize a skill string via the taxonomy
- `list_known_skills()` — the taxonomy's canonical skill names

The LangGraph agent connects as an MCP client via `langchain-mcp-adapters`
(`graph/mcp_client.py`), which spawns the server subprocess and converts its
MCP tools into LangChain tools. The server wraps the same `JobSource` backends,
so mock (offline) or Adzuna (live) postings flow through identical tools. Tools
are stateless across calls — live search results are persisted to a small
on-disk cache so a `posting_id` resolves even from a fresh server subprocess —
and the server logs to stderr only, because stdout is the JSON-RPC transport.

### LangGraph specifics

- **Typed state graph** — `StateGraph(GraphState)` with Pydantic state; each
  node returns a partial update dict (`graph/build.py`, `graph/state.py`).
- **Conditional edge** — the single loop-back edge `critic → source_postings`
  vs. `critic → report` (`route_after_critic`), which is what makes the graph
  cyclic rather than a straight pipeline.
- **Prebuilt ReAct sub-agent** — `create_react_agent` runs *inside* the
  sourcing node with its own recursion limit and no checkpointer, so the inner
  loop's chatter never pollutes the outer graph's state.
- **Checkpointing** — compiled with a `SqliteSaver`; a file-backed checkpointer
  lets a run resume after failure without re-sourcing or re-calling the LLM for
  completed nodes (the demo uses `:memory:` since each run is fresh).
