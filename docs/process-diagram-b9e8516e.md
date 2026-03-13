# communis Architecture Diagrams

**Goal-driven agent with durable execution, tool use, subcommunis spawning, and human-in-the-loop approval.**

---

## 1. Full Orchestration Flow

The orchestrator runs a `while` loop. Each iteration, the planner chooses one of three actions: execute a step, spawn parallel subcommuniss, or signal goal completion.

```mermaid
flowchart TD
    subgraph INIT["INITIALIZATION"]
        CLI["CLI parses flags<br/>--provider --model --turns --dangerous --auto"]
        RC["CommunisConfig:<br/>idea, max_turns, model, provider,<br/>goal_complete_detection, max_subcommunis, dangerous"]
        IW["init_workspace activity<br/>Creates .communis/workflow-id/"]
        RIFF["Writes communis.md<br/>(YAML manifest)"]
        CLI --> RC --> IW --> RIFF
    end

    RIFF --> LOOP

    subgraph LOOP["WHILE LOOP (turn < effective_max)"]
        direction TB
        RCTX["read_turn_context activity<br/>Reads plan.md + summary.md + last 3 turn files"]
        BPC["_build_planner_context()<br/>idea + plan + summary + insights + step position<br/>+ subcommunis capability + approaching-limit warning"]
        PLAN["plan_next_turn activity<br/>PLANNER_PROMPT --> LLM --> JSON"]

        RCTX --> BPC --> PLAN

        DECIDE{{"Planner action?"}}
        PLAN --> DECIDE

        DECIDE -->|"goal_complete: true"| GOAL_DONE["state.goal_complete = True<br/>Break loop"]
        DECIDE -->|"action: step"| STEP
        DECIDE -->|"action: spawn"| SPAWN

        subgraph STEP["ACTION: STEP"]
            WPF["write_plan_file activity<br/>Updates plan.md"]
            CHILD["Execute CommunisTurnWorkflow<br/>(child workflow)"]
            WPF --> CHILD
        end

        subgraph SPAWN["ACTION: SPAWN"]
            WPF2["write_plan_file activity"]
            SUBS["_spawn_subcommunis()<br/>Start N child CommunisOrchestratorWorkflows<br/>in parallel (max_subcommunis cap)"]
            WAIT["await all subcommuniss"]
            SUMSA["summarize_subcommunis_results activity<br/>LLM condenses all subcommunis output"]
            WSS["write_subcommunis_summary activity<br/>Writes subcommunis-step-NN.md"]
            WPF2 --> SUBS --> WAIT --> SUMSA --> WSS
        end

        SUMCHECK{"turn_results > 4?"}
        STEP --> SUMCHECK
        SPAWN --> SUMCHECK

        SUMCHECK -->|Yes| COLLECT["collect_older_turns_text<br/>+ summarize_artifacts<br/>+ write_workspace_summary"]
        SUMCHECK -->|No| FEEDBACK
        COLLECT --> FEEDBACK

        FEEDBACK{{"auto mode?"}}
        FEEDBACK -->|Yes| RCTX
        FEEDBACK -->|No| WAIT_FB["Wait for user feedback signal<br/>(120s timeout)"]
        WAIT_FB --> VALIDATE["validate_user_feedback activity<br/>Guards against off-topic input"]
        VALIDATE --> RCTX
    end

    GOAL_DONE --> COMPLETE["state.status = complete"]

    style INIT fill:#1a1a2e,stroke:#16213e,color:#e2e2e2
    style LOOP fill:#0f3460,stroke:#16213e,color:#e2e2e2
    style STEP fill:#2c3e50,stroke:#34495e,color:#ecf0f1
    style SPAWN fill:#8e44ad,stroke:#9b59b6,color:#ecf0f1
```

---

## 2. Child Turn Workflow — Agent Loop with Tool Use

Each step runs as a child `CommunisTurnWorkflow`. The agent calls the LLM, which can invoke the `run` tool to execute shell commands. The loop repeats until the LLM stops requesting tools (max 20 iterations).

```mermaid
flowchart TD
    subgraph TURN["CommunisTurnWorkflow"]
        direction TB
        RCTX["read_turn_context activity<br/>Reads summary.md + last 3 turn files (full content)"]
        BUM["_build_user_message()<br/>Goal + Position + Summary + Recent Work + Feedback"]
        SYS["TURN_AGENT_PROMPT_WITH_TOOLS<br/>System: 'You are: {role} ... You have a run tool ...'"]

        RCTX --> BUM --> SYS

        subgraph AGENT_LOOP["AGENT LOOP (max 20 iterations)"]
            direction TB
            LLM["call_claude activity<br/>messages + system_prompt + tools --> LLM"]
            CHECK{{"stop_reason?"}}
            LLM --> CHECK

            CHECK -->|"end_turn<br/>(no tool_use)"| DONE_LOOP["Break -- agent is done"]
            CHECK -->|"tool_use"| TOOL_PROCESS

            subgraph TOOL_PROCESS["TOOL EXECUTION"]
                direction TB
                APPROVAL{{"--dangerous?"}}
                APPROVAL -->|Yes| EXEC["execute_run_command activity<br/>subprocess with shell=True"]
                APPROVAL -->|No| WAIT_APPROVE["Set pending_tool query<br/>Wait for approve_tool signal"]
                WAIT_APPROVE -->|Approved| EXEC
                WAIT_APPROVE -->|Denied| DENIED["Return denial message"]

                subgraph PRESENTATION["Presentation Layer"]
                    BINARY["Binary guard<br/>(rejects non-text output)"]
                    OVERFLOW["Overflow truncation<br/>(200 lines / 50KB cap)"]
                    FOOTER["Metadata footer<br/>[exit:N | Xms]"]
                    STDERR["Stderr attachment"]
                    BINARY --> OVERFLOW --> FOOTER --> STDERR
                end

                EXEC --> PRESENTATION
            end

            TOOL_PROCESS -->|"tool_result<br/>appended to messages"| LLM
        end

        SYS --> AGENT_LOOP

        EKI["extract_key_insights activity<br/>LLM --> JSON array of 3-5 insights"]
        WTA["write_turn_artifact activity<br/>Writes turn-NN-role.md<br/>(YAML frontmatter + content)"]
        RESULT["Return TurnResult<br/>(metadata only -- content in workspace file)"]

        DONE_LOOP --> EKI --> WTA --> RESULT
    end

    style TURN fill:#533483,stroke:#16213e,color:#e2e2e2
    style AGENT_LOOP fill:#2c3e50,stroke:#34495e,color:#ecf0f1
    style TOOL_PROCESS fill:#c0392b,stroke:#e74c3c,color:#ecf0f1
    style PRESENTATION fill:#d35400,stroke:#e67e22,color:#ecf0f1
```

---

## 3. Sub-Agent Spawning

When the planner returns `action: "spawn"`, the orchestrator starts parallel child orchestrator workflows. Subcommuniss have `max_subcommunis=0` to prevent recursion.

```mermaid
flowchart TD
    PARENT["Parent CommunisOrchestratorWorkflow<br/>communis-abc123"]

    PARENT -->|"Planner: action=spawn"| SPAWN_FN["_spawn_subcommunis()"]

    SPAWN_FN -->|"child orchestrator"| SA1["CommunisOrchestratorWorkflow<br/>communis-abc123-subcommunis-3-0<br/>task: 'Research API docs'<br/>max_subcommunis=0, auto=True"]
    SPAWN_FN -->|"child orchestrator"| SA2["CommunisOrchestratorWorkflow<br/>communis-abc123-subcommunis-3-1<br/>task: 'Analyze user model'<br/>max_subcommunis=0, auto=True"]

    SA1 -->|"own turn loop"| SA1T1["Turn 1"]
    SA1T1 --> SA1T2["Turn 2<br/>goal_complete"]
    SA2 -->|"own turn loop"| SA2T1["Turn 1<br/>goal_complete"]

    SA1T2 --> COLLECT["Collect SubAgentResults"]
    SA2T1 --> COLLECT

    COLLECT --> SUMMARIZE["summarize_subcommunis_results activity<br/>LLM condenses findings"]
    SUMMARIZE --> WRITE["write_subcommunis_summary<br/>subcommunis-step-03.md"]
    WRITE --> CONTINUE["Parent continues<br/>next planner iteration"]

    subgraph SUB_CONFIG["Sub-Agent Config (inherited from parent)"]
        direction LR
        SC1["model: parent.model"]
        SC2["provider: parent.provider"]
        SC3["dangerous: parent.dangerous"]
        SC4["auto: True (always)"]
        SC5["max_subcommunis: 0 (no recursion)"]
        SC6["goal_complete_detection: True"]
    end

    style PARENT fill:#3498db,color:#fff
    style SA1 fill:#8e44ad,stroke:#9b59b6,color:#fff
    style SA2 fill:#8e44ad,stroke:#9b59b6,color:#fff
    style SUB_CONFIG fill:#34495e,color:#ecf0f1
```

---

## 4. Workspace File Structure

```
.communis/<workflow-id>/
  communis.md                       # Session manifest (YAML: idea, max_turns, model)
  plan.md                       # Rolling plan summary (updated each step by planner)
  turn-01-researcher.md         # Turn artifact (YAML frontmatter + full content)
  turn-02-implementer.md
  subcommunis-step-03.md          # LLM summary of subcommunis results from step 3
  subcommunis/                    # Subcommunis workspaces
    <id>-subcommunis-3-0/          # Each subcommunis gets its own full workspace
      communis.md
      plan.md
      turn-01-worker.md
    <id>-subcommunis-3-1/
      ...
  turn-04-synthesizer.md
  summary.md                    # Rolling summary (replaces old turns once > MAX_RECENT+1)
```

---

## 5. Sliding Context Window

What each step's LLM calls can see (MAX_RECENT_ARTIFACTS = 3):

```mermaid
flowchart LR
    subgraph WINDOW["Context Window Per Step"]
        direction TB

        W1["Step 1: no prior work"]
        W2["Step 2: turn-01"]
        W3["Step 3: turn-01 + turn-02"]
        W4["Step 4: turn-01 + turn-02 + turn-03"]
        W5["Step 5: turn-02 + turn-03 + turn-04<br/>(turn-01 exits window)"]
        W6["Step 6: summary.md + turn-03 + turn-04 + turn-05<br/>(turns 1-2 compressed)"]
        W7["Step 7: summary.md + turn-04 + turn-05 + turn-06<br/>(turns 1-3 compressed)"]

        W1 --> W2 --> W3 --> W4 --> W5 --> W6 --> W7
    end

    style WINDOW fill:#1a1a2e,color:#e2e2e2
```

The **planner** sees: idea + plan.md + summary + recent turn **insights** (metadata only).
The **turn agent** sees: idea + summary + recent turn **full content** + user feedback.

---

## 6. Prompt Anatomy — 7 LLM Prompts

```mermaid
flowchart TD
    subgraph PROMPTS["LLM PROMPTS"]
        direction TB

        subgraph P1["PLANNER_PROMPT (orchestrator, per-step)"]
            P1D["Input: goal + plan.md + summary + insights + step position + subcommunis budget<br/>Output: JSON with action (step / spawn / goal_complete)<br/>Decides WHAT to do next"]
        end

        subgraph P2["TURN_AGENT_PROMPT_WITH_TOOLS (child workflow, per-step)"]
            P2D["Input: goal + summary + recent turns full content + position<br/>Has run tool for shell commands<br/>Output: concrete work product (code, files, analysis)<br/>Does THE ACTUAL WORK"]
        end

        subgraph P3["EXTRACT_INSIGHTS_PROMPT (child workflow, post-step)"]
            P3D["Input: step's full output content<br/>Output: JSON array of 3-5 insight strings<br/>Creates METADATA for future planner decisions"]
        end

        subgraph P4["SUMMARIZE_ARTIFACTS_PROMPT (orchestrator, periodic)"]
            P4D["Input: full text of older turns being compressed<br/>Output: prose summary for summary.md<br/>COMPRESSES old turns to fit context window"]
        end

        subgraph P5["VALIDATE_FEEDBACK_PROMPT (orchestrator, on feedback)"]
            P5D["Input: goal + user feedback text<br/>Output: JSON {relevant, reason}<br/>Guards against off-topic human input"]
        end

        subgraph P6["SUMMARIZE_SUBAGENT_RESULTS_PROMPT (orchestrator, after spawn)"]
            P6D["Input: all subcommunis task/status/summary pairs<br/>Output: condensed summary for parent context<br/>Bridges subcommunis work back to parent"]
        end

        subgraph P7["APPROACHING_LIMIT / FINAL_TURN ADDENDA (orchestrator)"]
            P7D["Injected into planner context when within 2 steps of max<br/>or on the final step<br/>Signals urgency to wrap up or synthesize"]
        end
    end

    subgraph FLOW["HOW THEY CONNECT"]
        direction LR
        A["Planner<br/>decides action"] -->|"step: role + instructions"| B["Turn Agent<br/>executes with tools"]
        A -->|"spawn: tasks"| E["Sub-Agents<br/>(own turn loops)"]
        E -->|"results"| F["Sub-Agent Summarizer"]
        F -->|"subcommunis-step-N.md"| A
        B -->|"content"| C["Insight Extractor"]
        C -->|"insights in turn file"| A
        B -->|"content accumulates"| D["Artifact Summarizer"]
        D -->|"summary.md"| A
        D -->|"summary.md"| B
    end

    style P1 fill:#3498db,stroke:#2980b9,color:#fff
    style P2 fill:#e74c3c,stroke:#c0392b,color:#fff
    style P3 fill:#2ecc71,stroke:#27ae60,color:#fff
    style P4 fill:#e67e22,stroke:#d35400,color:#fff
    style P5 fill:#95a5a6,stroke:#7f8c8d,color:#fff
    style P6 fill:#8e44ad,stroke:#9b59b6,color:#fff
    style P7 fill:#f39c12,stroke:#e67e22,color:#fff
    style FLOW fill:#1a1a2e,color:#e2e2e2
```

---

## 7. Temporal Workflow Hierarchy

```mermaid
flowchart TD
    subgraph TEMPORAL["Temporal Durable Execution"]
        O["CommunisOrchestratorWorkflow<br/>(parent -- runs the while loop)"]

        O -->|"child workflow"| T1["CommunisTurnWorkflow<br/>-turn-1"]
        O -->|"child workflow"| T2["-turn-2"]

        O -->|"child orchestrator"| SA1["CommunisOrchestratorWorkflow<br/>-subcommunis-3-0"]
        O -->|"child orchestrator"| SA2["-subcommunis-3-1"]

        SA1 -->|"child workflow"| SA1T1["CommunisTurnWorkflow<br/>-subcommunis-3-0-turn-1"]
        SA2 -->|"child workflow"| SA2T1["-subcommunis-3-1-turn-1"]

        O -->|"child workflow"| T4["-turn-4"]

        subgraph ACTIVITIES["Activities (15 total)"]
            direction TB
            subgraph LLM_ACT["LLM Activities"]
                A3["plan_next_turn"]
                A4["call_claude"]
                A5["extract_key_insights"]
                A8["summarize_artifacts"]
                A10["validate_user_feedback"]
                A11["summarize_subcommunis_results"]
            end
            subgraph TOOL_ACT["Tool Activities"]
                A12["execute_run_command"]
            end
            subgraph IO_ACT["Workspace I/O Activities"]
                A1["init_workspace"]
                A2["read_turn_context"]
                A6["write_turn_artifact"]
                A7["collect_older_turns_text"]
                A9["write_workspace_summary"]
                A13["write_plan_file"]
                A14["write_subcommunis_summary"]
            end
        end
    end

    FS[(".communis/workflow-id/<br/>File System Workspace")]

    IO_ACT -->|"read/write"| FS

    style O fill:#3498db,color:#fff
    style SA1 fill:#8e44ad,color:#fff
    style SA2 fill:#8e44ad,color:#fff
    style FS fill:#2ecc71,stroke:#27ae60,color:#fff
    style LLM_ACT fill:#e74c3c,stroke:#c0392b,color:#ecf0f1
    style TOOL_ACT fill:#d35400,stroke:#e67e22,color:#ecf0f1
    style IO_ACT fill:#27ae60,stroke:#2ecc71,color:#ecf0f1
```

---

## 8. CLI Modes

| Flag | Effect |
|------|--------|
| `--turns 0` (default) | Indefinite with goal detection (capped at 50) |
| `--turns N` | Max N steps with goal detection |
| `--no-goal-detect` | Fixed N steps, no early exit (requires `--turns > 0`) |
| `--dangerous` | Auto-approve all tool calls |
| `--auto` | Skip user feedback prompts |
| `--max-subcommunis N` | 0-5, default 3 (0 disables spawning) |
| `--provider openai` | Use OpenAI-compatible API (LM Studio, vLLM, etc.) |
| `--base-url URL` | Override OpenAI base URL |
| `--model MODEL` | Model for all LLM calls (planner, agent, insights, summaries) |

## 9. LLM Provider Support

Both Anthropic and OpenAI-compatible providers are supported. The `_call_openai` path converts Anthropic-format messages (tool_use/tool_result content blocks) to OpenAI format (tool_calls on assistant messages, role=tool for results) transparently. All activities accept `provider`, `base_url`, and `model` parameters passed from the CLI through `CommunisConfig`.

| Provider | Use Case | Message Format |
|----------|----------|----------------|
| `anthropic` (default) | Claude API direct | Native Anthropic format |
| `openai` | LM Studio, vLLM, OpenRouter, OpenAI | Auto-converted from internal Anthropic format |
