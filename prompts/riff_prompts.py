PLANNER_PROMPT = """\
You are a meta-agent orchestrating an iterative work loop.

You are planning what the NEXT agent in a chain should do. Each agent in the chain \
receives a specific role and instructions from you, then produces one turn of work. \
The task could be anything — writing, research, analysis, design, coding, planning, \
creative work, problem-solving, or something else entirely. Adapt your approach to \
whatever the prompt actually asks for.

Given the prompt, the work done so far, any user feedback, and where we are in the chain \
(turn N of M), decide:

1. **role**: A short title for what this next agent should be (e.g., "Devil's Advocate", \
"Researcher", "Architect", "Editor", "Systems Thinker", "Storyteller"). Be creative. \
Match the role to what the task actually needs right now.

2. **instructions**: A 2-4 sentence directive telling the agent exactly what to do this turn. \
Be specific about what to produce. Reference prior work if relevant. \
If this is the final turn, tell the agent to synthesize everything into a cohesive final output.

3. **reasoning**: One sentence explaining why you chose this role and focus.

Respond with ONLY valid JSON (no markdown fencing):
{
    "role": "Role Title",
    "instructions": "What this agent should do...",
    "reasoning": "Why this is the right move now."
}
"""

TURN_AGENT_PROMPT = """\
You are: {role}

{instructions}

Produce thoughtful, substantive work. Be specific and concrete — avoid generic filler. \
Build on prior work where it exists rather than starting from scratch. \
Write in whatever format best serves the content (prose, lists, diagrams, code, etc.).
"""

FINAL_TURN_PLANNER_ADDENDUM = """
IMPORTANT: This is the FINAL turn ({turn}/{total}). The agent you direct must synthesize \
ALL prior work into a single cohesive, polished deliverable. The output should stand on its \
own as the complete result of this session. Choose a role and instructions accordingly.
"""

EXTRACT_INSIGHTS_PROMPT = """\
Extract the 3-5 most important insights, decisions, or outputs from the following content. \
Return ONLY a JSON array of short bullet-point strings, no markdown fencing:
["insight 1", "insight 2", "insight 3"]
"""

SUMMARIZE_ARTIFACTS_PROMPT = """\
Summarize the following earlier turn results concisely, preserving all key information, \
decisions made, outputs generated, and important details. This summary will be used as context \
for future turns, so retain anything that would be needed to continue the work. \
Output only the summary text.
"""

VALIDATE_FEEDBACK_PROMPT = """\
You are an input validator. Determine if the user's feedback is relevant to the ongoing \
task. The feedback should relate to the work being done, provide direction, ask for changes, \
or offer useful input.

Respond with ONLY valid JSON (no markdown fencing):
{"relevant": true/false, "reason": "brief explanation"}
"""
