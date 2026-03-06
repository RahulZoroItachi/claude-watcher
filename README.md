# claude-watcher

Understand your Claude Code costs and usage patterns.
**Your data never leaves your machine** — everything reads from `~/.claude/` locally.

## Run

```bash
npx claude-watcher
```

That's it. No install step. `npx` downloads and runs the tool in one command.

Requires **Python 3** (any version ≥ 3.8) — the tool installs `streamlit` and `pandas` automatically if they're missing.

Opens a dashboard at `http://localhost:7755`.

To use a different port:

```bash
PORT=8080 npx claude-watcher
```

### Or install globally

If you run it often:

```bash
npm install -g claude-watcher
claude-watcher
```

## What it shows

- **How you use Claude** — which tools Claude invokes most, what you ask about, when you're most active
- **Where the money goes** — cost breakdown by token type and model
- **Key cost drivers** — automatically flagged patterns that are inflating your bill
- **Break & continue** — sessions where you returned after a break and carried old context forward
- **Idle resets** — turns where the cache expired and had to be rewritten at full price
- **Large early content** — files or logs pasted early that rode in context for the whole session
- **By project / day** — spend across your different codebases

## Privacy

This tool reads only the JSONL files Claude Code writes to `~/.claude/` on your local machine.
No data is sent anywhere. No API calls are made. No telemetry (Streamlit's usage stats are disabled).

## How costs are calculated

Costs are **API-equivalent estimates** based on token counts logged by Claude Code and Anthropic's published pricing.
If you're on a subscription plan (Pro/Max), you are not billed per token — but these numbers show what your usage would cost at API rates, and where the heaviest usage is concentrated.
