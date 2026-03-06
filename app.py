#!/usr/bin/env python3
"""
Claude Code — Cost & Usage Analyzer  (Streamlit UI)
Run:  streamlit run app.py --server.port 7755
"""

import streamlit as st
import pandas as pd
from analyze import (
    compute, get_pricing, fmt_cost, fmt_tok, fmt_mins, trunc,
    CACHE_RESET_GAP, NEAR_FULL_THRESHOLD, CONTEXT_LIMIT, SPLIT_GAP_MINS,
)

st.set_page_config(page_title="Claude Code — Cost Analyzer", layout="wide")

@st.cache_data(ttl=300)
def get_data():
    return compute()

with st.spinner("Loading…"):
    data = get_data()

if data is None:
    st.error("No session data found under ~/.claude/")
    st.stop()

sessions         = data["sessions"]
subagents        = data["subagents"]
all_turns        = data["all_turns"]
all_resets       = data["all_resets"]
tot_cost         = data["tot_cost"]
sa_cost          = data["sa_cost"]
c_inp            = data["c_inp"]
c_out            = data["c_out"]
c_cr             = data["c_cr"]
c_cw             = data["c_cw"]
tot_inp          = data["tot_inp"]
tot_out          = data["tot_out"]
tot_cr           = data["tot_cr"]
tot_cw5          = data["tot_cw5"]
date_from        = data["date_from"]
date_to          = data["date_to"]
all_sa_sessions  = data["all_sa_sessions"]
sa_turns         = data["sa_turns"]
model_stats      = data["model_stats"]
cost_no_cache    = data["cost_no_cache"]
saving           = data["saving"]
hit_rate         = data["hit_rate"]
reset_actual     = data["reset_actual"]
reset_warm       = data["reset_warm"]
fix_1h           = data["fix_1h"]
hour_counts      = data["hour_counts"]
full_sessions    = data["full_sessions"]
total_full_turns = data["total_full_turns"]
total_full_cost  = data["total_full_cost"]
all_splits       = data["all_splits"]
all_artifacts    = data["all_artifacts"]
all_high_out     = data["all_high_out"]
sorted_proj      = data["sorted_proj"]
sorted_days      = data["sorted_days"]
sa_c_cw          = data["sa_c_cw"]
sa_c_cr          = data["sa_c_cr"]

split_carry = sum(e["carry_cost"] for e in all_splits)
art_carry   = sum(e["carry_cost"] for e in all_artifacts)
hiout_carry = sum(e["carry_cost"] for e in all_high_out)
reset_waste = reset_actual - reset_warm

# ── Tooltips (plain English for all jargon) ───────────────────────────────────
TT = {
    "total_cost":
        "Total estimated API-equivalent cost based on tokens used. "
        "Subscription users aren't billed per token, but this tells you what your usage would cost at API rates.",
    "storing_history":
        "Every Claude turn stores the entire conversation history in a temporary cache. "
        "This is the biggest cost because the history grows with every message.",
    "reading_history":
        "On every turn, Claude reads back the full conversation history from cache. "
        "It's cheap per token ($0.30/M) but happens on every single turn — so it adds up.",
    "responses":
        "Cost of Claude's actual replies. Expensive per token ($15/M) but responses are small "
        "compared to the history being carried.",
    "your_messages":
        "Cost of your actual prompts. Usually the smallest cost because your messages are short.",
    "saving":
        "Without caching, every turn would re-pay full price for the entire conversation history. "
        "Caching means you pay a fraction of that on repeat reads.",
    "hit_rate":
        "What fraction of all context tokens were read from cache (cheap) vs freshly processed (expensive). "
        "Higher is better.",
    "carry_cost":
        "When you take a break and come back to the same session, all the context from before the break "
        "keeps riding in every subsequent turn — even if that work is done. "
        "This is the cost of that stale context being read back repeatedly.",
    "resets":
        "When you go idle for 5+ minutes, the conversation cache expires. "
        "The next turn has to re-store the full history at write price instead of just reading it. "
        "This is more expensive than a normal turn.",
    "full_ctx":
        "Claude has a memory limit (~166K tokens). When a session fills it up, "
        "every turn pays the maximum possible context-read cost.",
    "artifact":
        "A large block of content (pasted error log, file read, tool output) added early in a session "
        "that then rides in context for every subsequent turn.",
    "subagent":
        "Claude Code sometimes launches sub-processes (subagents) to handle tasks in parallel. "
        "Each subagent is a separate session with its own token costs.",
}

# ── Rule-based action engine ──────────────────────────────────────────────────
# Evaluates your usage data against thresholds and produces plain-English actions.

def build_actions(d, split_carry, art_carry, reset_waste, c_cr, tot_cost, sa_cost):
    actions = []  # (icon, headline, detail, severity, info)

    total = d["tot_cost"] + d["sa_cost"]

    # Mid-session gaps — biggest lever
    if split_carry > 0 and split_carry / c_cr > 0.20:
        pct = split_carry / c_cr * 100
        actions.append((
            "🔴",
            f"Start a new session when switching tasks  →  saves ~{fmt_cost(split_carry)}",
            f"You returned to existing sessions after breaks {len(d['all_splits'])} times. "
            f"The old conversation history kept being read on every new turn. "
            f"That accounts for {pct:.0f}% of your context-read cost.",
            "red",
            {
                "what":     "When you return to an existing session after a break, the full conversation "
                            "history from before the break stays in memory and gets read on every new turn — "
                            "even if that earlier work is completely done.",
                "computed": "For each gap ≥30 min followed by 5+ more turns: "
                            "context_size_before_gap × turns_until_next_gap × $0.30/M. "
                            "Each turn is counted once (no overlap between gaps).",
                "reduce":   "After a break, check if you're starting a new task. "
                            "If yes, run /clear or open a new session. "
                            "A fresh session starts with ~13K tokens vs your accumulated history.",
            },
        ))

    # Context storage dominant
    if d["c_cw"] / total > 0.40:
        pct = d["c_cw"] / total * 100
        actions.append((
            "🔴",
            f"Your sessions are running too long  —  history storage = {pct:.0f}% of bill",
            f"Storing the growing conversation history costs {fmt_cost(d['c_cw'])}. "
            f"The longer a session runs, the more expensive each turn becomes.",
            "red",
            {
                "what":     "After each turn, Claude stores the full conversation history in a cache. "
                            "This write cost ($3.75/M tokens) grows with every message added to the session.",
                "computed": "Sum of cache_write tokens × $3.75/M across all turns. "
                            "Flagged when this exceeds 40% of total bill.",
                "reduce":   "Keep sessions focused on one task. "
                            "Use /compact to summarise history mid-session. "
                            "Start a new session when switching to a different task.",
            },
        ))

    # Full context sessions
    if full_sessions:
        actions.append((
            "🟠",
            f"{len(full_sessions)} sessions maxed out Claude's memory  →  {fmt_cost(total_full_cost)} extra",
            f"{total_full_turns} turns ran with memory completely full. "
            f"Every one of those turns paid the maximum possible read cost.",
            "orange",
            {
                "what":     "Claude's context window is ~166K tokens. "
                            "When a session fills it, every subsequent turn reads back the maximum "
                            "amount of history — the most expensive state to be in.",
                "computed": "Turns where cache_read > 90% of 166K limit. "
                            "Cost = those turns' cache_read tokens × $0.30/M.",
                "reduce":   "Run /compact when memory is ~50% full. "
                            "It summarises the history and resets the context size, "
                            "roughly halving the read cost for the rest of the session.",
            },
        ))

    # Cache resets
    if reset_waste > 5:
        pct_fixable = fix_1h / len(d["all_resets"]) * 100 if d["all_resets"] else 0
        actions.append((
            "🟠",
            f"Idle gaps caused {len(d['all_resets'])} cache expiries  →  {fmt_cost(reset_waste)} wasted",
            f"Going idle for 5+ minutes lets the cache expire. "
            f"The next turn re-writes the full history at 12.5× the normal read cost. "
            f"{pct_fixable:.0f}% of these were under 60 min.",
            "orange",
            {
                "what":     "The conversation cache expires after 5 minutes of inactivity. "
                            "When you return, the next turn must re-write the entire history "
                            "at cache-write price ($3.75/M) instead of just reading it ($0.30/M).",
                "computed": "A reset is detected when cache_read drops to <50% of the previous "
                            "turn's value after a 5+ min gap. "
                            "Waste = actual cost of that turn − cost if cache had been warm.",
                "reduce":   "Stay active within sessions to avoid expiry. "
                            f"{pct_fixable:.0f}% of your resets were under 60 min — "
                            "a Max plan (1h cache TTL) would eliminate those.",
            },
        ))

    # Early artifacts
    if art_carry > 1.0:
        actions.append((
            "🟠",
            f"Large content added early in {len(d['all_artifacts'])} sessions  →  {fmt_cost(art_carry)} carry cost",
            f"A large block of content (pasted log, file read, tool output) was added in the "
            f"first 15 turns and rode in memory for the rest of the session.",
            "orange",
            {
                "what":     "When large content is introduced early in a session — pasting an error log, "
                            "Claude reading a big file, a long tool output — it becomes part of the "
                            "cached history and gets read back on every subsequent turn.",
                "computed": "First turn in the opening 15 where cache_write spike > 10K above a 1K baseline. "
                            "Carry cost = extra tokens × remaining turns × $0.30/M. "
                            "One event per session (earliest spike only).",
                "reduce":   "After handling a one-off task (fix an error, read a file), "
                            "start a new session so that content stops riding. "
                            "We can't tell if it was necessary — but the cost of carrying it is shown.",
            },
        ))

    # Subagents notable
    if d["sa_cost"] / total > 0.05:
        actions.append((
            "🟡",
            f"Sub-processes cost {fmt_cost(d['sa_cost'])}  ({d['sa_cost']/total*100:.0f}% of total)",
            f"{len(d['all_sa_sessions'])} background sub-processes were launched across "
            f"{len(d['subagents'])} main sessions.",
            "yellow",
            {
                "what":     "Claude Code's Task tool spawns sub-sessions to handle work in parallel "
                            "(e.g. running tests, searching files). Each sub-session is independent "
                            "and builds its own conversation history.",
                "computed": "Sessions found under ~/.claude/projects/*/subagents/*.jsonl. "
                            "Cost calculated identically to main sessions.",
                "reduce":   "Sub-process costs are usually necessary. "
                            "High cost here typically means the parent session ran many parallel tasks. "
                            "Check the Sub-processes tab to see which sessions spawned the most.",
            },
        ))

    # Positive
    actions.append((
        "🟢",
        f"Caching saved you {fmt_cost(d['saving'])}  ({d['saving']/d['cost_no_cache']*100:.0f}% off)",
        f"Without caching you'd have paid {fmt_cost(d['cost_no_cache'])}. "
        f"Caching makes repeated history reads 10× cheaper than the first write.",
        "green",
        {
            "what":     "Claude caches the conversation history after each turn. "
                        "Subsequent turns read from that cache at 0.1× the normal input price "
                        "instead of re-processing the full history from scratch.",
            "computed": "cost_no_cache = all context tokens × $3.00/M (full input rate). "
                        "Saving = cost_no_cache − actual cost.",
            "reduce":   "Nothing to do here — this is caching working as intended. "
                        "The saving grows the longer your sessions run.",
        },
    ))

    return actions

actions = build_actions(data, split_carry, art_carry, reset_waste, c_cr, tot_cost, sa_cost)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Claude Code — Usage & Cost")
st.caption(f"{date_from} → {date_to}  ·  {len(sessions)} conversations  ·  {len(all_turns):,} turns")
st.info("🔒 Your data never leaves your machine — everything is read directly from `~/.claude/` on this computer. No API calls are made and no data is sent anywhere.", icon=None)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Cost", fmt_cost(tot_cost + sa_cost),
          help=TT["total_cost"])
c2.metric("Conversations", f"{len(sessions)} (+{len(all_sa_sessions)} sub)",
          help="Main sessions + background sub-process sessions")
c3.metric("Turns", f"{len(all_turns):,}",
          help="Total number of Claude responses across all sessions")
c4.metric("History Reuse Rate", f"{hit_rate:.1f}%",
          help=TT["hit_rate"])
c5.metric("Saved by Caching", fmt_cost(saving),
          help=TT["saving"])

st.divider()

# ── Action cards ──────────────────────────────────────────────────────────────
st.markdown("### What's driving your cost — and what to do about it")

for icon, headline, detail, sev, info in actions:
    col_card, col_info = st.columns([20, 1])
    with col_card:
        if sev == "red":
            st.error(f"**{icon} {headline}**\n\n{detail}")
        elif sev == "orange":
            st.warning(f"**{icon} {headline}**\n\n{detail}")
        elif sev == "yellow":
            st.info(f"**{icon} {headline}**\n\n{detail}")
        else:
            st.success(f"**{icon} {headline}**\n\n{detail}")
    with col_info:
        with st.popover("ℹ️"):
            st.markdown(f"**What it means**\n\n{info['what']}")
            st.markdown("---")
            st.markdown(f"**How it's computed**\n\n{info['computed']}")
            st.markdown("---")
            st.markdown(f"**How to reduce it**\n\n{info['reduce']}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tool_counts         = data["tool_counts"]
session_lengths     = data["session_lengths"]
hourly_activity     = data["hourly_activity"]
word_freq           = data["word_freq"]
turns_with_tools    = data["turns_with_tools"]
turns_without_tools = data["turns_without_tools"]
avg_tools_per_turn  = data["avg_tools_per_turn"]
all_prompts         = data["all_prompts"]

tabs = st.tabs([
    "How you use Claude",
    "Where the money goes",
    "Caching",
    "Idle Gaps & Resets",
    "Maxed-out Sessions",
    "Break & Continue",
    "Large Content",
    "Big Responses",
    "Sub-processes",
    "By Project",
    "Day by Day",
])

# ── 0. How you use Claude ────────────────────────────────────────────────────
with tabs[0]:
    import math

    col_l, col_r = st.columns(2)

    with col_l:
        # Tool usage frequency
        st.markdown("#### What Claude does on each turn")
        if tool_counts:
            top_tools = tool_counts.most_common(15)
            tool_df = pd.DataFrame(top_tools, columns=["Tool", "Uses"])
            tool_df = tool_df.set_index("Tool")
            st.bar_chart(tool_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Turns with tool use",    turns_with_tools,
                      help="Turns where Claude called at least one tool (file read, bash, edit, etc.)")
            c2.metric("Turns — text only",      turns_without_tools,
                      help="Turns where Claude responded without using any tools")
            c3.metric("Avg tools per active turn", f"{avg_tools_per_turn:.1f}",
                      help="Average number of tool calls in turns where tools were used")
        else:
            st.info("No tool usage data found.")

    with col_r:
        # Top prompt keywords
        st.markdown("#### What you ask about most  (top words in your prompts)")
        if word_freq:
            top_words = word_freq.most_common(25)
            word_df = pd.DataFrame(top_words, columns=["Word", "Count"]).set_index("Word")
            st.bar_chart(word_df)
            st.caption(f"From {len(all_prompts):,} logged prompts. Stop words and short words removed.")
        else:
            st.info("No prompt history found (history.jsonl empty or missing).")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        # Session length distribution
        st.markdown("#### Session length distribution  (turns per session)")
        if session_lengths:
            buckets = {"1–5": 0, "6–20": 0, "21–50": 0, "51–100": 0,
                       "101–200": 0, "200+": 0}
            for n in session_lengths:
                if   n <= 5:   buckets["1–5"]     += 1
                elif n <= 20:  buckets["6–20"]    += 1
                elif n <= 50:  buckets["21–50"]   += 1
                elif n <= 100: buckets["51–100"]  += 1
                elif n <= 200: buckets["101–200"] += 1
                else:          buckets["200+"]    += 1
            bucket_df = pd.DataFrame({"Sessions": buckets})
            st.bar_chart(bucket_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Median session length", f"{sorted(session_lengths)[len(session_lengths)//2]} turns")
            c2.metric("Longest session",       f"{max(session_lengths)} turns")
            c3.metric("Avg session length",    f"{sum(session_lengths)/len(session_lengths):.0f} turns")

    with col_r:
        # Active hours
        st.markdown("#### When you use Claude  (turns by hour of day, local time)")
        if hourly_activity:
            hour_df = pd.DataFrame({
                "Turns": {f"{h:02d}:00": hourly_activity.get(h, 0) for h in range(24)}
            })
            st.bar_chart(hour_df)
            peak_hour = max(hourly_activity, key=hourly_activity.get)
            st.caption(f"Peak hour: {peak_hour:02d}:00  ({hourly_activity[peak_hour]:,} turns)")

# ── 1. Where the money goes ───────────────────────────────────────────────────
with tabs[1]:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Cost by what Claude is doing**")
        cost_items = [
            ("Storing conversation history",  c_cw,  "$3.75/M tokens",
             "Every turn, the growing history is stored in a fast cache."),
            ("Reading back the history",       c_cr,  "$0.30/M tokens",
             "Every turn, Claude reads the full history before responding."),
            ("Claude's actual responses",      c_out, "$15.00/M tokens",
             "The text Claude writes back to you."),
            ("Your actual messages",           c_inp, "$3.00/M tokens",
             "The text you type to Claude."),
        ]
        for label, cost_v, rate, tip in cost_items:
            pct = cost_v / (tot_cost + sa_cost)
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.progress(pct, text=label)
            c2.markdown(f"**{fmt_cost(cost_v)}**")
            c3.markdown(f"*{pct*100:.0f}%*")

        st.divider()
        type_df = pd.DataFrame([
            {
                "What":           lbl,
                "Tokens":         fmt_tok(tok),
                "Cost":           fmt_cost(cv),
                "Share of bill":  f"{cv/(tot_cost+sa_cost)*100:.1f}%",
                "Price":          rate,
            }
            for lbl, tok, cv, rate in [
                ("Storing history",   tot_cw5, c_cw,  "$3.75 / M tokens"),
                ("Reading history",   tot_cr,  c_cr,  "$0.30 / M tokens"),
                ("Claude responses",  tot_out, c_out, "$15.00 / M tokens"),
                ("Your messages",     tot_inp, c_inp, "$3.00 / M tokens"),
            ]
        ])
        st.dataframe(type_df, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Main conversations",   fmt_cost(tot_cost),
                  help="Cost from your main sessions only")
        c2.metric("Including sub-processes", fmt_cost(tot_cost + sa_cost),
                  help="Add in background sub-process sessions")

    with col_r:
        st.markdown("**Cost by AI model used**")
        mdf = []
        for m, d in sorted(model_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
            p = get_pricing(m)
            mdf.append({
                "Model":          m,
                "Turns":          d["turns"],
                "Cost":           fmt_cost(d["cost"]),
                "Share of bill":  f"{d['cost']/(tot_cost+sa_cost)*100:.1f}%",
                "Input $/M":      p[0],
                "Output $/M":     p[1],
            })
        st.dataframe(pd.DataFrame(mdf), hide_index=True, use_container_width=True)

# ── 2. Caching ────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "Claude caches (stores) your conversation history after each turn. "
        "Subsequent turns read from that cache at a fraction of the original cost. "
        "Without this, every turn would re-pay full price for the entire history."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Without caching",      fmt_cost(cost_no_cache),
              help="If every token was charged at full input price every turn")
    c2.metric("With caching (actual)", fmt_cost(tot_cost))
    c3.metric("Saved",                fmt_cost(saving),
              f"{saving/cost_no_cache*100:.0f}% cheaper",
              help=TT["saving"])
    c4.metric("History reuse rate",   f"{hit_rate:.1f}%",
              f"{fmt_tok(tot_cr)} tokens read from cache",
              help=TT["hit_rate"])

# ── 3. Idle gaps & resets ─────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        "The conversation cache expires after **5 minutes** of inactivity. "
        "When you return, the next turn has to re-store the full history — "
        "which costs **12.5× more** than a normal cached read."
    )
    if not all_resets:
        st.success("No idle expiries detected.")
    else:
        n = len(all_resets)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Times cache expired",       n,
                  help=TT["resets"])
        c2.metric("Cost on those turns",       fmt_cost(reset_actual),
                  help="What you actually paid on the turns where cache had expired")
        c3.metric("Cost if cache was warm",    fmt_cost(reset_warm),
                  help="What those same turns would have cost with an active cache")
        c4.metric("Extra cost from expiry",    fmt_cost(reset_waste),
                  f"{reset_actual/reset_warm:.1f}× more expensive per turn")
        c5.metric("Avoidable with longer cache",
                  f"{fix_1h}/{n}  ({fix_1h/n*100:.0f}%)",
                  "gaps under 60 min — Max plan keeps cache for 1h")


        if hour_counts:
            st.markdown("**When do expiries happen?**  (by hour of day)")
            hour_df = pd.DataFrame(
                {"Expiries": {f"{h:02d}:00": c for h, c in sorted(hour_counts.items())}}
            )
            st.bar_chart(hour_df)

# ── 4. Maxed-out sessions ─────────────────────────────────────────────────────
with tabs[4]:
    st.markdown(
        "Claude can only hold ~166K tokens of history at once. "
        "When a session fills that limit, every subsequent turn pays the **maximum possible** "
        "read cost — there's nowhere left to be efficient."
    )
    if not full_sessions:
        st.success("No sessions hit the memory limit.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions that maxed out",   len(full_sessions),
                  help=TT["full_ctx"])
        c2.metric("Turns at the limit",        total_full_turns,
                  help="Number of Claude responses where memory was completely full")
        c3.metric("Extra cost from being full", fmt_cost(total_full_cost),
                  help="Cache-read cost on those max-memory turns")
        c4.metric("Avg cost per maxed-out turn",
                  fmt_cost(total_full_cost / total_full_turns if total_full_turns else 0))

        rows = []
        for s in full_sessions:
            rows.append({
                "Date":            s["start"].strftime("%Y-%m-%d %H:%M"),
                "Project":         trunc(s["project"], 45),
                "Turns at limit":  len(s["full_ctx_turns"]),
                "Cost at limit":   fmt_cost(s["full_ctx_cost"]),
                "Peak memory":     fmt_tok(s["max_cr"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Tip: run /compact when memory is ~50% full to summarise and reset — halves the per-turn cost for the rest of the session.")

# ── 5. Break & Continue ───────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(
        "When you take a break and come back to the **same session**, "
        "everything Claude knew before the break stays in memory — and gets read back "
        "on every new turn, even if that earlier work is finished. "
        "Starting a new session after a break costs almost nothing to begin with (~13K tokens)."
    )
    if not all_splits:
        st.success("No significant break-and-continue patterns detected.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Times you continued after a break", len(all_splits),
                  help=TT["carry_cost"])
        c2.metric("Cost of carrying old history",      fmt_cost(split_carry),
                  f"{split_carry/c_cr*100:.0f}% of your total history-read cost",
                  help="Each turn after a break paid to read back the pre-break history")
        c3.metric("Average per break",                 fmt_cost(split_carry / len(all_splits)))

        rows = []
        for ev in all_splits:
            rows.append({
                "When":               ev["ts"].strftime("%Y-%m-%d %H:%M"),
                "Break length":       fmt_mins(ev["gap_mins"]),
                "History size":       fmt_tok(ev["cr_before"]),
                "Turns continued":    ev["turns_after"],
                "Cost of stale history": fmt_cost(ev["carry_cost"]),
                "Project":            trunc(ev["project"], 35),
                "Last message before break": trunc(ev["before_text"], 50) if ev["before_text"] else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── 6. Large content ──────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown(
        "When a large block of content is added early in a session "
        "(a pasted error log, a file Claude read, a long tool output), "
        "it stays in memory for every subsequent turn — even after the task is done."
    )
    if not all_artifacts:
        st.success("No large early content detected.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions with large early content", len(all_artifacts),
                  help=TT["artifact"])
        c2.metric("Cost of carrying it",               fmt_cost(art_carry))
        c3.metric("Average per session",               fmt_cost(art_carry / len(all_artifacts)))

        rows = []
        for ev in all_artifacts:
            rows.append({
                "When":              ev["ts"].strftime("%Y-%m-%d %H:%M"),
                "Turn added":        ev["turn_num"],
                "Extra content size": fmt_tok(ev["artifact_size"]),
                "Turns it rode for": ev["remaining"],
                "Carry cost":        fmt_cost(ev["carry_cost"]),
                "Project":           trunc(ev["project"], 35),
                "Prompt":            trunc(ev["prompt"], 50) if ev["prompt"] else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Extra content size = tokens above a normal message (~1K). "
            "Carry cost = extra tokens × turns it stayed × $0.30/M."
        )

# ── 7. Big responses ──────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown(
        "When Claude writes a very long response (a full file, a detailed plan), "
        "that response stays in memory for all future turns in the session."
    )
    if not all_high_out:
        st.success("No unusually large responses detected.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Turns with very large responses", len(all_high_out))
        c2.metric("Cost of those responses riding in memory", fmt_cost(hiout_carry))
        c3.metric("Average per response",            fmt_cost(hiout_carry / len(all_high_out)))

        rows = []
        for ev in all_high_out:
            rows.append({
                "When":              ev["ts"].strftime("%Y-%m-%d %H:%M"),
                "Response size":     fmt_tok(ev["out"]),
                "Turns it rode for": ev["remaining"],
                "Carry cost":        fmt_cost(ev["carry_cost"]),
                "Project":           trunc(ev["project"], 35),
                "Your message":      trunc(ev["prompt"], 50) if ev["prompt"] else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Long responses aren't inherently wasteful — but if you're done with that task, a fresh session keeps subsequent costs low.")

# ── 8. Sub-processes ─────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown(
        "Claude Code sometimes launches background sub-processes to handle tasks "
        "in parallel (e.g. running tests, searching files). Each has its own session cost."
    )
    if not all_sa_sessions:
        st.info("No sub-process sessions found.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sub-process sessions",    len(all_sa_sessions),
                  help=TT["subagent"])
        c2.metric("Total turns",             len(sa_turns))
        c3.metric("Total cost",              fmt_cost(sa_cost),
                  f"{sa_cost/(tot_cost+sa_cost)*100:.1f}% of combined bill")
        c4.metric("History storage cost",   fmt_cost(sa_c_cw))
        c5.metric("History read cost",      fmt_cost(sa_c_cr))

        rows = []
        for sid, sa_list in sorted(subagents.items(),
                                   key=lambda x: sum(s["cost"] for s in x[1]),
                                   reverse=True):
            prj  = sessions[sid]["project"] if sid in sessions else "unknown"
            cost = sum(s["cost"] for s in sa_list)
            rows.append({
                "Main session":    sid[:8],
                "Project":         trunc(prj, 45),
                "Sub-processes":   len(sa_list),
                "Cost":            fmt_cost(cost),
                "% of sub total":  f"{cost/sa_cost*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── 9. By project ─────────────────────────────────────────────────────────────
with tabs[9]:
    proj_df = pd.DataFrame([
        {
            "Project":         trunc(name, 50),
            "Cost":            fmt_cost(d["cost"]),
            "Share of total":  f"{d['cost']/(tot_cost+sa_cost)*100:.1f}%",
            "Cost per turn":   fmt_cost(d["cost"] / d["turns"]) if d["turns"] else "$0",
            "Sessions":        d["sessions"],
            "Turns":           d["turns"],
        }
        for name, d in sorted_proj
    ])

    col_l, col_r = st.columns([1, 1])
    with col_l:
        chart_data = pd.DataFrame({
            "Cost ($)": {trunc(n, 35): round(d["cost"], 4) for n, d in sorted_proj}
        })
        st.bar_chart(chart_data)
    with col_r:
        max_cost = sorted_proj[0][1]["cost"]
        for name, d in sorted_proj:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.progress(d["cost"] / max_cost, text=trunc(name, 38))
            c2.markdown(f"**{fmt_cost(d['cost'])}**")
            c3.markdown(f"*{d['cost']/(tot_cost+sa_cost)*100:.0f}%*")

# ── 10. Day by day ────────────────────────────────────────────────────────────
with tabs[10]:
    peak_day, peak_d = max(sorted_days, key=lambda x: x[1]["cost"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak day",       peak_day)
    c2.metric("Peak day cost",  fmt_cost(peak_d["cost"]))
    c3.metric("Peak day turns", f"{peak_d['turns']:,}")

    daily_df = pd.DataFrame([
        {"Date": day, "Cost ($)": round(d["cost"], 4), "Turns": d["turns"]}
        for day, d in sorted_days
    ]).set_index("Date")

    st.bar_chart(daily_df[["Cost ($)"]])
    st.dataframe(daily_df.reset_index(), hide_index=True, use_container_width=True)
