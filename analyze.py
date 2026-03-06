#!/usr/bin/env python3
"""
Claude Code — Cost & Usage Analyzer
Works for any Claude Code installation on macOS / Linux.

Data sources (all under ~/.claude/):
  projects/<project>/<session>.jsonl              — conversation + per-turn token usage
  projects/<project>/<session>/subagents/*.jsonl  — subagent sessions
  history.jsonl                                   — prompt text log

Usage:  python analyze.py
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────

CLAUDE_DIR   = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
HISTORY_FILE = CLAUDE_DIR / "history.jsonl"

# Per-model pricing per million tokens — source: platform.claude.com/docs/en/about-claude/pricing
# Each entry: (input, output, cache_write_5m, cache_write_1h, cache_read)
# Keyed by model ID prefix for flexible matching.
MODEL_PRICING = {
    "claude-opus-4-6":            (5.00,  25.00,  6.25, 10.00, 0.50),
    "claude-opus-4-5":            (5.00,  25.00,  6.25, 10.00, 0.50),
    "claude-opus-4-1":            (15.00, 75.00, 18.75, 30.00, 1.50),
    "claude-opus-4":              (15.00, 75.00, 18.75, 30.00, 1.50),
    "claude-opus-3":              (15.00, 75.00, 18.75, 30.00, 1.50),
    "claude-sonnet-4-6":          (3.00,  15.00,  3.75,  6.00, 0.30),
    "claude-sonnet-4-5":          (3.00,  15.00,  3.75,  6.00, 0.30),
    "claude-sonnet-4":            (3.00,  15.00,  3.75,  6.00, 0.30),
    "claude-sonnet-3-7":          (3.00,  15.00,  3.75,  6.00, 0.30),
    "claude-haiku-4-5":           (1.00,   5.00,  1.25,  2.00, 0.10),
    "claude-haiku-3-5":           (0.80,   4.00,  1.00,  1.60, 0.08),
    "claude-haiku-3":             (0.25,   1.25,  0.30,  0.50, 0.03),
}
# Fallback if model is unknown or synthetic
DEFAULT_PRICING = (3.00, 15.00, 3.75, 6.00, 0.30)  # Sonnet rates

def get_pricing(model: str) -> tuple:
    """Return (input, output, cw5m, cw1h, cr) per million tokens for a model."""
    if not model or model == "<synthetic>":
        return DEFAULT_PRICING
    m = model.lower()
    # Match longest prefix first so opus-4-6 beats opus-4
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if m.startswith(key):
            return MODEL_PRICING[key]
    return DEFAULT_PRICING

CONTEXT_LIMIT       = 166_000   # approximate Claude context window in tokens
NEAR_FULL_THRESHOLD = 0.90      # flag turns where cr > 90% of limit
CACHE_RESET_GAP     = 5         # minutes idle that expires the 5m cache
CACHE_RESET_DROP    = 0.50      # cache_read must drop to <50% of prev to count as reset
CACHE_RESET_MIN_CTX = 2_000     # ignore resets in tiny sessions
SPLIT_GAP_MINS      = 30        # idle gap suggesting session should have been two sessions
HIGH_OUTPUT_MULT    = 3.0       # output > N× session avg = notable turn
HIGH_OUTPUT_MIN     = 500       # ignore tiny outputs even if they're N× avg

# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_cost(c):
    return f"${c:.4f}" if c < 1 else f"${c:.2f}"

def fmt_tok(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def fmt_mins(m):
    if m < 60:   return f"{m:.0f}m"
    if m < 1440: return f"{m/60:.1f}h"
    return f"{m/1440:.1f}d"

def pct(part, total):
    return f"{part/total*100:.1f}%" if total else "—"

def bar(val, max_val, w=22):
    filled = round(val / max_val * w) if max_val else 0
    return "█" * filled + "░" * (w - filled)

def trunc(s, n):
    return s if len(s) <= n else "…" + s[-(n-1):]

W = 72

def section(title):
    print(); print("─" * W); print(f"  {title}"); print("─" * W)

# ── Path helpers ──────────────────────────────────────────────────────────────

def decode_project(dir_name: str) -> str:
    decoded = "/" + dir_name.lstrip("-").replace("-", "/")
    home    = str(Path.home())
    return ("~" + decoded[len(home):]) if decoded.startswith(home) else decoded

# ── Cost ──────────────────────────────────────────────────────────────────────

def calc_cost(inp, out, cr, cw5, cw1=0, model=""):
    p_inp, p_out, p_cw5, p_cw1, p_cr = get_pricing(model)
    return (inp * p_inp + out * p_out + cr * p_cr +
            cw5 * p_cw5 + cw1 * p_cw1) / 1_000_000

def calc_cost_no_cache(inp, out, cr, cw5, cw1=0, model=""):
    p_inp, p_out, _, _, _ = get_pricing(model)
    return ((inp + cr + cw5 + cw1) * p_inp + out * p_out) / 1_000_000

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_ts(s):
    try:    return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except: return None

def parse_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except json.JSONDecodeError: pass
    except OSError: pass
    return out

def extract_turns(records):
    """
    Extract per-turn usage from a session's records.
    Returns list of turn dicts with token counts + behavioural flags.
    """
    turns      = []
    prev_cr    = None
    prev_ts    = None

    for r in records:
        if r.get("type") != "assistant":
            continue
        usage = r.get("message", {}).get("usage", {})
        if not usage:
            continue
        ts = parse_ts(r.get("timestamp", ""))
        if ts is None:
            continue

        model = r.get("message", {}).get("model", "")
        cc    = usage.get("cache_creation", {})
        inp   = usage.get("input_tokens", 0)
        out   = usage.get("output_tokens", 0)
        cr    = usage.get("cache_read_input_tokens", 0)
        cw5   = cc.get("ephemeral_5m_input_tokens",
                usage.get("cache_creation_input_tokens", 0))
        cw1   = cc.get("ephemeral_1h_input_tokens", 0)

        gap_mins   = (ts - prev_ts).total_seconds() / 60 if prev_ts else None
        is_reset   = False
        is_compact = False

        if prev_cr is not None and prev_cr >= CACHE_RESET_MIN_CTX:
            dropped = cr < prev_cr * CACHE_RESET_DROP
            if dropped:
                if gap_mins is not None and gap_mins >= CACHE_RESET_GAP:
                    is_reset   = True
                elif gap_mins is not None:
                    is_compact = True

        # Capture tool names used in this turn
        tools_used = [
            blk.get("name", "unknown")
            for blk in r.get("message", {}).get("content", [])
            if isinstance(blk, dict) and blk.get("type") == "tool_use"
        ]

        turns.append({
            "ts":            ts,
            "model":         model,
            "inp":           inp,
            "out":           out,
            "cr":            cr,
            "cw5":           cw5,
            "cw1":           cw1,
            "cost":          calc_cost(inp, out, cr, cw5, cw1, model),
            "cost_no_cache": calc_cost_no_cache(inp, out, cr, cw5, cw1, model),
            "gap_mins":      gap_mins,
            "is_reset":      is_reset,
            "is_compact":    is_compact,
            "parent_uuid":   r.get("parentUuid"),
            "tools":         tools_used,
        })

        prev_cr = cr
        prev_ts = ts

    return turns

# ── Session-level insights ────────────────────────────────────────────────────

def analyse_session(turns, session_id, project):
    """Compute all session-level behavioural metrics."""
    if not turns:
        return None

    n         = len(turns)
    avg_out   = sum(t["out"] for t in turns) / n
    near_full = int(CONTEXT_LIMIT * NEAR_FULL_THRESHOLD)

    # ── Insight 3: Full-context turns ────────────────────────────────────
    # Turns where context was near the limit — every one paid maximum cache_read cost.
    full_ctx_turns = [t for t in turns if t["cr"] >= near_full]
    full_ctx_cost  = sum(t["cr"] * get_pricing(t["model"])[4] / 1_000_000 for t in full_ctx_turns)

    # ── Insight 5: Should-have-been-split ────────────────────────────────
    # Mid-session idle gaps >= SPLIT_GAP_MINS where significant work followed.
    #
    # Double-counting fix: each turn is attributed to its MOST RECENT preceding
    # gap only.  Gap k "owns" turns from gap_k up to (but not including) gap_{k+1}.
    # This prevents long sessions with many gaps from summing overlapping windows.
    gap_indices = [i for i, t in enumerate(turns)
                   if t["gap_mins"] and t["gap_mins"] >= SPLIT_GAP_MINS]

    split_events = []
    for k, i in enumerate(gap_indices):
        t             = turns[i]
        cr_before     = turns[i - 1]["cr"] if i > 0 else 0
        # Responsibility window: turns from this gap up to the next gap (or end)
        next_gap_i    = gap_indices[k + 1] if k + 1 < len(gap_indices) else n
        window        = turns[i:next_gap_i]
        # For display: total turns remaining after this gap
        turns_after   = turns[i:]
        # Carry cost uses only the non-overlapping window (no double-counting)
        carry_cost    = sum(cr_before * get_pricing(ta["model"])[4] / 1_000_000
                            for ta in window)
        out_after     = sum(ta["out"] for ta in turns_after)
        split_events.append({
            "ts":           t["ts"],
            "gap_mins":     t["gap_mins"],
            "cr_before":    cr_before,
            "turns_after":  len(turns_after),
            "out_after":    out_after,
            "carry_cost":   carry_cost,
            "is_reset":     t["is_reset"],
        })

    # ── Insight: Large early artifact ────────────────────────────────────
    # Turns in the first EARLY_TURN_WINDOW turns where a large amount was added
    # to context — a pasted error log, uploaded file, or Claude reading a big file.
    # That artifact then rides in context for every subsequent turn.
    #
    # Baseline: a plain message adds ~1K tokens to context (cw5 ≈ 1K).
    # An artifact turn adds much more. We flag cw5 > ARTIFACT_MIN_TOKENS early on.
    EARLY_TURN_WINDOW   = 15        # first N turns of session = "early"
    ARTIFACT_MIN_TOKENS = 10_000    # cw5 spike above this = likely artifact
    NORMAL_DELTA        = 1_000     # expected incremental cw5 without artifact

    # Find the single largest early spike (dedup: one event per session).
    # Rationale: if multiple early turns have large cw5, the FIRST one is when
    # the artifact entered context; subsequent turns are just carrying it.
    early_artifact_events = []
    best = None
    for i, t in enumerate(turns[:EARLY_TURN_WINDOW]):
        artifact_size = t["cw5"] - NORMAL_DELTA
        if artifact_size < ARTIFACT_MIN_TOKENS:
            continue
        turns_tail = turns[i + 1:]
        remaining  = len(turns_tail)
        carry_cost = sum(artifact_size * get_pricing(ta["model"])[4] / 1_000_000
                         for ta in turns_tail)
        candidate  = {
            "ts":            t["ts"],
            "turn_num":      i + 1,
            "cw5":           t["cw5"],
            "artifact_size": artifact_size,
            "remaining":     remaining,
            "carry_cost":    carry_cost,
            "parent_uuid":   t["parent_uuid"],
        }
        # Keep the first occurrence (earliest turn), not the largest
        if best is None:
            best = candidate
    if best:
        early_artifact_events.append(best)

    # ── Insight 6: High-output turns ─────────────────────────────────────
    # Turns where Claude generated significantly more than usual.
    # NOT a judgment — code generation should be long.
    # But that output then rides in context for all subsequent turns.
    high_out_events = []
    for i, t in enumerate(turns):
        if t["out"] >= avg_out * HIGH_OUTPUT_MULT and t["out"] >= HIGH_OUTPUT_MIN:
            t_out      = t["out"]
            turns_tail = turns[i + 1:]
            remaining  = len(turns_tail)
            carry_cost = sum(t_out * get_pricing(ta["model"])[4] / 1_000_000
                             for ta in turns_tail)
            high_out_events.append({
                "ts":          t["ts"],
                "turn_num":    i + 1,
                "out":         t["out"],
                "remaining":   remaining,
                "carry_cost":  carry_cost,
                "parent_uuid": t["parent_uuid"],
            })

    return {
        "id":              session_id,
        "project":         project,
        "turns":           turns,
        "cost":            sum(t["cost"] for t in turns),
        "cost_no_cache":   sum(t["cost_no_cache"] for t in turns),
        "start":           min(t["ts"] for t in turns),
        "end":             max(t["ts"] for t in turns),
        "max_cr":          max(t["cr"] for t in turns),
        # reset / compact
        "resets":          [t for t in turns if t["is_reset"]],
        "compactions":     [t for t in turns if t["is_compact"]],
        # insight 3
        "full_ctx_turns":  full_ctx_turns,
        "full_ctx_cost":   full_ctx_cost,
        # insight 5
        "split_events":          split_events,
        # early artifact
        "early_artifact_events": early_artifact_events,
        # insight 6
        "high_out_events":       high_out_events,
    }

# ── Data loading ──────────────────────────────────────────────────────────────

def load_sessions():
    sessions    = {}
    subagents   = {}   # session_id -> list of subagent sessions

    if not PROJECTS_DIR.exists():
        print(f"ERROR: {PROJECTS_DIR} not found.", file=sys.stderr)
        return sessions, subagents

    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        project = decode_project(project_dir.name)

        for session_file in project_dir.glob("*.jsonl"):
            sid    = session_file.stem
            turns  = extract_turns(parse_jsonl(session_file))
            result = analyse_session(turns, sid, project)
            if result:
                sessions[sid] = result

            # ── Insight 8: subagents ──────────────────────────────────
            sa_dir = project_dir / sid / "subagents"
            if sa_dir.is_dir():
                sa_sessions = []
                for sa_file in sa_dir.glob("*.jsonl"):
                    sa_turns  = extract_turns(parse_jsonl(sa_file))
                    sa_result = analyse_session(sa_turns, sa_file.stem, project)
                    if sa_result:
                        sa_result["parent_session"] = sid
                        sa_sessions.append(sa_result)
                if sa_sessions:
                    subagents[sid] = sa_sessions

    return sessions, subagents


def load_history():
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try: entries.append(json.loads(line))
            except json.JSONDecodeError: pass
    return entries


def build_prompt_map(history):
    by_session = defaultdict(list)
    for h in history:
        sid  = h.get("sessionId")
        ms   = h.get("timestamp", 0)
        text = h.get("display", "").strip()
        if sid and text and ms:
            by_session[sid].append({
                "ts":   datetime.fromtimestamp(ms / 1000).astimezone(),
                "text": text,
            })
    for v in by_session.values():
        v.sort(key=lambda x: x["ts"])
    return dict(by_session)


def nearest_prompt(turn_ts, prompts):
    for p in reversed(prompts):
        if p["ts"] <= turn_ts:
            return p["text"]
    return ""

# ── Report ────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  CLAUDE CODE — COST & USAGE ANALYSIS")
    print("=" * W)

    sessions, subagents = load_sessions()
    history             = load_history()
    prompt_map          = build_prompt_map(history)

    if not sessions:
        print("\n  No session data found. Check that Claude Code has been used.")
        return

    all_turns  = [t for s in sessions.values() for t in s["turns"]]
    all_resets = [t for t in all_turns if t["is_reset"]]
    all_compacts = [t for t in all_turns if t["is_compact"]]

    tot_inp  = sum(t["inp"]  for t in all_turns)
    tot_out  = sum(t["out"]  for t in all_turns)
    tot_cr   = sum(t["cr"]   for t in all_turns)
    tot_cw5  = sum(t["cw5"]  for t in all_turns)
    tot_cw1  = sum(t["cw1"]  for t in all_turns)
    tot_cost = sum(s["cost"] for s in sessions.values())

    c_inp = sum(t["inp"] * get_pricing(t["model"])[0] / 1_000_000 for t in all_turns)
    c_out = sum(t["out"] * get_pricing(t["model"])[1] / 1_000_000 for t in all_turns)
    c_cr  = sum(t["cr"]  * get_pricing(t["model"])[4] / 1_000_000 for t in all_turns)
    c_cw  = sum((t["cw5"] * get_pricing(t["model"])[2] +
                 t["cw1"] * get_pricing(t["model"])[3]) / 1_000_000 for t in all_turns)

    all_ts    = [t["ts"] for t in all_turns]
    date_from = min(all_ts).strftime("%Y-%m-%d")
    date_to   = max(all_ts).strftime("%Y-%m-%d")

    # Subagent totals
    all_sa_sessions = [sa for sas in subagents.values() for sa in sas]
    sa_turns        = [t for sa in all_sa_sessions for t in sa["turns"]]
    sa_cost         = sum(sa["cost"] for sa in all_sa_sessions)
    sa_inp          = sum(t["inp"] for t in sa_turns)
    sa_out          = sum(t["out"] for t in sa_turns)
    sa_cr           = sum(t["cr"]  for t in sa_turns)
    sa_cw5          = sum(t["cw5"] for t in sa_turns)

    # ── Overview ──────────────────────────────────────────────────────────
    section("OVERVIEW")
    print(f"  Period          {date_from}  →  {date_to}")
    print(f"  Projects        {len({s['project'] for s in sessions.values()})}")
    print(f"  Sessions        {len(sessions)}  (+{len(all_sa_sessions)} subagent sessions)")
    print(f"  Turns           {len(all_turns):,}  main  +  {len(sa_turns):,} subagent")
    print(f"  Prompts logged  {len(history):,}  (from history.jsonl)")
    print(f"  Cache resets    {len(all_resets):,}  (went idle >{CACHE_RESET_GAP} min mid-session)")
    print(f"  Compactions     {len(all_compacts):,}  (/compact or auto)")

    # ── 1. Cost Breakdown ─────────────────────────────────────────────────
    section("1. WHAT IS COSTING YOU  (main sessions only)")

    rows = [
        ("Output tokens",  tot_out, c_out, "$15.00/M"),
        ("Cache writes",   tot_cw5, c_cw,  "$3.75/M"),
        ("Input tokens",   tot_inp, c_inp, "$3.00/M"),
        ("Cache reads",    tot_cr,  c_cr,  "$0.30/M"),
    ]
    max_c = max(c_out, c_cw, c_inp, c_cr, 1e-9)

    print(f"  {'Type':<18} {'Tokens':>9}  {'Cost':>9}  {'% of bill':>9}  Chart")
    print(f"  {'─'*18}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*22}")
    for label, tokens, cost_v, rate in rows:
        print(f"  {label:<18} {fmt_tok(tokens):>9}  {fmt_cost(cost_v):>9}  "
              f"{pct(cost_v, tot_cost):>9}  {bar(cost_v, max_c)}  {rate}")

    print()
    print(f"  TOTAL (main sessions)  : {fmt_cost(tot_cost)}")
    print(f"  TOTAL (incl subagents) : {fmt_cost(tot_cost + sa_cost)}")
    print()

    # Cost and token breakdown per model (exact pricing per model used)
    model_stats = defaultdict(lambda: {"cost": 0.0, "inp": 0, "out": 0,
                                        "cr": 0, "cw5": 0, "turns": 0})
    for t in all_turns:
        m = t["model"] or "<unknown>"
        model_stats[m]["cost"]  += t["cost"]
        model_stats[m]["inp"]   += t["inp"]
        model_stats[m]["out"]   += t["out"]
        model_stats[m]["cr"]    += t["cr"]
        model_stats[m]["cw5"]   += t["cw5"]
        model_stats[m]["turns"] += 1

    if len(model_stats) > 1 or (len(model_stats) == 1 and "<synthetic>" not in model_stats):
        print(f"  Cost by model (exact pricing per model):")
        print(f"  {'Model':<35} {'Turns':>6}  {'Cost':>9}  Pricing used")
        print(f"  {'─'*35}  {'─'*6}  {'─'*9}  {'─'*20}")
        for m, d in sorted(model_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
            p = get_pricing(m)
            rate = f"in=${p[0]}/M out=${p[1]}/M"
            print(f"  {trunc(m,35):<35} {d['turns']:>6}  {fmt_cost(d['cost']):>9}  {rate}")

    # ── 2. Caching Savings ────────────────────────────────────────────────
    section("2. HOW MUCH CACHING IS SAVING YOU")

    cost_no_cache = sum(s["cost_no_cache"] for s in sessions.values())
    saving        = cost_no_cache - tot_cost
    ctx_tokens    = tot_inp + tot_cr + tot_cw5 + tot_cw1
    hit_rate      = tot_cr / ctx_tokens * 100 if ctx_tokens else 0

    print(f"  Without caching (all context at full input price): {fmt_cost(cost_no_cache)}")
    print(f"  With caching (actual cost):                        {fmt_cost(tot_cost)}")
    print(f"  Saved by caching:                                  {fmt_cost(saving)}  ({pct(saving, cost_no_cache)} reduction)")
    print()
    print(f"  Cache hit rate : {hit_rate:.1f}%  ({fmt_tok(tot_cr)} read  /  {fmt_tok(ctx_tokens)} total context)")

    # ── 3. Cache Reset Events ─────────────────────────────────────────────
    section("3. CACHE RESET EVENTS  (idle >{} min — full context re-written)".format(CACHE_RESET_GAP))

    if not all_resets:
        print("  None detected.")
    else:
        reset_actual = sum((t["cr"]  * get_pricing(t["model"])[4] +
                            t["cw5"] * get_pricing(t["model"])[2]) / 1_000_000
                           for t in all_resets)
        reset_warm   = sum((t["cw5"] * get_pricing(t["model"])[4] +
                            1_000    * get_pricing(t["model"])[2]) / 1_000_000
                           for t in all_resets)
        reset_waste  = reset_actual - reset_warm

        cw5s  = sorted(t["cw5"] for t in all_resets)
        n     = len(cw5s)
        gaps  = [t["gap_mins"] for t in all_resets if t["gap_mins"]]
        fix_1h = sum(1 for g in gaps if g < 60)

        print(f"  Total resets  : {n}")
        print(f"  Context at reset:  median={fmt_tok(cw5s[n//2])}  "
              f"p75={fmt_tok(cw5s[3*n//4])}  max={fmt_tok(cw5s[-1])}")
        print()
        print(f"  Cache cost on reset turns:")
        print(f"    Actual (full re-write) : {fmt_cost(reset_actual)}")
        print(f"    If cache was warm      : {fmt_cost(reset_warm)}")
        print(f"    Waste                  : {fmt_cost(reset_waste)}  "
              f"({n} resets × {fmt_cost(reset_waste/n)} avg)  —  {reset_actual/reset_warm:.1f}x more expensive per turn")
        print()
        print(f"  {fix_1h}/{n} resets had gap < 60 min  →  "
              f"Max plan (1h cache) prevents {pct(fix_1h, n)}")

        # Time-of-day pattern
        hour_counts = defaultdict(int)
        for t in all_resets:
            hour_counts[t["ts"].hour] += 1
        peak_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]
        print(f"  Peak reset hours: {', '.join(f'{h:02d}:00 ({hour_counts[h]}x)' for h in peak_hours)}")

    # ── 4. Full-Context Sessions ──────────────────────────────────────────
    section("4. SESSIONS THAT HIT NEAR-FULL CONTEXT  (>{:.0f}% of {}K limit)".format(
        NEAR_FULL_THRESHOLD * 100, CONTEXT_LIMIT // 1000))

    full_sessions = [s for s in sessions.values() if s["full_ctx_turns"]]
    full_sessions.sort(key=lambda s: len(s["full_ctx_turns"]), reverse=True)

    if not full_sessions:
        print("  No sessions hit the near-full threshold.")
    else:
        total_full_turns = sum(len(s["full_ctx_turns"]) for s in full_sessions)
        total_full_cost  = sum(s["full_ctx_cost"] for s in full_sessions)

        print(f"  {len(full_sessions)} sessions ran at near-full context.")
        print(f"  {total_full_turns} turns executed with >{int(NEAR_FULL_THRESHOLD*100)}% context filled.")
        print(f"  Cache-read cost for those turns: {fmt_cost(total_full_cost)}")
        print(f"  (Each turn at 150K context pays {fmt_cost(150_000 * DEFAULT_PRICING[4] / 1_000_000)} in cache reads alone)")
        print()

        print(f"  {'Date':<17} {'Project':<30} {'Turns@full':>10}  {'CR cost':>9}  {'Last prompts'}")
        print(f"  {'─'*17}  {'─'*30}  {'─'*10}  {'─'*9}  {'─'*20}")
        for s in full_sessions[:8]:
            dt  = s["start"].strftime("%Y-%m-%d %H:%M")
            prj = trunc(s["project"], 30)
            prompts = prompt_map.get(s["id"], [])
            # Last 2 prompts while at full context
            full_ts   = [t["ts"] for t in s["full_ctx_turns"]]
            last_prmpt = ""
            if full_ts and prompts:
                last_prmpt = trunc(nearest_prompt(max(full_ts), prompts), 30)
            print(f"  {dt:<17}  {prj:<30} {len(s['full_ctx_turns']):>10}  "
                  f"{fmt_cost(s['full_ctx_cost']):>9}  \"{last_prmpt}\"")

        print()
        print(f"  → Running /compact at 50% context instead of waiting for the limit")
        print(f"    would roughly halve the cache-read cost on those turns.")

    # ── 5. Sessions That Should Have Been Split ───────────────────────────
    section("5. SESSIONS WITH LARGE MID-SESSION GAPS  (old context carried after a break)")

    all_splits = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["split_events"]:
            # Only flag if there was meaningful work after the gap
            if ev["turns_after"] < 5:
                continue
            before_text = nearest_prompt(ev["ts"], prompts) if prompts else ""
            all_splits.append({
                **ev,
                "project":      s["project"],
                "session_id":   s["id"],
                "before_text":  before_text,
            })

    all_splits.sort(key=lambda x: x["carry_cost"], reverse=True)

    if not all_splits:
        print("  No significant mid-session gaps detected.")
    else:
        total_carry = sum(e["carry_cost"] for e in all_splits)
        print(f"  {len(all_splits)} mid-session gaps found where work continued after {SPLIT_GAP_MINS}+ min break.")
        print(f"  Total carry cost of old context into post-break turns: {fmt_cost(total_carry)}")
        print()
        print(f"  How to read this: after each gap, the context built before the break")
        print(f"  rides in cache_read for every subsequent turn — even if that work is done.")
        print()

        print(f"  {'Timestamp':<20} {'Gap':>7}  {'Ctx before':>10}  "
              f"{'Turns after':>11}  {'Carry cost':>11}  Project")
        print(f"  {'─'*20}  {'─'*7}  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*20}")
        for ev in all_splits[:10]:
            ts  = ev["ts"].strftime("%Y-%m-%d %H:%M")
            prj = trunc(ev["project"], 20)
            print(f"  {ts:<20} {fmt_mins(ev['gap_mins']):>7}  "
                  f"{fmt_tok(ev['cr_before']):>10}  {ev['turns_after']:>11}  "
                  f"{fmt_cost(ev['carry_cost']):>11}  {prj}")
            if ev["before_text"]:
                print(f"  {'':20}  {'':7}  last prompt before gap: \"{trunc(ev['before_text'], 45)}\"")

        print()
        print(f"  → After a break, consider: is this a new task? If yes, /clear or")
        print(f"    open a new session. The {fmt_cost(total_carry)} carry cost above was avoidable.")

    # ── 6. Early Artifact Carry Cost ─────────────────────────────────────
    section("6. LARGE EARLY ARTIFACTS  (added in first turns, carried for entire session)")

    all_artifacts = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["early_artifact_events"]:
            all_artifacts.append({
                **ev,
                "project": s["project"],
                "prompt":  nearest_prompt(ev["ts"], prompts),
            })

    all_artifacts.sort(key=lambda x: x["carry_cost"], reverse=True)

    if not all_artifacts:
        print("  No large early artifacts detected.")
    else:
        total_artifact_carry = sum(e["carry_cost"] for e in all_artifacts)
        print(f"  {len(all_artifacts)} sessions had a large content spike in the first 15 turns.")
        print(f"  Total carry cost of those artifacts riding for the rest of the session:")
        print(f"  {fmt_cost(total_artifact_carry)}")
        print()
        print(f"  This includes: pasted error logs, uploaded files, Claude reading large files,")
        print(f"  or any content added early that stayed in context long after it was needed.")
        print()

        print(f"  {'Timestamp':<20} {'Turn':>5}  {'Artifact':>9}  {'Rode for':>9}  "
              f"{'Carry cost':>11}  Prompt that triggered it")
        print(f"  {'─'*20}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*25}")
        for ev in all_artifacts[:10]:
            ts   = ev["ts"].strftime("%Y-%m-%d %H:%M")
            text = trunc(ev["prompt"], 35) if ev["prompt"] else "(unavailable)"
            print(f"  {ts:<20} {ev['turn_num']:>5}  {fmt_tok(ev['artifact_size']):>9}  "
                  f"{ev['remaining']:>8}t  {fmt_cost(ev['carry_cost']):>11}  \"{text}\"")

        print()
        print(f"  How to read this: 'artifact size' is the extra tokens added above a normal")
        print(f"  message. 'Rode for' is how many turns it was in context after that point.")
        print(f"  Carry cost = artifact_size × turns_after × $0.30/M.")
        print()
        print(f"  → We can't tell if the artifact was necessary — but if it was a one-off")
        print(f"    (e.g. paste an error, fix it, move on), starting a fresh session after")
        print(f"    the fix would have saved the carry cost shown above.")

    # ── 7. High-Output Turns & Their Carry Cost ───────────────────────────
    section("7. HIGH-OUTPUT TURNS  (large responses that rode in context)")

    all_high_out = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["high_out_events"]:
            all_high_out.append({
                **ev,
                "project": s["project"],
                "prompt":  nearest_prompt(ev["ts"], prompts),
            })

    all_high_out.sort(key=lambda x: x["carry_cost"], reverse=True)

    if not all_high_out:
        print("  No high-output turns detected.")
    else:
        total_carry = sum(e["carry_cost"] for e in all_high_out)
        print(f"  {len(all_high_out)} turns where Claude generated significantly more than usual.")
        print(f"  Total carry cost (output tokens × remaining turns × $0.30/M): {fmt_cost(total_carry)}")
        print()
        print(f"  Note: long output is often necessary (code generation, explanations).")
        print(f"  This shows the downstream cost of that output riding in context.")
        print()

        print(f"  {'Timestamp':<20} {'Output':>8}  {'Turns after':>11}  "
              f"{'Carry cost':>11}  Prompt")
        print(f"  {'─'*20}  {'─'*8}  {'─'*11}  {'─'*11}  {'─'*20}")
        for ev in all_high_out[:10]:
            ts   = ev["ts"].strftime("%Y-%m-%d %H:%M")
            text = trunc(ev["prompt"], 35) if ev["prompt"] else "(unavailable)"
            print(f"  {ts:<20} {fmt_tok(ev['out']):>8}  {ev['remaining']:>11}  "
                  f"{fmt_cost(ev['carry_cost']):>11}  \"{text}\"")

        print()
        print(f"  → High carry cost doesn't mean the output was wrong — it means")
        print(f"    starting a fresh session after large generation tasks keeps")
        print(f"    subsequent context (and costs) small.")

    # ── 7. Subagent Cost ──────────────────────────────────────────────────
    section("8. SUBAGENT COST")

    if not all_sa_sessions:
        print("  No subagent sessions found.")
    else:
        sa_c_out = sum(t["out"] * get_pricing(t["model"])[1] / 1_000_000 for t in sa_turns)
        sa_c_cr  = sum(t["cr"]  * get_pricing(t["model"])[4] / 1_000_000 for t in sa_turns)
        sa_c_cw  = sum(t["cw5"] * get_pricing(t["model"])[2] / 1_000_000 for t in sa_turns)

        print(f"  Subagent sessions  : {len(all_sa_sessions)}")
        print(f"  Subagent turns     : {len(sa_turns)}")
        print(f"  Subagent cost      : {fmt_cost(sa_cost)}  "
              f"({pct(sa_cost, tot_cost + sa_cost)} of combined total)")
        print()
        print(f"  Breakdown:  output={fmt_cost(sa_c_out)}  "
              f"cache_read={fmt_cost(sa_c_cr)}  cache_write={fmt_cost(sa_c_cw)}")
        print()

        # Which main sessions spawned the most subagents
        print(f"  Main sessions that spawned subagents:")
        print(f"  {'Session':<10} {'Project':<35} {'Subagents':>10}  {'SA cost':>9}")
        print(f"  {'─'*10}  {'─'*35}  {'─'*10}  {'─'*9}")
        for sid, sa_list in sorted(subagents.items(),
                                   key=lambda x: sum(s["cost"] for s in x[1]),
                                   reverse=True):
            prj  = sessions[sid]["project"] if sid in sessions else "unknown"
            cost = sum(s["cost"] for s in sa_list)
            print(f"  {sid[:8]:<10}  {trunc(prj, 35):<35} {len(sa_list):>10}  {fmt_cost(cost):>9}")

    # ── 8. Cost by Project ────────────────────────────────────────────────
    section("9. COST BY PROJECT")

    proj = defaultdict(lambda: {"cost": 0.0, "turns": 0, "sessions": 0})
    for s in sessions.values():
        p = s["project"]
        proj[p]["cost"]     += s["cost"]
        proj[p]["turns"]    += len(s["turns"])
        proj[p]["sessions"] += 1

    sorted_proj = sorted(proj.items(), key=lambda x: x[1]["cost"], reverse=True)
    max_pc = sorted_proj[0][1]["cost"] if sorted_proj else 1e-9

    print(f"  {'Project':<40} {'Cost':>9}  {'$/turn':>7}  {'Sessions':>8}  Chart")
    print(f"  {'─'*40}  {'─'*9}  {'─'*7}  {'─'*8}  {'─'*18}")
    for name, d in sorted_proj:
        cpt = d["cost"] / d["turns"] if d["turns"] else 0
        print(f"  {trunc(name,40):<40} {fmt_cost(d['cost']):>9}  "
              f"{fmt_cost(cpt):>7}  {d['sessions']:>8}  {bar(d['cost'], max_pc, 18)}")

    # ── 9. Daily Trend ────────────────────────────────────────────────────
    section("10. DAILY COST TREND")

    daily = defaultdict(lambda: {"cost": 0.0, "turns": 0})
    for t in all_turns:
        day = t["ts"].strftime("%Y-%m-%d")
        daily[day]["cost"]  += t["cost"]
        daily[day]["turns"] += 1

    sorted_days = sorted(daily.items())
    max_dc = max(d["cost"] for d in daily.values()) if daily else 1e-9

    print(f"  {'Date':<12} {'Cost':>9}  {'Turns':>6}   Chart")
    print(f"  {'─'*12}  {'─'*9}  {'─'*6}   {'─'*25}")
    for day, d in sorted_days:
        print(f"  {day:<12} {fmt_cost(d['cost']):>9}  {d['turns']:>6}   {bar(d['cost'], max_dc, 25)}")

    # ── 10. Recommendations ───────────────────────────────────────────────
    section("11. RECOMMENDATIONS")

    print(f"  Total estimated cost  {date_from} → {date_to}: {fmt_cost(tot_cost + sa_cost)} (incl. subagents)")
    print()

    recs = []

    split_cost = sum(e["carry_cost"] for e in all_splits)
    if split_cost > 0.10:
        recs.append((
            f"Split sessions at natural task boundaries  (saves ~{fmt_cost(split_cost)})",
            f"You have {len(all_splits)} mid-session gaps where old context kept riding.",
            "After fixing a bug or finishing a feature, open a fresh session.",
            "Fresh session cost: ~13K tokens (system+tools only). Much cheaper.",
        ))

    if full_sessions:
        recs.append((
            f"Use /compact before hitting the context limit  ({len(full_sessions)} sessions affected)",
            f"{total_full_turns} turns ran at near-full context, costing {fmt_cost(total_full_cost)} in cache reads.",
            "Compacting at 50% context halves the per-turn cache-read cost for the rest of the session.",
        ))

    if all_resets:
        n, waste = len(all_resets), reset_actual - reset_warm if all_resets else 0
        recs.append((
            f"Reduce idle gaps in sessions  ({n} cache resets, ~{fmt_cost(waste)} wasted)",
            "Each reset re-writes the full context at cache-write price instead of cache-read.",
            f"Max plan (1h cache) would eliminate ~{pct(fix_1h, n)} of your resets.",
        ))

    recs.append((
        "Start fresh sessions for unrelated tasks",
        "Each topic added to an existing session grows context permanently for that session.",
        "A fresh session starts with ~13K tokens. One large file read can add 30-50K instantly.",
    ))

    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec[0]}")
        for line in rec[1:]:
            print(f"     {line}")
        print()

    print("=" * W)
    print(f"  Pricing: exact per-model rates from platform.claude.com/docs/en/about-claude/pricing")
    print(f"  Costs are API-equivalent estimates (subscription users are not billed per-token).")
    print("=" * W)


def compute():
    """Return all analysis data as a dict — used by the Streamlit UI."""
    sessions, subagents = load_sessions()
    history             = load_history()
    prompt_map          = build_prompt_map(history)

    if not sessions:
        return None

    all_turns    = [t for s in sessions.values() for t in s["turns"]]
    all_resets   = [t for t in all_turns if t["is_reset"]]
    all_compacts = [t for t in all_turns if t["is_compact"]]

    tot_inp  = sum(t["inp"]  for t in all_turns)
    tot_out  = sum(t["out"]  for t in all_turns)
    tot_cr   = sum(t["cr"]   for t in all_turns)
    tot_cw5  = sum(t["cw5"]  for t in all_turns)
    tot_cw1  = sum(t["cw1"]  for t in all_turns)
    tot_cost = sum(s["cost"] for s in sessions.values())

    c_inp = sum(t["inp"] * get_pricing(t["model"])[0] / 1_000_000 for t in all_turns)
    c_out = sum(t["out"] * get_pricing(t["model"])[1] / 1_000_000 for t in all_turns)
    c_cr  = sum(t["cr"]  * get_pricing(t["model"])[4] / 1_000_000 for t in all_turns)
    c_cw  = sum((t["cw5"] * get_pricing(t["model"])[2] +
                 t["cw1"] * get_pricing(t["model"])[3]) / 1_000_000 for t in all_turns)

    all_ts    = [t["ts"] for t in all_turns]
    date_from = min(all_ts).strftime("%Y-%m-%d")
    date_to   = max(all_ts).strftime("%Y-%m-%d")

    all_sa_sessions = [sa for sas in subagents.values() for sa in sas]
    sa_turns        = [t for sa in all_sa_sessions for t in sa["turns"]]
    sa_cost         = sum(sa["cost"] for sa in all_sa_sessions)
    sa_c_out = sum(t["out"] * get_pricing(t["model"])[1] / 1_000_000 for t in sa_turns)
    sa_c_cr  = sum(t["cr"]  * get_pricing(t["model"])[4] / 1_000_000 for t in sa_turns)
    sa_c_cw  = sum(t["cw5"] * get_pricing(t["model"])[2] / 1_000_000 for t in sa_turns)

    model_stats = defaultdict(lambda: {"cost": 0.0, "inp": 0, "out": 0,
                                        "cr": 0, "cw5": 0, "turns": 0})
    for t in all_turns:
        m = t["model"] or "<unknown>"
        model_stats[m]["cost"]  += t["cost"]
        model_stats[m]["inp"]   += t["inp"]
        model_stats[m]["out"]   += t["out"]
        model_stats[m]["cr"]    += t["cr"]
        model_stats[m]["cw5"]   += t["cw5"]
        model_stats[m]["turns"] += 1

    cost_no_cache = sum(s["cost_no_cache"] for s in sessions.values())
    saving        = cost_no_cache - tot_cost
    ctx_tokens    = tot_inp + tot_cr + tot_cw5 + tot_cw1
    hit_rate      = tot_cr / ctx_tokens * 100 if ctx_tokens else 0

    reset_actual = reset_warm = 0.0
    fix_1h = 0
    hour_counts = defaultdict(int)
    if all_resets:
        reset_actual = sum((t["cr"]  * get_pricing(t["model"])[4] +
                            t["cw5"] * get_pricing(t["model"])[2]) / 1_000_000
                           for t in all_resets)
        reset_warm   = sum((t["cw5"] * get_pricing(t["model"])[4] +
                            1_000    * get_pricing(t["model"])[2]) / 1_000_000
                           for t in all_resets)
        gaps   = [t["gap_mins"] for t in all_resets if t["gap_mins"]]
        fix_1h = sum(1 for g in gaps if g < 60)
        for t in all_resets:
            hour_counts[t["ts"].hour] += 1

    full_sessions    = [s for s in sessions.values() if s["full_ctx_turns"]]
    full_sessions.sort(key=lambda s: len(s["full_ctx_turns"]), reverse=True)
    total_full_turns = sum(len(s["full_ctx_turns"]) for s in full_sessions)
    total_full_cost  = sum(s["full_ctx_cost"]        for s in full_sessions)

    all_splits = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["split_events"]:
            if ev["turns_after"] < 5:
                continue
            before_text = nearest_prompt(ev["ts"], prompts) if prompts else ""
            all_splits.append({**ev, "project": s["project"],
                                "session_id": s["id"], "before_text": before_text})
    all_splits.sort(key=lambda x: x["carry_cost"], reverse=True)

    all_artifacts = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["early_artifact_events"]:
            all_artifacts.append({**ev, "project": s["project"],
                                   "prompt": nearest_prompt(ev["ts"], prompts)})
    all_artifacts.sort(key=lambda x: x["carry_cost"], reverse=True)

    all_high_out = []
    for s in sessions.values():
        prompts = prompt_map.get(s["id"], [])
        for ev in s["high_out_events"]:
            all_high_out.append({**ev, "project": s["project"],
                                  "prompt": nearest_prompt(ev["ts"], prompts)})
    all_high_out.sort(key=lambda x: x["carry_cost"], reverse=True)

    proj = defaultdict(lambda: {"cost": 0.0, "turns": 0, "sessions": 0})
    for s in sessions.values():
        p = s["project"]
        proj[p]["cost"]     += s["cost"]
        proj[p]["turns"]    += len(s["turns"])
        proj[p]["sessions"] += 1
    sorted_proj = sorted(proj.items(), key=lambda x: x[1]["cost"], reverse=True)

    daily = defaultdict(lambda: {"cost": 0.0, "turns": 0})
    for t in all_turns:
        day = t["ts"].strftime("%Y-%m-%d")
        daily[day]["cost"]  += t["cost"]
        daily[day]["turns"] += 1
    sorted_days = sorted(daily.items())

    # ── Usage patterns ────────────────────────────────────────────────────────

    # Tool usage: count every tool_use block across all turns
    tool_counts = Counter()
    for t in all_turns:
        for name in t.get("tools", []):
            tool_counts[name] += 1

    # Turns per session — for session-length distribution
    session_lengths = [len(s["turns"]) for s in sessions.values()]

    # Active hours — turns by hour of day
    hourly_activity = Counter()
    for t in all_turns:
        hourly_activity[t["ts"].hour] += 1

    # Prompt word frequency — from history.jsonl display text
    STOP = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","have","has","had","do",
        "does","did","will","would","could","should","may","might","can","it",
        "its","this","that","these","those","i","you","he","she","we","they",
        "my","your","our","not","no","so","if","as","up","out","what","which",
        "who","how","when","where","why","all","just","now","also","into","than",
        "then","there","here","get","got","want","need","make","use","see","know",
        "like","good","new","one","two","more","some","any","much","very","too",
        "can","also","please","hi","hey","ok","yes","let","me","us","them","its",
    }
    word_freq = Counter()
    all_prompts = [p["text"] for plist in prompt_map.values() for p in plist]
    for text in all_prompts:
        for word in re.findall(r'\b[a-z]{3,}\b', text.lower()):
            if word not in STOP:
                word_freq[word] += 1

    # Turns with at least one tool call vs pure text responses
    turns_with_tools    = sum(1 for t in all_turns if t.get("tools"))
    turns_without_tools = len(all_turns) - turns_with_tools

    # Average tools per turn (where tools were used)
    tool_turn_counts = [len(t["tools"]) for t in all_turns if t.get("tools")]
    avg_tools_per_turn = sum(tool_turn_counts) / len(tool_turn_counts) if tool_turn_counts else 0

    return {
        "sessions":         sessions,
        "subagents":        subagents,
        "prompt_map":       prompt_map,
        "all_turns":        all_turns,
        "all_resets":       all_resets,
        "all_compacts":     all_compacts,
        "tot_inp":  tot_inp,  "tot_out": tot_out,
        "tot_cr":   tot_cr,   "tot_cw5": tot_cw5, "tot_cw1": tot_cw1,
        "tot_cost": tot_cost,
        "c_inp": c_inp, "c_out": c_out, "c_cr": c_cr, "c_cw": c_cw,
        "date_from": date_from, "date_to": date_to,
        "all_sa_sessions": all_sa_sessions,
        "sa_turns":  sa_turns,  "sa_cost": sa_cost,
        "sa_c_out":  sa_c_out,  "sa_c_cr": sa_c_cr, "sa_c_cw": sa_c_cw,
        "model_stats":    dict(model_stats),
        "cost_no_cache":  cost_no_cache,
        "saving":         saving,
        "hit_rate":       hit_rate,
        "reset_actual":   reset_actual,
        "reset_warm":     reset_warm,
        "fix_1h":         fix_1h,
        "hour_counts":    dict(hour_counts),
        "full_sessions":      full_sessions,
        "total_full_turns":   total_full_turns,
        "total_full_cost":    total_full_cost,
        "all_splits":     all_splits,
        "all_artifacts":  all_artifacts,
        "all_high_out":   all_high_out,
        "sorted_proj":    sorted_proj,
        "sorted_days":    sorted_days,
        # usage patterns
        "tool_counts":         tool_counts,
        "session_lengths":     session_lengths,
        "hourly_activity":     hourly_activity,
        "word_freq":           word_freq,
        "turns_with_tools":    turns_with_tools,
        "turns_without_tools": turns_without_tools,
        "avg_tools_per_turn":  avg_tools_per_turn,
        "all_prompts":         all_prompts,
    }


if __name__ == "__main__":
    main()
