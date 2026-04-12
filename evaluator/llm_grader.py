"""
evaluator/llm_grader.py
========================
Uses the Anthropic API (Claude) to evaluate a completed irrigation episode
and generate a structured improvement plan.

The grader takes:
  - Episode history (step-by-step observations + actions)
  - Programmatic scores (from ProgrammaticGrader)

And returns:
  - Narrative analysis (what went well, what didn't)
  - 3 specific mistakes with corrections
  - Concrete improvement plan with actionable strategies
  - Estimated score if the plan is followed
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """You are an expert RL environment evaluator and precision agriculture AI coach
specializing in Punjab rice farming and water conservation.

You evaluate human irrigation decisions made in a simulation environment and provide
specific, actionable feedback to help them improve.

The environment is PrecisionIrrigationEnvV3:
- 90-day rice growing season
- 5 actions: 0=No Water, 1=Low(1kL), 2=Medium(3kL), 3=High(5kL), 4=Drain
- Optimal soil moisture: 0.40 – 0.72
- Critical stage: Reproductive (stage 2) — needs consistent moisture
- Efficiency goal: maximize crop yield while minimizing water use

Be specific with your feedback. Reference exact days and actions from the history.
Be constructive, practical, and grounded in irrigation science."""

EVALUATION_PROMPT_TEMPLATE = """Analyze this completed irrigation episode and provide structured feedback.

=== EPISODE DATA ===
{episode_summary}

=== PROGRAMMATIC SCORES ===
{prog_scores}

=== STEP-BY-STEP HISTORY (sampled) ===
{history_sample}

=== ACTION DISTRIBUTION ===
{action_dist}

=== KEY EVENTS ===
{key_events}

=== MISSED WINDOWS ===
{missed_windows}

Please provide your evaluation in the following JSON format:
{{
  "narrative": "2-3 paragraph overall assessment of performance",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "mistakes": [
    {{"day": N, "mistake": "what was done wrong", "correction": "what should have been done", "impact": "why this matters"}},
    {{"day": N, "mistake": "...", "correction": "...", "impact": "..."}},
    {{"day": N, "mistake": "...", "correction": "...", "impact": "..."}}
  ],
  "improvement_plan": {{
    "immediate_rules": ["rule 1: specific threshold-based rule to follow", "rule 2", "rule 3"],
    "stage_strategy": {{
      "Seedling": "what to do in days 1-22",
      "Vegetative": "what to do in days 23-49",
      "Reproductive": "what to do in days 50-72",
      "Maturity": "what to do in days 73-90"
    }},
    "weather_adaptation": "how to read the 3-day forecast and adapt",
    "water_conservation_tip": "one specific water-saving technique"
  }},
  "predicted_improvement": {{
    "current_score": {current_score},
    "achievable_score": <number 0-100>,
    "yield_improvement": "estimated yield improvement if plan is followed",
    "water_savings": "estimated water savings"
  }}
}}

Return ONLY valid JSON. No markdown, no preamble."""


class LLMGrader:
    """
    Calls Claude to evaluate an episode and generate an improvement plan.
    Falls back to a rule-based analysis if API key is not set.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self._available = bool(self.api_key)

    @property
    def is_available(self) -> bool:
        return self._available

    def evaluate(
        self,
        history: list,
        prog_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns a structured evaluation dict.
        Falls back gracefully if API key not set.
        """
        if not self._available:
            return self._fallback_evaluation(prog_result)

        try:
            return self._call_api(history, prog_result)
        except Exception as e:
            return {**self._fallback_evaluation(prog_result), "error": str(e)}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _call_api(self, history: list, prog_result: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request

        # Sample up to 20 steps evenly
        n = len(history)
        if n > 20:
            indices = [int(i * n / 20) for i in range(20)]
            sample = [history[i] for i in indices]
        else:
            sample = history

        history_text = "\n".join(
            f"Day {s['day']:02d} | Stage:{s['crop_stage']} | "
            f"Moisture:{s['moisture']:.2f} | Rain:{s['rainfall_mm']:.1f}mm | "
            f"Action:{s['action']}({['NoWater','Low','Med','High','Drain'][s['action']]}) | "
            f"Reward:{s['reward']:.2f}"
            for s in sample
        )

        sm = prog_result["summary"]
        episode_summary = (
            f"Duration: {sm['total_days']} days | "
            f"Final growth: {sm['final_crop_growth']:.3f} | "
            f"Est. yield: {sm['estimated_yield_tha']} t/ha | "
            f"Total water: {sm['total_water_kL']} kL | "
            f"Total reward: {sm['total_reward']} | "
            f"Days in optimal band: {sm['pct_days_optimal']}%"
        )

        from evaluator.programmatic_grader import ProgrammaticGrader
        prog_text = ProgrammaticGrader.format_scores_text(prog_result)

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            episode_summary=episode_summary,
            prog_scores=prog_text,
            history_sample=history_text,
            action_dist=json.dumps(prog_result["action_distribution"], indent=2),
            key_events="\n".join(prog_result["key_events"]) or "None",
            missed_windows="\n".join(prog_result["missed_windows"]) or "None",
            current_score=prog_result["scores"]["overall"],
        )

        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        raw = data["content"][0]["text"]
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _fallback_evaluation(self, prog_result: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""
        sm = prog_result["summary"]
        sc = prog_result["scores"]
        overall = sc["overall"]

        strengths = []
        if sc["water_efficiency"] > 70:
            strengths.append("Good water conservation — used water judiciously")
        if sc["crop_health"] > 70:
            strengths.append("Maintained soil moisture in optimal band frequently")
        if sc["strategic_timing"] > 70:
            strengths.append("Well-timed irrigation during critical growth stages")
        if not strengths:
            strengths = ["Completed the full episode", "Gained experience with the environment"]

        mistakes = []
        if prog_result["missed_windows"]:
            w = prog_result["missed_windows"][0]
            mistakes.append({
                "day": "Multiple",
                "mistake": "Skipped irrigation during dry periods in growth-critical stages",
                "correction": "Apply Low or Medium irrigation when moisture < 0.35 in Vegetative/Reproductive",
                "impact": "Dry stress during these stages permanently reduces yield potential",
            })
        if sc["water_efficiency"] < 50:
            mistakes.append({
                "day": "Multiple",
                "mistake": "Over-irrigated when soil was already wet (moisture > 0.72)",
                "correction": "Check moisture reading before irrigating. No action needed if > 0.60",
                "impact": "Waterlogging wastes water and can damage roots",
            })
        if sc["strategic_timing"] < 60:
            mistakes.append({
                "day": "Days 50-72",
                "mistake": "Did not prioritize the Reproductive stage for consistent moisture",
                "correction": "Keep moisture between 0.50-0.70 during days 50-72 (Reproductive)",
                "impact": "Reproductive stage determines grain fill — moisture here = yield",
            })
        if not mistakes:
            mistakes = [{
                "day": "N/A",
                "mistake": "Minor suboptimal decisions in low-impact steps",
                "correction": "Continue current strategy but reduce irrigation in Maturity stage",
                "impact": "Small efficiency gains add up over 90 days",
            }]

        achievable = min(95, overall + 15)
        return {
            "narrative": (
                f"You completed a {sm['total_days']}-day irrigation episode achieving "
                f"{sm['estimated_yield_tha']} t/ha yield with {sm['total_water_kL']} kL water used. "
                f"Your overall score of {overall}/100 reflects {'solid' if overall > 60 else 'developing'} "
                f"irrigation management. "
                f"You kept the crop in the optimal moisture band {sm['pct_days_optimal']}% of days, "
                f"which {'is excellent' if sm['pct_days_optimal'] > 60 else 'has room for improvement'}. "
                f"The programmatic analysis identified {len(prog_result['missed_windows'])} missed irrigation "
                f"windows and {len(prog_result['key_events'])} key events. "
                f"Focus on the Reproductive stage (days 50-72) for the biggest yield improvement."
            ),
            "strengths": strengths,
            "mistakes": mistakes[:3],
            "improvement_plan": {
                "immediate_rules": [
                    "Rule 1: Irrigate (Low) if moisture < 0.35 AND stage is Vegetative or Reproductive",
                    "Rule 2: Apply Medium if moisture < 0.25 at any growth stage",
                    "Rule 3: No irrigation if moisture > 0.65 AND no high-value growth stage upcoming",
                    "Rule 4: Drain only if moisture > 0.80 to prevent waterlogging",
                ],
                "stage_strategy": {
                    "Seedling": "Keep moisture 0.45-0.65. Light irrigation every 3-4 days",
                    "Vegetative": "Target 0.50-0.70. Respond quickly to moisture < 0.38",
                    "Reproductive": "CRITICAL: Maintain 0.55-0.70 consistently. Irrigate every 1-2 days if rain < 3mm",
                    "Maturity": "Reduce irrigation. Allow moisture to drop to 0.35-0.45 for grain hardening",
                },
                "weather_adaptation": (
                    "If 3-day forecast shows total rain > 8mm, skip next irrigation cycle. "
                    "If forecast rain < 3mm/day for 3 days, pre-irrigate to 0.60 today."
                ),
                "water_conservation_tip": (
                    "Use the Low action (1kL) for maintenance irrigation instead of Medium — "
                    "this reduces total water use by ~60% while keeping moisture above 0.38."
                ),
            },
            "predicted_improvement": {
                "current_score": overall,
                "achievable_score": achievable,
                "yield_improvement": f"+{round((achievable - overall) * 0.04, 2)} t/ha",
                "water_savings": f"~{round((1 - sc['water_efficiency'] / 100) * 20, 1)}% reduction possible",
            },
            "_note": "LLM grader not available — set ANTHROPIC_API_KEY for Claude-powered analysis",
        }

    @staticmethod
    def format_evaluation_text(result: Dict[str, Any]) -> str:
        """Format the LLM evaluation as readable text."""
        lines = []
        if "_note" in result:
            lines.append(f"ℹ️  {result['_note']}\n")
        if "error" in result:
            lines.append(f"⚠️  API Error: {result['error']}\n")

        lines.append("═══ AI COACH ANALYSIS ═══\n")
        lines.append(result.get("narrative", ""))
        lines.append("")

        if result.get("strengths"):
            lines.append("✅ STRENGTHS:")
            for s in result["strengths"]:
                lines.append(f"  • {s}")
            lines.append("")

        if result.get("mistakes"):
            lines.append("❌ KEY MISTAKES:")
            for i, m in enumerate(result["mistakes"], 1):
                lines.append(f"  {i}. Day {m.get('day','?')}: {m.get('mistake','')}")
                lines.append(f"     → Fix: {m.get('correction','')}")
                lines.append(f"     → Why: {m.get('impact','')}")
            lines.append("")

        plan = result.get("improvement_plan", {})
        if plan:
            lines.append("📋 YOUR IMPROVEMENT PLAN:")
            lines.append("\n  Immediate Rules:")
            for r in plan.get("immediate_rules", []):
                lines.append(f"    • {r}")
            ss = plan.get("stage_strategy", {})
            if ss:
                lines.append("\n  Stage-by-Stage Strategy:")
                for stage, advice in ss.items():
                    lines.append(f"    [{stage}] {advice}")
            wa = plan.get("weather_adaptation", "")
            if wa:
                lines.append(f"\n  🌧️  Weather Adaptation: {wa}")
            wc = plan.get("water_conservation_tip", "")
            if wc:
                lines.append(f"\n  💧 Water Saving Tip: {wc}")
            lines.append("")

        pred = result.get("predicted_improvement", {})
        if pred:
            lines.append("📈 PREDICTED IMPROVEMENT:")
            lines.append(f"  Current Score    : {pred.get('current_score', '?')}/100")
            lines.append(f"  Achievable Score : {pred.get('achievable_score', '?')}/100")
            lines.append(f"  Yield Improvement: {pred.get('yield_improvement', '?')}")
            lines.append(f"  Water Savings    : {pred.get('water_savings', '?')}")

        return "\n".join(lines)
