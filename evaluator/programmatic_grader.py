"""
evaluator/programmatic_grader.py
=================================
Programmatic scoring of a completed irrigation episode.
Used both standalone and as context for the LLM grader.

Scoring dimensions (each 0-100):
  • water_efficiency   – did they use water wisely?
  • crop_health        – did they keep moisture in optimal band?
  • strategic_timing   – did they irrigate during critical growth stages?
  • drought_response   – how well did they adapt to low rainfall?
  • overall            – weighted composite

Also produces per-action commentary and summary statistics.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

# ── Constants mirrored from v3 env ────────────────────────────────────────────
OPT_LO  = 0.40
OPT_HI  = 0.72
DRY_THR = 0.22
WET_THR = 0.78

STAGES = {0: "Seedling", 1: "Vegetative", 2: "Reproductive", 3: "Maturity"}
ACTIONS = {0: "No Water", 1: "Low (1kL)", 2: "Medium (3kL)", 3: "High (5kL)", 4: "Drain"}

STAGE_WEIGHTS = {0: 0.8, 1: 1.0, 2: 1.5, 3: 0.7}  # reproductive most critical


class ProgrammaticGrader:
    """
    Grades a completed episode given the step-by-step history.

    history: list of dicts, one per step, each containing:
        {
          'day': int,
          'action': int,
          'moisture': float,
          'crop_growth': float,
          'crop_stage': int,
          'rainfall_mm': float,
          'reward': float,
          'tank_level': float,
          'water_used_L': float,
        }
    """

    def __init__(self, history: List[Dict[str, Any]]):
        self.history = history
        self._scores: Dict[str, float] = {}
        self._commentary: List[str] = []

    def __call__(self, history=None) -> float:
        if history is not None:
            self.history = history
        result = self.evaluate()
        return float(result["scores"]["overall"]) / 100.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self) -> Dict[str, Any]:
        """Return full evaluation dict."""
        if not self.history:
            return {"error": "Empty episode history"}

        we  = self._water_efficiency()
        ch  = self._crop_health()
        st  = self._strategic_timing()
        dr  = self._drought_response()
        ys  = self._yield_score()

        # Weighted formula — crop yield (YS) carries 30% so tasks where the
        # crop completely dies (hard) score very differently from tasks where
        # the crop partially or fully survives (medium/easy).
        # Weights: WE=0.10, CH=0.30, ST=0.20, DR=0.10, YS=0.30
        raw = 0.10 * we + 0.30 * ch + 0.20 * st + 0.10 * dr + 0.30 * ys

        # Strictly enforce 0 < overall < 1 (both ends excluded) by clamping
        # the raw score to (0.1, 99.9) before dividing by 100.
        overall = round(max(0.1, min(99.9, raw)), 1)

        final_growth  = self.history[-1]["crop_growth"]
        total_water   = sum(s["water_used_L"] for s in self.history)
        total_reward  = sum(s["reward"] for s in self.history)
        stress_days   = sum(1 for s in self.history if s["moisture"] < DRY_THR)
        overwet_days  = sum(1 for s in self.history if s["moisture"] > WET_THR)
        opt_days      = sum(1 for s in self.history
                            if OPT_LO <= s["moisture"] <= OPT_HI)
        pct_optimal   = round(100 * opt_days / max(1, len(self.history)), 1)

        yield_tha = round(6.5 * final_growth, 2)

        return {
            "scores": {
                "water_efficiency":  round(we, 1),
                "crop_health":       round(ch, 1),
                "strategic_timing":  round(st, 1),
                "drought_response":  round(dr, 1),
                "yield_score":       round(ys, 1),
                "overall":           overall,
            },
            "summary": {
                "final_crop_growth": round(final_growth, 3),
                "estimated_yield_tha": yield_tha,
                "total_water_kL":   round(total_water / 1000, 2),
                "total_reward":     round(total_reward, 2),
                "days_stressed":    stress_days,
                "days_overwet":     overwet_days,
                "pct_days_optimal": pct_optimal,
                "total_days":       len(self.history),
            },
            "action_distribution": self._action_distribution(),
            "key_events":         self._detect_key_events(),
            "missed_windows":     self._missed_irrigation_windows(),
        }

    # ── Scoring sub-methods ────────────────────────────────────────────────────

    def _water_efficiency(self) -> float:
        """Penalise over-irrigation, reward irrigation during dry spells."""
        if not self.history:
            return 50.0
        total_water   = sum(s["water_used_L"] for s in self.history)
        optimal_water = 3_000 * len(self.history) * 0.20  # ~20% of steps should be medium
        waste_events  = sum(
            1 for s in self.history
            if s["action"] in (2, 3) and s["moisture"] > OPT_HI
        )
        drain_when_dry = sum(
            1 for s in self.history
            if s["action"] == 4 and s["moisture"] < OPT_LO
        )
        score = 100.0
        # Penalise excess water use
        if total_water > optimal_water * 1.5:
            score -= 20
        # Penalise irrigating when already wet
        score -= min(30, waste_events * 5)
        # Penalise draining when dry
        score -= min(25, drain_when_dry * 8)
        return max(0.0, min(100.0, score))

    def _crop_health(self) -> float:
        """Reward time spent in optimal moisture band."""
        if not self.history:
            return 50.0
        opt_days   = sum(1 for s in self.history if OPT_LO <= s["moisture"] <= OPT_HI)
        stress_days = sum(1 for s in self.history if s["moisture"] < DRY_THR)
        overwet    = sum(1 for s in self.history if s["moisture"] > WET_THR)
        n = len(self.history)
        base   = 100 * opt_days / n
        penalty = (stress_days * 2 + overwet) / n * 30
        return max(0.0, min(100.0, base - penalty))

    def _strategic_timing(self) -> float:
        """Did the agent irrigate during reproductive stage?"""
        repro_steps = [s for s in self.history if s["crop_stage"] == 2]
        if not repro_steps:
            return 60.0  # can't penalise if stage never reached
        repro_opt = sum(
            1 for s in repro_steps
            if OPT_LO <= s["moisture"] <= OPT_HI
        )
        base = 100 * repro_opt / len(repro_steps)
        # Bonus for not wasting water in maturity stage (stage 3)
        mat_steps = [s for s in self.history if s["crop_stage"] == 3]
        if mat_steps:
            mat_waste = sum(1 for s in mat_steps if s["action"] in (2, 3))
            base -= min(15, (mat_waste / len(mat_steps)) * 30)
        return max(0.0, min(100.0, base))

    def _drought_response(self) -> float:
        """Did the agent actually irrigate on low-rain days?

        Only counts steps where the agent chose an irrigation action (1, 2, or 3)
        when rainfall was low. The old 'moisture >= OPT_LO' bypass is removed
        because it rewarded doing nothing when rain happened to keep moisture up,
        which is luck — not a decision.
        """
        low_rain_days = [s for s in self.history if s["rainfall_mm"] < 2.0]
        if not low_rain_days:
            return 80.0
        responded = sum(
            1 for s in low_rain_days
            if s["action"] in (1, 2, 3)
        )
        return min(100.0, 100 * responded / len(low_rain_days))

    def _yield_score(self) -> float:
        """Score based on final crop growth (0-100).

        This is the key differentiator between medium and hard tasks:
        - Easy:   crop survives and grows fully   → yield_score ≈ 90-100
        - Medium: crop partially survives          → yield_score ≈ 25-70
        - Hard:   crop collapses (growth → 0)     → yield_score ≈ 0-5

        Computed as a smooth sigmoid-like curve so partial growth is
        rewarded proportionally, not as a step function.
        """
        if not self.history:
            return 0.0
        final_growth = float(self.history[-1]["crop_growth"])
        # Scale: growth of 0.95+ → 100, growth of 0 → 0, growth of 0.5 → ~62
        # Use a simple power curve that rewards higher growth non-linearly.
        score = 100.0 * (final_growth ** 0.75)
        return max(0.0, min(100.0, score))

    # ── Analysis helpers ───────────────────────────────────────────────────────

    def _action_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {v: 0 for v in ACTIONS.values()}
        for s in self.history:
            dist[ACTIONS[s["action"]]] += 1
        return dist

    def _detect_key_events(self) -> List[str]:
        events = []
        for s in self.history:
            d = s["day"]
            if s["moisture"] < DRY_THR:
                events.append(f"Day {d}: ⚠️ Severe drought stress (moisture={s['moisture']:.2f})")
            if s["moisture"] > WET_THR and s["action"] in (2, 3):
                events.append(f"Day {d}: 💧 Over-irrigated already-wet field (moisture={s['moisture']:.2f})")
            if s["action"] == 4 and s["moisture"] < 0.35:
                events.append(f"Day {d}: ❌ Drained a dry field (moisture={s['moisture']:.2f})")
        return events[:10]  # cap at 10

    def _missed_irrigation_windows(self) -> List[str]:
        missed = []
        for s in self.history:
            # Was in vegetative/reproductive + dry + did nothing
            if s["crop_stage"] in (1, 2) and s["moisture"] < 0.35 and s["action"] == 0:
                missed.append(
                    f"Day {s['day']}: Missed irrigation in {STAGES[s['crop_stage']]} stage "
                    f"(moisture={s['moisture']:.2f}, no action taken)"
                )
        return missed[:8]

    # ── Static helper ──────────────────────────────────────────────────────────

    @staticmethod
    def format_scores_text(result: Dict[str, Any]) -> str:
        s = result["scores"]
        sm = result["summary"]
        lines = [
            "═══ PROGRAMMATIC EVALUATION ═══",
            f"  Water Efficiency   : {s['water_efficiency']:>5.1f}/100",
            f"  Crop Health        : {s['crop_health']:>5.1f}/100",
            f"  Strategic Timing   : {s['strategic_timing']:>5.1f}/100",
            f"  Drought Response   : {s['drought_response']:>5.1f}/100",
            f"  Yield Score        : {s['yield_score']:>5.1f}/100",
            f"  ─────────────────────────────",
            f"  OVERALL SCORE      : {s['overall']:>5.1f}/100",
            "",
            "═══ EPISODE SUMMARY ═══",
            f"  Est. Yield         : {sm['estimated_yield_tha']} t/ha",
            f"  Total Water Used   : {sm['total_water_kL']} kL",
            f"  Total Reward       : {sm['total_reward']}",
            f"  Days in Opt. Band  : {sm['pct_days_optimal']}%",
            f"  Stress Days        : {sm['days_stressed']}",
            f"  Overwet Days       : {sm['days_overwet']}",
        ]
        if result["key_events"]:
            lines += ["", "═══ KEY EVENTS ═══"] + [f"  {e}" for e in result["key_events"]]
        if result["missed_windows"]:
            lines += ["", "═══ MISSED WINDOWS ═══"] + [f"  {m}" for m in result["missed_windows"]]
        return "\n".join(lines)