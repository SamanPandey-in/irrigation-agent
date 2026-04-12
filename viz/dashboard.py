"""
viz/dashboard.py — Phase 5 Hackathon Dashboard
================================================
Generates a multi-panel PDF/PNG dashboard:
  1. Training reward curve + yield progression
  2. Agent comparison bar charts (reward, yield, water, efficiency)
  3. Ablation study table
  4. Full-episode render (PPO vs Rule-Based side-by-side)
  5. Policy heatmap: action frequency by moisture × stage
  6. Impact statement overlay

Run:  python viz/dashboard.py
Out:  outputs/dashboard.png   outputs/episode_render_ppo.png
"""

from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODELS  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUTS, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"
GREEN  = "#3fb950"
BLUE   = "#58a6ff"
ORANGE = "#e3b341"
RED    = "#f85149"
PURPLE = "#bc8cff"
PINK   = "#ff7b72"

AGENT_COLORS = {
    "Random":         "#6e7681",
    "Rule-Based":     "#58a6ff",
    "Greedy":         PINK,
    "Smart Forecast": ORANGE,
    "PPO (trained)":  GREEN,
}

def sax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    if title:   ax.set_title(title, color=MUTED, fontsize=9, pad=6)
    if xlabel:  ax.set_xlabel(xlabel, color=MUTED, fontsize=7.5)
    if ylabel:  ax.set_ylabel(ylabel, color=MUTED, fontsize=7.5)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    return ax


# ── Load metrics ──────────────────────────────────────────────────────────────

def load_metrics():
    path = os.path.join(OUTPUTS, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print("⚠️  metrics.json not found — run train_and_eval.py first")
    return None


# ── Run one episode & collect history ────────────────────────────────────────

def run_episode(policy_fn, model, seed=42):
    from envs.precision_irrigation_v3 import PrecisionIrrigationEnvV3
    from rl.obs_normalizer import manual_normalize
    from rl.ppo import softmax
    env = PrecisionIrrigationEnvV3(seed=seed)
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    while True:
        if model is not None:
            obs_n     = manual_normalize(obs)[np.newaxis]
            logits, _ = model.forward(obs_n)
            action    = int(np.argmax(logits[0]))
        else:
            action = policy_fn(obs)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        if terminated or truncated:
            break
    h = env._history.copy()
    h["total_reward"] = total_r
    h["final_yield"]  = info["yield_t_ha"]
    h["cum_water_kl"] = info["cum_water_kl"]
    h["stress_days"]  = info["stress_days"]
    frame = env.render()
    env.close()
    return h, frame


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard panels
# ─────────────────────────────────────────────────────────────────────────────

def draw_training_curve(ax_r, ax_y, training):
    """Panel 1: reward + yield during training."""
    rewards  = training["ep_rewards"]
    yields   = training["ep_yields"]
    if not rewards: return

    window = 20
    def smooth(s):
        return np.convolve(s, np.ones(window)/window, mode="valid")

    x_r = np.arange(len(smooth(rewards)))
    ax_r.plot(smooth(rewards), color=BLUE, lw=1.5, label="Smoothed (w=20)")
    ax_r.scatter(np.arange(len(rewards)), rewards, color=BLUE, alpha=0.12, s=2)
    ax_r.axhline(0, color=BORDER, lw=0.5)
    ax_r.set_xlim(0, len(rewards))
    sax(ax_r, "PPO Training — Episode Reward (avg/step)", "Episode", "Reward")
    ax_r.legend(fontsize=6, labelcolor=TEXT, facecolor=PANEL, edgecolor=BORDER)

    x_y = np.arange(len(smooth(yields)))
    ax_y.plot(smooth(yields), color=GREEN, lw=1.5)
    ax_y.scatter(np.arange(len(yields)), yields, color=GREEN, alpha=0.12, s=2)
    ax_y.set_ylim(0, 8.5)
    ax_y.set_xlim(0, len(yields))
    sax(ax_y, "PPO Training — Yield per Episode", "Episode", "Yield (t/ha)")
    ax_y.axhline(6.0, color=ORANGE, lw=0.8, ls="--", label="6 t/ha baseline")
    ax_y.legend(fontsize=6, labelcolor=TEXT, facecolor=PANEL, edgecolor=BORDER)


def draw_comparison(ax_rwd, ax_yld, ax_wat, ax_eff, results):
    """Panel 2: comparison bar charts."""
    main = [r for r in results if r["name"] in AGENT_COLORS]
    names   = [r["name"] for r in main]
    colors  = [AGENT_COLORS.get(n, BLUE) for n in names]
    short_n = [n.replace(" (trained)","★").replace("Smart ","Smart\n") for n in names]

    for ax, key, title, unit in [
        (ax_rwd, "mean_reward",     "Mean Reward (avg/step)",       ""),
        (ax_yld, "mean_yield",      "Mean Yield",        "t/ha"),
        (ax_wat, "mean_water_kl",   "Water Used",        "kL"),
        (ax_eff, "mean_efficiency", "Water Efficiency",  "t/kL"),
    ]:
        vals = [r[key] for r in main]
        bars = ax.barh(short_n, vals, color=colors, height=0.6)
        ax.tick_params(axis="y", labelsize=7.5, colors=TEXT)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + abs(bar.get_width())*0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}{unit}", va="center", ha="left",
                    color=TEXT, fontsize=7)
        ax.set_xlim(right=max(vals)*1.25 if max(vals) > 0 else max(vals)*0.75)
        sax(ax, title)
        ax.axvline(0, color=BORDER, lw=0.5)


def draw_ablation_table(ax, results):
    """Panel: ablation study as a styled table."""
    ax.axis("off")
    ablation_names = ["PPO — no forecast", "PPO — drought",
                      "Forecast — drought", "Rule-Based — drought"]
    abl = [r for r in results if r["name"] in ablation_names]
    if not abl:
        ax.text(0.5, 0.5, "No ablation data", transform=ax.transAxes,
                color=MUTED, ha="center", va="center")
        return

    headers = ["Agent", "Reward", "Yield t/ha", "Water kL", "Δ vs Normal"]
    col_x   = [0.0, 0.38, 0.54, 0.70, 0.86]
    row_h   = 0.18
    header_y = 0.92

    ax.text(0.5, 1.0, "Ablation Studies", transform=ax.transAxes,
            color=MUTED, fontsize=9, ha="center", va="top")

    for i, (h, cx) in enumerate(zip(headers, col_x)):
        ax.text(cx, header_y, h, transform=ax.transAxes,
                color=TEXT, fontsize=7.5, fontweight="bold")

    # reference PPO result
    ppo_ref = next((r for r in results if r["name"] == "PPO (trained)"), None)

    for row_i, r in enumerate(abl):
        y = header_y - (row_i + 1) * row_h
        bg_color = "#1c2128" if row_i % 2 == 0 else "#21262d"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.05), 1.0, row_h * 0.95,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor=bg_color, edgecolor=BORDER, lw=0.5))

        delta = ""
        if ppo_ref:
            d = r["mean_reward"] - ppo_ref["mean_reward"]
            delta = f"{d:+.2f}"

        vals = [r["name"], f"{r['mean_reward']:+.2f}", f"{r['mean_yield']:.2f}",
                f"{r['mean_water_kl']:.1f}", delta]
        for cx, v in zip(col_x, vals):
            color = GREEN if "+" in v and v != vals[0] else RED if "-" in v and v != vals[0] else TEXT
            ax.text(cx, y, v, transform=ax.transAxes,
                    color=color, fontsize=7, va="center")


def draw_policy_heatmap(ax, model):
    """Panel: what action does PPO choose at each (moisture, stage) combo?"""
    from rl.obs_normalizer import manual_normalize
    import numpy as np

    if model is None:
        ax.text(0.5, 0.5, "No model", transform=ax.transAxes,
                color=MUTED, ha="center", va="center")
        return

    moistures = np.linspace(0.05, 0.95, 30)
    stages    = [0, 1, 2, 3]
    action_grid = np.zeros((len(stages), len(moistures)))

    for si, stage in enumerate(stages):
        for mi, moist in enumerate(moistures):
            obs = {
                "soil_moisture_obs": np.array([moist],        dtype=np.float32),
                "crop_growth":       np.array([0.5],          dtype=np.float32),
                "crop_stage":        np.array([stage],        dtype=np.int32),
                "weather_forecast":  np.zeros((3,2),          dtype=np.float32),
                "water_tank":        np.array([0.6],          dtype=np.float32),
                "power_status":      np.array([1],            dtype=np.int8),
                "day_of_season":     np.array([stage * 22],   dtype=np.int32),
                "grid_moisture":     np.full((4,4), moist,    dtype=np.float32),
            }
            obs_n = manual_normalize(obs)[np.newaxis]
            logits, _ = model.forward(obs_n)
            action_grid[si, mi] = np.argmax(logits[0])

    action_colors = ["#444444", "#74b9ff", "#0984e3", "#6c5ce7", "#fd79a8"]
    cmap = matplotlib.colors.ListedColormap(action_colors)
    im = ax.imshow(action_grid, aspect="auto", cmap=cmap, vmin=0, vmax=4,
                   extent=[moistures[0], moistures[-1], -0.5, 3.5],
                   origin="lower", interpolation="nearest")

    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["Seedling","Vegetative","Reproductive","Maturity"],
                       color=TEXT, fontsize=7)
    ax.axvline(0.22, color=RED,  lw=1.2, ls="--", alpha=0.7, label="Dry limit")
    ax.axvline(0.78, color=PINK, lw=1.2, ls="--", alpha=0.7, label="Wet limit")
    ax.axvline(0.40, color=GREEN, lw=0.8, ls=":",  alpha=0.5, label="Optimal lo")
    ax.axvline(0.72, color=GREEN, lw=0.8, ls=":",  alpha=0.5, label="Optimal hi")
    ax.legend(fontsize=6, labelcolor=TEXT, facecolor=PANEL, edgecolor=BORDER,
              loc="upper right", ncol=2)

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(action_colors,
                                      ["No water","Low","Medium","High","Drain"])]
    ax2 = ax.inset_axes([0.0, -0.22, 1.0, 0.15])
    ax2.axis("off")
    ax2.legend(handles=legend_patches, loc="center", ncol=5,
               fontsize=6.5, labelcolor=TEXT, facecolor=BG,
               edgecolor=BORDER, framealpha=1)

    sax(ax, "PPO Policy Heatmap: Action by Moisture × Stage", "Soil Moisture", "Crop Stage")


def draw_impact_panel(ax, results):
    """Panel: India impact statement with key numbers."""
    ax.set_facecolor("#0a1628")
    for sp in ax.spines.values(): sp.set_edgecolor("#1f4f8a")
    ax.set_xticks([]); ax.set_yticks([])

    ppo = next((r for r in results if r["name"] == "PPO (trained)"), None)
    rb  = next((r for r in results if r["name"] == "Rule-Based"),     None)
    rnd = next((r for r in results if r["name"] == "Random"),         None)

    ax.text(0.5, 0.93, "🌾 India Impact Projection", transform=ax.transAxes,
            color="#74b9ff", fontsize=11, fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.83, "Punjab Rice Farming · 140M Smallholders",
            transform=ax.transAxes, color=MUTED, fontsize=8, ha="center")

    stats = []
    if ppo and rb:
        y_gain = (ppo["mean_yield"] - rb["mean_yield"]) / max(rb["mean_yield"], 0.01) * 100
        w_save = (rb["mean_water_kl"] - ppo["mean_water_kl"]) / max(rb["mean_water_kl"], 0.01) * 100
        stats = [
            (f"{y_gain:+.1f}%",   "Yield vs Rule-Based",   GREEN if y_gain>0 else RED),
            (f"{w_save:+.1f}%",   "Water Δ vs Rule-Based",  GREEN if w_save>0 else RED),
            (f"{ppo['mean_yield']:.1f} t/ha", "PPO Mean Yield", BLUE),
            (f"{ppo['mean_water_kl']:.0f} kL","PPO Water Used", ORANGE),
        ]
    if not stats and rnd:
        stats = [
            (f"{rnd['mean_yield']:.1f} t/ha", "Random Yield",     RED),
            ("40%",                            "India Water Waste", RED),
            ("140M",                           "Farmers at Risk",   ORANGE),
            ("$10B",                           "Annual Losses",     ORANGE),
        ]

    for i, (val, label, color) in enumerate(stats):
        x = 0.12 + (i % 2) * 0.50
        y = 0.60 - (i // 2) * 0.28
        ax.text(x, y,     val,   transform=ax.transAxes, color=color,
                fontsize=18, fontweight="bold", ha="left")
        ax.text(x, y-0.10, label, transform=ax.transAxes, color=MUTED,
                fontsize=7.5, ha="left")

    ax.text(0.5, 0.05,
            "Training: 100k steps · 980 episodes · Pure NumPy PPO · No GPU required",
            transform=ax.transAxes, color=MUTED, fontsize=6.5, ha="center")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    metrics = load_metrics()
    results  = metrics["results"]  if metrics else []
    training = metrics["training"] if metrics else {"ep_rewards":[], "ep_yields":[], "update_losses":[]}

    # Load trained model
    model = None
    model_path = os.path.join(MODELS, "ppo_irrigation.pkl")
    if os.path.exists(model_path):
        from rl.ppo import ActorCritic
        model = ActorCritic.load(model_path)
        print(f"✅  Loaded PPO model from {model_path}")
    else:
        print("⚠️  No trained model found — run train_and_eval.py first")

    # ── Build dashboard ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle("🌾  Precision Irrigation RL Agent — Hackathon Dashboard",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.58, wspace=0.40,
                           left=0.05, right=0.97, top=0.94, bottom=0.04)

    # Row 0: training curves
    ax_tr = sax(fig.add_subplot(gs[0, :2]))
    ax_ty = sax(fig.add_subplot(gs[0, 2:]))
    draw_training_curve(ax_tr, ax_ty, training)

    # Row 1: comparison bars (4 metrics)
    ax_rwd = sax(fig.add_subplot(gs[1, 0]))
    ax_yld = sax(fig.add_subplot(gs[1, 1]))
    ax_wat = sax(fig.add_subplot(gs[1, 2]))
    ax_eff = sax(fig.add_subplot(gs[1, 3]))
    if results:
        draw_comparison(ax_rwd, ax_yld, ax_wat, ax_eff, results)

    # Row 2: policy heatmap | ablations | impact
    ax_heat = sax(fig.add_subplot(gs[2, 0:2]))
    draw_policy_heatmap(ax_heat, model)

    ax_abl = fig.add_subplot(gs[2, 2])
    ax_abl.set_facecolor(PANEL)
    for sp in ax_abl.spines.values(): sp.set_edgecolor(BORDER)
    if results:
        draw_ablation_table(ax_abl, results)

    ax_imp = fig.add_subplot(gs[2, 3])
    draw_impact_panel(ax_imp, results)

    out_path = os.path.join(OUTPUTS, "dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"📊  Dashboard → {out_path}")

    # ── Episode renders ──────────────────────────────────────────────────────
    def rb_policy(obs):
        m = float(obs["soil_moisture_obs"][0])
        if m < 0.28: return 3
        if m < 0.40: return 2
        if m > 0.78: return 4
        return 0

    print("  Running PPO episode render...")
    ppo_h, ppo_frame = run_episode(None, model, seed=7)
    if ppo_frame is not None:
        ppo_render_path = os.path.join(OUTPUTS, "episode_render_ppo.png")
        import matplotlib.image as mpimg
        plt.imsave(ppo_render_path, ppo_frame)
        print(f"🖼️   PPO episode render → {ppo_render_path}")

    print("  Running Rule-Based episode render...")
    rb_h, rb_frame = run_episode(rb_policy, None, seed=7)
    if rb_frame is not None:
        rb_render_path = os.path.join(OUTPUTS, "episode_render_rb.png")
        plt.imsave(rb_render_path, rb_frame)
        print(f"🖼️   Rule-Based episode render → {rb_render_path}")

    # ── Side-by-side comparison render ───────────────────────────────────────
    if ppo_frame is not None and rb_frame is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(24, 9), facecolor=BG)
        for ax, frame, title in zip(axes, [ppo_frame, rb_frame],
                                    ["PPO Agent", "Rule-Based Agent"]):
            ax.imshow(frame); ax.axis("off")
            ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=8)
        plt.tight_layout()
        side_path = os.path.join(OUTPUTS, "comparison_render.png")
        fig2.savefig(side_path, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"🖼️   Side-by-side → {side_path}")

    print("\n✅  Phase 5 complete — all outputs saved to outputs/")


if __name__ == "__main__":
    main()
