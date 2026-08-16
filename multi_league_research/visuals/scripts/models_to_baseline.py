from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Run this script from the repository root.
input_file = Path("artifacts/scotland_model_evaluation/overall_metrics.csv")
output_file = Path("multi_league_research/visuals/out/model_comparison.png")

results = pd.read_csv(input_file)

labels = {
    "frequency_baseline": "Long-run result rates",
    "dixon_coles": "Team histor model",
    "player_form": "Player history model",
    "dixon_coles_player_form": "Team + player history model",
    "closing_market": "Bookmaker closing odds",
    "recalibrated_market": "Recalibrated bookmaker odds",
    "market_plus_player_form": "Bookmaker + player history model",
    "player_form_lightgbm":"Team + player history model (LGBM 1)",
    "expanded_player_form_lightgbm":"Team + player history model (LGBM 2)"
}

order = list(labels)

chart = results.set_index("model").loc[order].reset_index()
chart["label"] = chart["model"].map(labels)

# Convert the comparison with the market into percentages.
chart["difference"] = (chart["log_loss_relative_to_closing_market"] * 100)

colors = [
    "gray" if value == 0 else
    "seagreen" if value < 0 else
    "lightpink" if value < 10 else
    "palevioletred"
    for value in chart["difference"]
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(chart["label"], chart["difference"], color=colors)

ax.axvline(0, color="black", linewidth=1)
ax.invert_yaxis()

for bar, value in zip(bars, chart["difference"]):
    ax.text(
        value + (0.25 if value >= 0 else -0.25),
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.1f}%",
        va="center",
        ha="left" if value >= 0 else "right",
    )

ax.set_title("Scotland forecast models compared to bookmaker closing odds")
ax.set_xlabel("Difference from bookmaker odds (smaller is better)")

ax.set_ylabel("")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(output_file, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved chart to {output_file}")
