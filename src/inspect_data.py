
##############################################################
### This script contains functions for inspecting the data ###
##############################################################

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent

matches_path = script_dir.parent / "exe" / "output" / "matches.csv"

matches = pd.read_csv(matches_path)

# Scatterplot to compare favourite probability with Pinnacle overround
plt.figure(figsize=(8, 6))
plt.scatter(matches['fav_prob'], matches['total_prob'], alpha=0.6)
plt.xlabel('Favorite Probability')
plt.ylabel('Total Probability')
plt.title('fav_prob vs. total_prob')
plt.grid(True)
plt.tight_layout()
plt.show()

# Define mapping from prob column to actual result column
prob_result_map = {
    'psch': 'home_win',
    'psca': 'away_win',
    'pscd': 'draw'
}

plt.figure(figsize=(10, 6))

for prob_col, result_col in prob_result_map.items():
    # Bin predicted probabilities into deciles
    matches[f'{prob_col}_decile'] = pd.qcut(matches[prob_col], q=10)

    # Group by bin: mean predicted prob and mean actual result
    grouped = matches.groupby(f'{prob_col}_decile').agg(
        mean_pred_prob=(prob_col, 'mean'),
        mean_actual_result=(result_col, 'mean')
    )

    # Plot
    plt.plot(grouped['mean_pred_prob'], grouped['mean_actual_result'],
             marker='o', label=prob_col.upper())

# Reference line x = y
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

# Labels and formatting
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Frequency')
plt.title('Calibration Plot: Bookmaker Probabilities vs. Outcomes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()