
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