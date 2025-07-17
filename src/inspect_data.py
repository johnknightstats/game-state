
##############################################################
### This script contains functions for inspecting the data ###
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    grouped = matches.groupby(f'{prob_col}_decile', observed = False).agg(
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

gamestate_path = script_dir.parent / "exe" / "output" / "game_state_timeline.csv"

gamestate = pd.read_csv(gamestate_path)

# Define columns that should be renamed or swapped
static_cols = ['country', 'league', 'season', 'date', 'match_id', 'minute']

# Home team perspective
home_df = gamestate[static_cols].copy()
home_df['goals_for_before'] = gamestate['home_goals_before']
home_df['goals_against_before'] = gamestate['away_goals_before']
home_df['red_cards_before'] = gamestate['home_red_cards_before']
home_df['opp_red_cards_before'] = gamestate['away_red_cards_before']
home_df['team_id'] = gamestate['home_team_id']
home_df['opp_id'] = gamestate['away_team_id']
home_df['team_name'] = gamestate['hometeam']
home_df['opp_name'] = gamestate['awayteam']
home_df['pscw'] = gamestate['psch']
home_df['pscl'] = gamestate['psca']
home_df['goal_for'] = gamestate['home_goal']
home_df['goal_against'] = gamestate['away_goal']
home_df['xg_for'] = gamestate['home_xg']
home_df['xg_against'] = gamestate['away_xg']
home_df['shots_for'] = gamestate['home_shots']
home_df['shots_against'] = gamestate['away_shots']

# Away team perspective
away_df = gamestate[static_cols].copy()
away_df['goals_for_before'] = gamestate['away_goals_before']
away_df['goals_against_before'] = gamestate['home_goals_before']
away_df['red_cards_before'] = gamestate['away_red_cards_before']
away_df['opp_red_cards_before'] = gamestate['home_red_cards_before']
away_df['team_id'] = gamestate['away_team_id']
away_df['opp_id'] = gamestate['home_team_id']
away_df['team_name'] = gamestate['awayteam']
away_df['opp_name'] = gamestate['hometeam']
away_df['pscw'] = gamestate['psca']
away_df['pscl'] = gamestate['psch']
away_df['goal_for'] = gamestate['away_goal']
away_df['goal_against'] = gamestate['home_goal']
away_df['xg_for'] = gamestate['away_xg']
away_df['xg_against'] = gamestate['home_xg']
away_df['shots_for'] = gamestate['away_shots']
away_df['shots_against'] = gamestate['home_shots']

# Combine both
gamestate_long = pd.concat([home_df, away_df], ignore_index=True)

# Ensure 'minute' column exists and is limited to 1–90
gamestate_long = gamestate_long[gamestate_long['minute'].between(1, 90)]

# Group by minute and calculate mean stats
by_minute = gamestate_long.groupby('minute').agg(
    mean_goals=('goal_for', 'mean'),
    mean_shots=('shots_for', 'mean'),
    mean_xg=('xg_for', 'mean')
).reset_index()

# Plot 1: Mean Goals by Minute
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_goals', scatter=True, lowess=False)
plt.title('Mean Goals by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Mean Shots by Minute
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_shots', scatter=True, lowess=False)
plt.title('Mean Shots by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean Shots')
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Mean xG by Minute
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_xg', scatter=True, lowess=False)
plt.title('Mean xG by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean xG')
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()
