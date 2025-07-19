
##############################################################
### This script contains functions for inspecting the data ###
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess

script_dir = Path(__file__).resolve().parent

matches_path = script_dir.parent / "exe" / "output" / "matches.csv"

matches = pd.read_csv(matches_path)

plt.style.use('ggplot')
sns.set_theme(style='whitegrid')

# Apply custom tweaks
plt.rcParams.update({
    'axes.facecolor': '#E5E5E5',       # light grey background
    'figure.facecolor': '#E5E5E5',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#111111',
    'xtick.color': '#111111',
    'ytick.color': '#111111',
    'text.color': '#111111',
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'axes.grid': True,
    'legend.facecolor': '#F5F5F5',
    'legend.edgecolor': '#DDDDDD',
    'savefig.facecolor': '#E5E5E5'
})

my_palette = ["#233D4D", "#FF9F1C", "#41EAD4", "#FDFFFC", "#F71735"]

# Scatterplot to compare favourite probability with Pinnacle overround
plt.figure(figsize=(8, 6))
plt.scatter(matches['fav_prob'], matches['total_prob'], color=my_palette[0], alpha=0.6)
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

for i, (prob_col, result_col) in enumerate(prob_result_map.items()):
    # Bin predicted probabilities into deciles
    matches[f'{prob_col}_decile'] = pd.qcut(matches[prob_col], q=10)

    # Group by bin: mean predicted prob and mean actual result
    grouped = matches.groupby(f'{prob_col}_decile', observed = False).agg(
        mean_pred_prob=(prob_col, 'mean'),
        mean_actual_result=(result_col, 'mean')
    )

    # Plot
    plt.plot(grouped['mean_pred_prob'], grouped['mean_actual_result'],
             marker='o', label=prob_col.upper(), color=my_palette[i])

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
home_df['lead_before'] = gamestate['home_lead_before']
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
away_df['lead_before'] = gamestate['home_lead_before'] * -1
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
sns.regplot(data=by_minute, x='minute', y='mean_goals', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean Goals by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.025)
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Mean Shots by Minute
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_shots', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean Shots by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean Shots')
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Mean xG by Minute
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_xg', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean xG by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean xG')
plt.ylim(0, 0.03)
plt.xticks(ticks=range(0,91, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- plot lead by minute ----
gamestate_long['abs_lead'] = gamestate_long['lead_before'].abs()
gamestate_long['lead_cat'] = gamestate_long['abs_lead'].apply(lambda x: str(int(x)) if x <= 2 else '3+')

# Proportion per minute
lead_counts = gamestate_long.groupby(['minute', 'lead_cat']).size().unstack(fill_value=0)
lead_props = lead_counts.div(lead_counts.sum(axis=1), axis=0).reset_index()

# Plot with custom colors
plt.figure(figsize=(12, 6))
for idx, col in enumerate(['0', '1', '2', '3+']):
    if col in lead_props.columns:
        plt.plot(
            lead_props['minute'], lead_props[col],
            marker='o', label=f'Lead {col}',
            color=my_palette[idx]
        )

plt.xlabel('Minute')
plt.ylabel('Proportion')
plt.title('Proportion of Game States by Minute (Absolute Lead)')
plt.xticks(ticks=range(0, 91, 10))
plt.grid(True)
plt.legend(title='Lead (Absolute)')
plt.tight_layout()
plt.show()

# ---- plot smoothed mean goals by minute for each lead_before value on one plot ----

lead_values = [-2, -1, 0, 1, 2]
colors = my_palette


# Plot mean goals per minute with LOESS smoothing
plt.figure(figsize=(12, 6))
for i, lead in enumerate(lead_values):
    subset = gamestate_long[gamestate_long['lead_before'] == lead]
    if not subset.empty:
        by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()
        sns.regplot(
            data=by_minute, x='minute', y='mean_goals',
            lowess=True, scatter=False, label=f'Lead {lead}',
            color=colors[i]
        )
plt.title('Smoothed Mean Goals by Minute for Each Lead')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks(ticks=range(0, 91, 10))
plt.grid(True)
plt.legend(title='Lead')
plt.tight_layout()
plt.show()

# Plot mean xG per minute with LOESS smoothing
plt.figure(figsize=(12, 6))
for i, lead in enumerate(lead_values):
    subset = gamestate_long[gamestate_long['lead_before'] == lead]
    if not subset.empty:
        by_minute = subset.groupby('minute').agg(mean_xg=('xg_for', 'mean')).reset_index()
        sns.regplot(
            data=by_minute, x='minute', y='mean_xg',
            lowess=True, scatter=False, label=f'Lead {lead}',
            color=colors[i]
        )
plt.title('Smoothed Mean xG by Minute for Each Lead')
plt.xlabel('Minute')
plt.ylabel('Mean xG')
plt.ylim(0, 0.03)
plt.xticks(ticks=range(0, 91, 10))
plt.grid(True)
plt.legend(title='Lead')
plt.tight_layout()
plt.show()


filters = {
    'Team Has 1 Red Card': (gamestate_long['red_cards_before'] == 1) & (gamestate_long['opp_red_cards_before'] == 0),
    'Opponent Has 1 Red Card': (gamestate_long['red_cards_before'] == 0) & (gamestate_long['opp_red_cards_before'] == 1)
}

lead_values = [-2, -1, 0, 1, 2]
colors = my_palette  # assumes 5 colors

for title_suffix, condition in filters.items():
    filtered_data = gamestate_long[condition]

    # ---- Smoothed Mean Goals Plot ----
    plt.figure(figsize=(12, 6))
    for i, lead in enumerate(lead_values):
        subset = filtered_data[filtered_data['lead_before'] == lead]
        if not subset.empty:
            by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()
            sns.regplot(
                data=by_minute, x='minute', y='mean_goals',
                lowess=True, scatter=False, label=f'Lead {lead}',
                color=colors[i]
            )
    plt.title(f'Smoothed Mean Goals by Minute ({title_suffix})')
    plt.xlabel('Minute')
    plt.ylabel('Mean Goals')
    plt.ylim(0, 0.03)
    plt.xticks(ticks=range(0, 91, 10))
    plt.grid(True)
    plt.legend(title='Lead')
    plt.tight_layout()
    plt.show()

    # ---- Smoothed Mean xG Plot ----
    plt.figure(figsize=(12, 6))
    for i, lead in enumerate(lead_values):
        subset = filtered_data[filtered_data['lead_before'] == lead]
        if not subset.empty:
            by_minute = subset.groupby('minute').agg(mean_xg=('xg_for', 'mean')).reset_index()
            sns.regplot(
                data=by_minute, x='minute', y='mean_xg',
                lowess=True, scatter=False, label=f'Lead {lead}',
                color=colors[i]
            )
    plt.title(f'Smoothed Mean xG by Minute ({title_suffix})')
    plt.xlabel('Minute')
    plt.ylabel('Mean xG')
    plt.ylim(0, 0.03)
    plt.xticks(ticks=range(0, 91, 10))
    plt.grid(True)
    plt.legend(title='Lead')
    plt.tight_layout()
    plt.show()

# ---- Show goals per min by quartile ----

gamestate_long['pscw_bin'] = pd.qcut(gamestate_long['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(gamestate_long['pscw_bin'].unique()):
    subset = gamestate_long[gamestate_long['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Smoothed Mean Goals by Minute')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.legend(title='Team Strength (Quartile)')
plt.grid(True)
plt.tight_layout()
plt.show()


neutral_state = gamestate_long[gamestate_long['lead_before'] == 0].copy()
neutral_state['pscw_bin'] = pd.qcut(neutral_state['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(neutral_state['pscw_bin'].unique()):
    subset = neutral_state[neutral_state['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Smoothed Mean Goals by Minute (Score is Level)')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.legend(title='Team Strength (Quartile)')
plt.grid(True)
plt.tight_layout()
plt.show()




# Filter to game states where score is level
neutral_state = gamestate_long[gamestate_long['lead_before'] == 1].copy()
neutral_state['pscw_bin'] = pd.qcut(neutral_state['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(neutral_state['pscw_bin'].unique()):
    subset = neutral_state[neutral_state['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Smoothed Mean Goals by Minute (Leading by One Goal)')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.legend(title='Team Strength (Quartile)')
plt.grid(True)
plt.tight_layout()
plt.show()

one_down = gamestate_long[gamestate_long['lead_before'] == -1].copy()
one_down['pscw_bin'] = pd.qcut(one_down['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(one_down['pscw_bin'].unique()):
    subset = one_down[one_down['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Smoothed Mean Goals by Minute (Trailing by One Goal)')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.legend(title='Team Strength (Quartile)')
plt.grid(True)
plt.tight_layout()
plt.show()

output_path = script_dir.parent / "exe" / "output" / "gamestate_long.csv"
gamestate_long.to_csv(output_path)
