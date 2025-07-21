
##############################################################
### This script contains functions for inspecting the data ###
##############################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

matches_path = os.path.join(parent_dir, 'exe/output/matches.csv')
matches = pd.read_csv(matches_path)

viz_path = os.path.join(parent_dir, 'viz')

# ---- Custom styles and colours for plots ----

plt.style.use('ggplot')
sns.set_theme(style='whitegrid')

plt.rcParams.update({
    'axes.facecolor': '#E5E5E5',
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
    'savefig.facecolor': '#E5E5E5',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

my_palette = ["#233D4D", "#FF9F1C", "#41EAD4", "#FDFFFC", "#F71735"]

# ---- Compare favourite probability with Pinnacle overround ----

plt.figure(figsize=(8, 6))
plt.scatter(matches['fav_prob'], matches['total_prob'], color=my_palette[0], alpha=0.6)
plt.xlabel('Favorite Probability')
plt.ylabel('Total Probability')
plt.title('Favourite Probability vs. Market Overround')
plt.grid(True)
plt.tight_layout()
plt.show()

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

    plt.plot(grouped['mean_pred_prob'], grouped['mean_actual_result'],
             marker='o', label=prob_col.upper(), color=my_palette[i])

# Reference line x = y
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

plt.xlabel('Predicted Win Probability')
plt.ylabel('Observed Win Relative Frequency')
plt.title('Bookmaker Probabilities vs. Observed Outcomes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Reshape gamestate data to both home & away team perspective ----

gamestate_path = os.path.join(parent_dir, "exe/output/game_state_timeline.csv")

gamestate = pd.read_csv(gamestate_path)

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

# ---- Plot mean goals by minute ----

plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_goals', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean Goals by Minute\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.025)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_by_minute.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot mean shots by minute ----
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_shots', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean Shots by Minute\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Shots')
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot mean xG by minute ----
plt.figure(figsize=(10, 6))
sns.regplot(data=by_minute, x='minute', y='mean_xg', scatter=True, lowess=False, color=my_palette[0])
plt.title('Mean xG by Minute\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean xG')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_xg_by_minute.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot lead by minute ----
gamestate_long['abs_lead'] = gamestate_long['lead_before'].abs()
gamestate_long['lead_cat'] = gamestate_long['abs_lead'].apply(lambda x: str(int(x)) if x <= 2 else '3+')

lead_counts = gamestate_long.groupby(['minute', 'lead_cat']).size().unstack(fill_value=0)
lead_props = lead_counts.div(lead_counts.sum(axis=1), axis=0).reset_index()

plt.figure(figsize=(12, 6))
for idx, col in enumerate(['0', '1', '2', '3+']):
    if col in lead_props.columns:
        plt.plot(
            lead_props['minute'], lead_props[col],
            marker='o', label=str(col),
            color=my_palette[idx]
        )

plt.xlabel('Minute')
plt.ylabel('Proportion')
plt.title('Game State by Minute\nBig 5 Leagues 2017-2025')
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.legend(title='Lead')
plt.tight_layout()
save_path = os.path.join(viz_path, 'game_state_by_minute.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot loess-smoothed mean goals per minute for different leads ----

lead_values = [-2, -1, 0, 1, 2]

plt.figure(figsize=(12, 6))
for i, lead in enumerate(lead_values):
    subset = gamestate_long[gamestate_long['lead_before'] == lead]
    if not subset.empty:
        by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()
        sns.regplot(
            data=by_minute, x='minute', y='mean_goals',
            lowess=True, scatter=False, label=str(lead),
            color=my_palette[i]
        )
plt.title('Mean Goals per Minute by Lead (Loess Smoothed)\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.legend(title='Lead')
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_by_lead.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# --- Plot xG instead of goals
plt.figure(figsize=(12, 6))
for i, lead in enumerate(lead_values):
    subset = gamestate_long[gamestate_long['lead_before'] == lead]
    if not subset.empty:
        by_minute = subset.groupby('minute').agg(mean_xg=('xg_for', 'mean')).reset_index()
        sns.regplot(
            data=by_minute, x='minute', y='mean_xg',
            lowess=True, scatter=False, label=f'Lead {lead}',
            color=my_palette[i]
        )
plt.title('Mean xG per Minute by Lead (Smoothed)\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean xG')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.legend(title='Lead')
plt.tight_layout()
plt.show()


filters = {
    'Team Has 1 Red Card': (gamestate_long['red_cards_before'] == 1) & (gamestate_long['opp_red_cards_before'] == 0),
    'Opponent Has 1 Red Card': (gamestate_long['red_cards_before'] == 0) & (gamestate_long['opp_red_cards_before'] == 1)
}

lead_values = [-2, -1, 0, 1, 2]

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
                color=my_palette[i]
            )
    plt.title(f'Smoothed Mean Goals by Minute ({title_suffix})\nBig 5 Leagues 2017-2025')
    plt.xlabel('Minute')
    plt.ylabel('Mean Goals')
    plt.ylim(0, 0.03)
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
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
                color=my_palette[i]
            )
    plt.title(f'Smoothed Mean xG by Minute ({title_suffix})\nBig 5 Leagues 2017-2025')
    plt.xlabel('Minute')
    plt.ylabel('Mean xG')
    plt.ylim(0, 0.03)
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.grid(True)
    plt.legend(title='Lead')
    plt.tight_layout()
    plt.show()

# ---- Goals per min by pre-match odds quartile ----

gamestate_long['pscw_bin'] = pd.qcut(gamestate_long['pscw'], q=4, labels=False)

# Group by bin and compute min, max, and median
summary = gamestate_long.groupby('pscw_bin')['pscw'].agg(['min', 'median', 'max']).reset_index()
summary['pscw_bin'] = summary['pscw_bin'] + 1  # so bins are 1 to 4
summary.columns = ['Quartile', 'Min PSCW', 'Median PSCW', 'Max PSCW']
print(summary)

plt.figure(figsize=(12, 6))
for i in sorted(gamestate_long['pscw_bin'].unique()):
    subset = gamestate_long[gamestate_long['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'{i+1}', color=my_palette[i])

plt.title('Mean Goals per Minute by Pre-Match Odds Quartile\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.legend(title='Quartile')
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_by_odds.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot quartiles when scores are level ----

neutral_state = gamestate_long[gamestate_long['lead_before'] == 0].copy()
neutral_state['pscw_bin'] = pd.qcut(neutral_state['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(neutral_state['pscw_bin'].unique()):
    subset = neutral_state[neutral_state['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Mean Goals per Minute (Scores Level)\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.legend(title='Quartile')
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_scores_level.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot quartiles when team is one goal up ----

neutral_state = gamestate_long[gamestate_long['lead_before'] == 1].copy()
neutral_state['pscw_bin'] = pd.qcut(neutral_state['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(neutral_state['pscw_bin'].unique()):
    subset = neutral_state[neutral_state['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    # Apply LOWESS smoothing
    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Mean Goals per Minute (Leading by One Goal)\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.legend(title='Quartile')
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_one_up.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- Plot quartiles when team is one goal down ----

one_down = gamestate_long[gamestate_long['lead_before'] == -1].copy()
one_down['pscw_bin'] = pd.qcut(one_down['pscw'], q=4, labels=False)

plt.figure(figsize=(12, 6))
for i in sorted(one_down['pscw_bin'].unique()):
    subset = one_down[one_down['pscw_bin'] == i]
    by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

    smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'Bin {i+1}', color=my_palette[i])

plt.title('Mean Goals per Minute (Trailing by One Goal)\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.legend(title='Quartile')
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_one_down.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ---- All plots combined ----

# Define line styles
lead_styles = {0: '-', -1: '--', 1: ':'}
lead_labels = {0: 'Level', -1: 'Trailing by 1', 1: 'Leading by 1'}

plt.figure(figsize=(14, 7))

# Loop over each game state
for lead_before in [0, 1, -1]:
    subset_df = gamestate_long[gamestate_long['lead_before'] == lead_before].copy()
    subset_df['pscw_bin'] = pd.qcut(subset_df['pscw'], q=4, labels=False)

    for i in sorted(subset_df['pscw_bin'].unique()):
        subset = subset_df[subset_df['pscw_bin'] == i]
        by_minute = subset.groupby('minute').agg(mean_goals=('goal_for', 'mean')).reset_index()

        smoothed = lowess(by_minute['mean_goals'], by_minute['minute'], frac=0.2)
        plt.plot(
            smoothed[:, 0], smoothed[:, 1],
            label=f'Q {i+1} ({lead_labels[lead_before]})',
            color=my_palette[i],
            linestyle=lead_styles[lead_before]
        )

plt.title('Mean Goals per Minute by Pre-Match Odds and Lead\nBig 5 Leagues 2017-2025')
plt.xlabel('Minute')
plt.ylabel('Mean Goals')
plt.ylim(0, 0.03)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
legend = plt.legend(title='Quartile and Game State', ncol=2)
legend.get_frame().set_facecolor('lightgrey')
legend.get_frame().set_edgecolor('black')
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(viz_path, 'mean_goals_per_minute_combined.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


# ---- Write the gamestate csv to file ----

output_path = os.path.join(parent_dir, "exe/output/gamestate_long.csv")
gamestate_long.to_csv(output_path)
