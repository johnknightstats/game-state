#####################################################################################
### Logistic regression forecasting goal from minute, game state and team quality ###
#####################################################################################

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# ---- Load data ----

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

gamestate_path = os.path.join(parent_dir, 'exe/output/gamestate_long.csv')
df = pd.read_csv(gamestate_path)

viz_path = os.path.join(parent_dir, 'viz')

# ---- Categorize leads ----

def cap_lead(val):
    if val <= -2:
        return "-2 or less"
    elif val == -1:
        return "-1"
    elif val == 0:
        return "0"
    elif val == 1:
        return "1"
    elif val >= 2:
        return "2 or more"

df['lead_cat'] = df['lead_before'].apply(cap_lead)
df['lead_cat'] = pd.Categorical(
    df['lead_cat'],
    categories=["-2 or less", "-1", "0", "1", "2 or more"],
    ordered=True
)

# First half: minute <= 45
first_half = df[df['minute'] <= 45].copy()

first_half['min1'] = (first_half['minute'] == 1)
first_half['min2'] = (first_half['minute'] == 2)

# Second half: minute >= 46; adjust minute to start at 1
second_half = df[df['minute'] >= 46].copy()
second_half['minute'] = second_half['minute'] - 45

second_half['min1'] = (second_half['minute'] == 1)
second_half['min2'] = (second_half['minute'] == 2)

# Standard regression formula
formula = 'goal_for ~ minute + C(lead_cat) + red_cards_before + opp_red_cards_before' \
' + pscw + min1 + min2'
# Formula with interactions
formula_int = 'goal_for ~ minute + C(lead_cat) + red_cards_before + opp_red_cards_before' \
' + pscw + min1 + min2 + C(lead_cat):minute + C(lead_cat):pscw + C(lead_cat):red_cards_before + C(lead_cat):opp_red_cards_before'

# ---- Train logistic regression for first half ----
print("\n=== Logistic Regression: First Half ===")
model1 = smf.logit(formula=formula, data=first_half).fit()
print(model1.summary())

# ---- Train logistic regression for second half ----
print("\n=== Logistic Regression: Second Half ===")
model2 = smf.logit(formula=formula, data=second_half).fit()
print(model2.summary())

# ---- Train logistic regression for first half with interactions ----
print("\n=== Logistic Regression: First Half ===")
model1i = smf.logit(formula=formula_int, data=first_half).fit()
print(model1i.summary())

# ---- Train logistic regression for second half with interactions ----
print("\n=== Logistic Regression: Second Half ===")
model2i = smf.logit(formula=formula_int, data=second_half).fit()
print(model2i.summary())

# ---- Diagnostics to see which models perform best ----

# Likelihood Ratio Test
lr_stat = 2 * (model1i.llf - model1.llf)
df_diff = model1i.df_model - model1.df_model
p_value = chi2.sf(lr_stat, df_diff)

print(f"Likelihood Ratio Test: Chi2 = {lr_stat:.4f}, df = {df_diff}, p = {p_value:.4f}")

# Akaike Information Criterion 
print("AIC (simple):", model1.aic)
print("AIC (interact):", model1i.aic)

# Same for 2nd half

lr_stat = 2 * (model2i.llf - model2.llf)
df_diff = model2i.df_model - model2.df_model
p_value = chi2.sf(lr_stat, df_diff)

print(f"Likelihood Ratio Test: Chi2 = {lr_stat:.4f}, df = {df_diff}, p = {p_value:.4f}")

print("AIC (simple):", model2.aic)
print("AIC (interact):", model2i.aic)

# --- Demonstrate goal probabilities for big fav (p = 0.8) and underdog (0.2) ----

scenarios = [
    {'pscw': 0.8, 'lead_cat': '0', 'label': 'Strong Team, Level (0)'},
    {'pscw': 0.2, 'lead_cat': '0', 'label': 'Weak Team, Level (0)'},
    {'pscw': 0.8, 'lead_cat': '-1', 'label': 'Strong Team, Trailing (-1)'},
    {'pscw': 0.2, 'lead_cat': '-1', 'label': 'Weak Team, Trailing (-1)'}
]

plt.figure(figsize=(12, 6))

for i, scenario in enumerate(scenarios):
    # Create prediction DataFrame
    predict_df = pd.DataFrame({
        'minute': np.arange(1, 91),
        'lead_cat': scenario['lead_cat'],
        'red_cards_before': 0,
        'opp_red_cards_before': 0,
        'pscw': scenario['pscw']
    })

    # Create min1 and min2 to flag first two minutes of each half
    predict_df['min1'] = (predict_df['minute'] == 1) | (predict_df['minute'] == 46)
    predict_df['min2'] = (predict_df['minute'] == 2) | (predict_df['minute'] == 47)

    # Split the two halves
    predict_df_first = predict_df[predict_df['minute'] <= 45].copy()
    predict_df_second = predict_df[predict_df['minute'] >= 46].copy()
    predict_df_second['minute'] = predict_df_second['minute'] - 45

    # Assign categories
    cat_levels = ["-2 or less", "-1", "0", "1", "2 or more"]
    predict_df_first['lead_cat'] = pd.Categorical(predict_df_first['lead_cat'], categories=cat_levels, ordered=True)
    predict_df_second['lead_cat'] = pd.Categorical(predict_df_second['lead_cat'], categories=cat_levels, ordered=True)

    # Predict
    predict_df_first['predicted_prob'] = model1.predict(predict_df_first)
    predict_df_second['predicted_prob'] = model2i.predict(predict_df_second)

    # Combine and restore minute
    predict_df_second['minute'] = predict_df_second['minute'] + 45
    full_pred = pd.concat([predict_df_first, predict_df_second], ignore_index=True)

    # Plot
    plt.plot(full_pred['minute'], full_pred['predicted_prob'], label=scenario['label'], color=my_palette[i])

plt.axvline(45.5, color='gray', linestyle='--', label='Half Time')
plt.xlabel('Minute')
plt.ylabel('Predicted Probability of Scoring')
plt.title('Predicted Goal Probability by Minute (80% Fav vs. 20% Fav)')
plt.ylim(0, None)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = os.path.join(viz_path, 'regression_80_20.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


# ---- Now a comparison for game states in an evenly-matched game (p = 0.36) ----

# Define new scenarios
scenarios = [
    {'pscw': 0.36, 'lead_cat': '-1', 'label': 'Trailing (-1)'},
    {'pscw': 0.36, 'lead_cat': '0',  'label': 'Level (0)'},
    {'pscw': 0.36, 'lead_cat': '1',  'label': 'Leading (1)'}
]

plt.figure(figsize=(12, 6))

for i, scenario in enumerate(scenarios):
    # Create prediction DataFrame
    predict_df = pd.DataFrame({
        'minute': np.arange(1, 91),
        'lead_cat': scenario['lead_cat'],
        'red_cards_before': 0,
        'opp_red_cards_before': 0,
        'pscw': scenario['pscw']
    })


    # Create min1 and min2 to flag first two minutes of each half
    predict_df['min1'] = (predict_df['minute'] == 1) | (predict_df['minute'] == 46)
    predict_df['min2'] = (predict_df['minute'] == 2) | (predict_df['minute'] == 47)

    # Split the two halves
    predict_df_first = predict_df[predict_df['minute'] <= 45].copy()
    predict_df_second = predict_df[predict_df['minute'] >= 46].copy()
    predict_df_second['minute'] = predict_df_second['minute'] - 45

    # Assign categories
    cat_levels = ["-2 or less", "-1", "0", "1", "2 or more"]
    predict_df_first['lead_cat'] = pd.Categorical(predict_df_first['lead_cat'], categories=cat_levels, ordered=True)
    predict_df_second['lead_cat'] = pd.Categorical(predict_df_second['lead_cat'], categories=cat_levels, ordered=True)

    # Predict
    predict_df_first['predicted_prob'] = model1.predict(predict_df_first)
    predict_df_second['predicted_prob'] = model2i.predict(predict_df_second)

    # Combine and restore minute
    predict_df_second['minute'] = predict_df_second['minute'] + 45
    full_pred = pd.concat([predict_df_first, predict_df_second], ignore_index=True)

    # Plot
    plt.plot(full_pred['minute'], full_pred['predicted_prob'],
             label=scenario['label'], color=my_palette[i], linewidth=2)

plt.axvline(45.5, color='gray', linestyle='--', label='Half Time')
plt.xlabel('Minute')
plt.ylabel('Predicted Probability of Scoring')
plt.title('Predicted Goal Probability of 36% Fav by Lead')
plt.ylim(0, None)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = os.path.join(viz_path, 'regression_36_36.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
