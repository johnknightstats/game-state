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
from matplotlib.lines import Line2D
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
import os

# Fit the model
print("\n=== Logistic Regression: First Half ===")
model1 = smf.logit(formula=formula, data=first_half).fit()
print(model1.summary())

# Extract and rename for readability
summary_df = model1.summary2().tables[1].copy()
summary_df = summary_df.rename(columns={
    'Coef.': 'Coefficient',
    'Std.Err.': 'Std. Error',
    'z': 'z-value',
    'P>|z|': 'p-value',
    '[0.025': 'CI Lower',
    '0.975]': 'CI Upper'
})
summary_df = summary_df.round(3)

# Create a mapping of original variable names to nicer labels
var_names = {
    'pscw': 'Pre-match Win Prob',
    'C(lead_cat)[T.-1]': 'Trailing (-1)',
    'C(lead_cat)[T.0]': 'Level (0)',
    'C(lead_cat)[T.1]': 'Leading (1)',
    'C(lead_cat)[T.2 or more]': 'Leading (2 or more)',
    'red_cards_before': 'Red Card',
    'opp_red_cards_before': 'Opponent Red Card',
    'min1[T.True]': 'First Min of Half',
    'min2[T.True]': 'Second Min of Half',
    'minute': "Minute"
}

# Apply to the index
summary_df.rename(index=var_names, inplace=True)


# Convert to HTML table
table_html = summary_df.to_html(classes='styled-regression-table', border=0)

# Convert to HTML table
table_html = summary_df.to_html(classes='styled-regression-table', border=0)

# Combine into full HTML document
fragment_html = f"""
<style>
.styled-regression-table {{
  font-family: Arial, sans-serif;
  font-size: 14px;
  border-collapse: collapse;
  width: 100%;
  margin-top: 1em;
}}
.styled-regression-table th, .styled-regression-table td {{
  padding: 8px 12px;
  border: 1px solid #ccc;
  text-align: right;
}}
.styled-regression-table th {{
  background-color: #f2f2f2;
}}
.styled-regression-table tr:nth-child(even) {{
  background-color: #f9f9f9;
}}
</style>
<h2>Logistic Regression: First Half</h2>
{table_html}
"""

# Save
regress_path = os.path.join(parent_dir, 'viz', 'model1.html')
os.makedirs(os.path.dirname(regress_path), exist_ok=True)

with open(regress_path, 'w', encoding='utf-8') as f:
    f.write(fragment_html)



# ---- Train logistic regression for second half ----
print("\n=== Logistic Regression: Second Half ===")
model2 = smf.logit(formula=formula, data=second_half).fit()
print(model2.summary())
# Extract and rename for readability
summary_df = model2.summary2().tables[1].copy()
summary_df = summary_df.rename(columns={
    'Coef.': 'Coefficient',
    'Std.Err.': 'Std. Error',
    'z': 'z-value',
    'P>|z|': 'p-value',
    '[0.025': 'CI Lower',
    '0.975]': 'CI Upper'
})
summary_df = summary_df.round(3)

# Create a mapping of original variable names to nicer labels
var_names = {
    'pscw': 'Pre-match Win Prob',
    'C(lead_cat)[T.-1]': 'Trailing (-1)',
    'C(lead_cat)[T.0]': 'Level (0)',
    'C(lead_cat)[T.1]': 'Leading (1)',
    'C(lead_cat)[T.2 or more]': 'Leading (2 or more)',
    'red_cards_before': 'Red Card',
    'opp_red_cards_before': 'Opponent Red Card',
    'min1[T.True]': 'First Min of Half',
    'min2[T.True]': 'Second Min of Half',
    'minute': "Minute"
}

# Apply to the index
summary_df.rename(index=var_names, inplace=True)


# Convert to HTML table
table_html = summary_df.to_html(classes='styled-regression-table', border=0)

# Combine into full HTML document
fragment_html = f"""
<style>
.styled-regression-table {{
  font-family: Arial, sans-serif;
  font-size: 14px;
  border-collapse: collapse;
  width: 100%;
  margin-top: 1em;
}}
.styled-regression-table th, .styled-regression-table td {{
  padding: 8px 12px;
  border: 1px solid #ccc;
  text-align: right;
}}
.styled-regression-table th {{
  background-color: #f2f2f2;
}}
.styled-regression-table tr:nth-child(even) {{
  background-color: #f9f9f9;
}}
</style>
<h2>Logistic Regression: Second Half</h2>
{table_html}
"""

# Save
regress_path = os.path.join(parent_dir, 'viz', 'model2.html')
os.makedirs(os.path.dirname(regress_path), exist_ok=True)

with open(regress_path, 'w', encoding='utf-8') as f:
    f.write(fragment_html)


# ---- Train logistic regression for first half with interactions ----
print("\n=== Logistic Regression: First Half ===")
model1i = smf.logit(formula=formula_int, data=first_half).fit()
print(model1i.summary())
summary_df = model1i.summary2().tables[1]  # Coefficient table
html_table = summary_df.to_html(classes='table table-sm table-bordered', float_format="%.3f")
regress_path = os.path.join(parent_dir, 'viz\\model1i.html')
with open(regress_path, 'w') as f:
    f.write(html_table)


# ---- Train logistic regression for second half with interactions ----
print("\n=== Logistic Regression: Second Half ===")
model2i = smf.logit(formula=formula_int, data=second_half).fit()
print(model2i.summary())
summary_df = model2i.summary2().tables[1]  # Coefficient table
html_table = summary_df.to_html(classes='table table-sm table-bordered', float_format="%.3f")
regress_path = os.path.join(parent_dir, 'viz\\model2i.html')
with open(regress_path, 'w') as f:
    f.write(html_table)


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
    predict_df_second['predicted_prob'] = model2.predict(predict_df_second)

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
    predict_df_second['predicted_prob'] = model2.predict(predict_df_second)

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


# ---- Plot quartiles of pscw by game state (lead) ----

# Define PSCW and lead categories
pscw_values = [0.157, 0.297, 0.433, 0.641]
lead_cats = ['-1', '0', '1']
lead_styles = {'-1': '--', '0': '-', '1': ':'}
lead_labels = {'-1': 'Trailing by 1', '0': 'Level', '1': 'Leading by 1'}
pscw_labels = [f"p = {p:.3f}" for p in pscw_values]

plt.figure(figsize=(10, 6))

# Plot all lines
for i, pscw in enumerate(pscw_values):
    color = my_palette[i]
    for lead_cat in lead_cats:
        predict_df = pd.DataFrame({
            'minute': np.arange(1, 91),
            'lead_cat': lead_cat,
            'red_cards_before': 0,
            'opp_red_cards_before': 0,
            'pscw': pscw
        })

        predict_df['min1'] = (predict_df['minute'] == 1) | (predict_df['minute'] == 46)
        predict_df['min2'] = (predict_df['minute'] == 2) | (predict_df['minute'] == 47)

        predict_df_first = predict_df[predict_df['minute'] <= 45].copy()
        predict_df_second = predict_df[predict_df['minute'] >= 46].copy()
        predict_df_second['minute'] -= 45

        cat_levels = ["-2 or less", "-1", "0", "1", "2 or more"]
        predict_df_first['lead_cat'] = pd.Categorical(predict_df_first['lead_cat'], categories=cat_levels, ordered=True)
        predict_df_second['lead_cat'] = pd.Categorical(predict_df_second['lead_cat'], categories=cat_levels, ordered=True)

        predict_df_first['predicted_prob'] = model1.predict(predict_df_first)
        predict_df_second['predicted_prob'] = model2.predict(predict_df_second)

        predict_df_second['minute'] += 45
        full_pred = pd.concat([predict_df_first, predict_df_second], ignore_index=True)

        plt.plot(
            full_pred['minute'], full_pred['predicted_prob'],
            color=color,
            linestyle=lead_styles[lead_cat]
        )

# Labels and aesthetics
plt.xlabel('Minute')
plt.ylabel('Predicted Probability of Scoring')
plt.title('Predicted Goal Probability by Pre-Match Win Probability and Lead')
plt.ylim(0, None)
plt.xticks([0, 15, 30, 45, 60, 75, 90])
plt.grid(True)

# ---- Custom Legend ----

# Color legend: one per PSCW level
color_handles = [
    Line2D([0], [0], color=my_palette[i], lw=2, linestyle='-') for i in range(len(pscw_values))
]
# Color legend (left)
color_legend = plt.legend(
    color_handles,
    pscw_labels,
    title="Pre-Match Win Probability",
    loc='upper left',
    bbox_to_anchor=(0.01, 0.99)
)

# Style legend: one per lead category
style_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle=lead_styles[cat]) for cat in lead_cats
]
style_legend = plt.legend(
    style_handles,
    [lead_labels[cat] for cat in lead_cats],
    title="Lead State",
    loc='upper left',
    bbox_to_anchor=(0.25, 0.99)
)

# Add both legends
plt.gca().add_artist(color_legend)

# Light grey background
for leg in [color_legend, style_legend]:
    leg.get_frame().set_facecolor('lightgrey')
    leg.get_frame().set_edgecolor('black')

plt.tight_layout()
save_path = os.path.join(viz_path, 'combined_predictions_odds_lead.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()