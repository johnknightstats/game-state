
import pandas as pd
import numpy as np

def clean_and_merge_odds(odds: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:

    # Keep only required columns from odds df
    odds = odds[[
        'country', 'league', 'season', 'date', 'time', 'hometeam', 'awayteam',
        'fthg', 'ftag', 'ftr', 'psch', 'pscd', 'psca'
    ]].copy()

    # Drop rows where psch is missing or empty string
    odds = odds[odds['psch'].notna() & (odds['psch'] != "")]

    # Merge home team id and away team id
    xwalk_fd = xwalk[xwalk['source'] == 'Football_Data'][['team_name', 'team_id']]

    odds = odds.merge(
        xwalk_fd.rename(columns={'team_name': 'hometeam', 'team_id': 'home_team_id'}),
        on='hometeam', how='left'
    )

    odds = odds.merge(
        xwalk_fd.rename(columns={'team_name': 'awayteam', 'team_id': 'away_team_id'}),
        on='awayteam', how='left'
    )

    return odds

import pandas as pd

def find_unmatched_matches(shots: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare key columns
    shot_keys = shots[['match_date', 'home_team_id', 'away_team_id']].drop_duplicates().copy()
    odds_keys = odds[['date', 'home_team_id', 'away_team_id']].copy().rename(columns={'date': 'match_date'})

    # Convert to comparable date format
    shot_keys['match_date'] = pd.to_datetime(shot_keys['match_date']).dt.date
    odds_keys['match_date'] = pd.to_datetime(odds_keys['match_date']).dt.date

    # Find unmatched in shots (not in odds)
    merged_from_shots = shot_keys.merge(odds_keys, on=['match_date', 'home_team_id', 'away_team_id'], how='left', indicator=True)
    unmatched_from_shots = merged_from_shots[merged_from_shots['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Find unmatched in odds (not in shots)
    merged_from_odds = odds_keys.merge(shot_keys, on=['match_date', 'home_team_id', 'away_team_id'], how='left', indicator=True)
    unmatched_from_odds = merged_from_odds[merged_from_odds['_merge'] == 'left_only'].drop(columns=['_merge'])

    return unmatched_from_shots, unmatched_from_odds

def remove_unmatched_odds(odds: pd.DataFrame, unmatched_from_odds: pd.DataFrame) -> pd.DataFrame:
    # Ensure date columns are in the same format
    odds['date'] = pd.to_datetime(odds['date']).dt.date
    unmatched_from_odds['match_date'] = pd.to_datetime(unmatched_from_odds['match_date']).dt.date

    # Rename match_date in unmatched to align with odds
    unmatched_from_odds = unmatched_from_odds.rename(columns={'match_date': 'date'})

    # Perform anti-join: keep only rows NOT in unmatched_from_odds
    cleaned_odds = odds.merge(
        unmatched_from_odds,
        on=['date', 'home_team_id', 'away_team_id'],
        how='left',
        indicator=True
    )
    cleaned_odds = cleaned_odds[cleaned_odds['_merge'] == 'left_only'].drop(columns=['_merge'])

    return cleaned_odds

def add_match_id_to_odds(odds: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    # Ensure consistent date format
    odds['date'] = pd.to_datetime(odds['date']).dt.date
    shots['match_date'] = pd.to_datetime(shots['match_date']).dt.date

    # Reduce shots to distinct (match_id, match_date, home_team_id, away_team_id)
    shot_keys = shots[['match_id', 'match_date', 'home_team_id', 'away_team_id']].drop_duplicates()

    # Merge to add match_id to odds
    odds_with_match_id = odds.merge(
        shot_keys,
        left_on=['date', 'home_team_id', 'away_team_id'],
        right_on=['match_date', 'home_team_id', 'away_team_id'],
        how='left'
    )

    # Drop the redundant match_date column
    odds_with_match_id = odds_with_match_id.drop(columns=['match_date'])

    return odds_with_match_id

def odds_to_probs(odds_with_match_id: pd.DataFrame) -> pd.DataFrame:

    odds_with_match_id['psch'] = 1 / odds_with_match_id['psch']
    odds_with_match_id['pscd'] = 1 / odds_with_match_id['pscd']
    odds_with_match_id['psca'] = 1 / odds_with_match_id['psca']

    odds_with_match_id['fav_prob'] = odds_with_match_id[['psch', 'psca']].max(axis=1)
    odds_with_match_id['total_prob'] = odds_with_match_id[['psch', 'pscd', 'psca']].sum(axis=1)

    odds_with_match_id['home_win'] = (odds_with_match_id['ftr'] == 'H')
    odds_with_match_id['away_win'] = (odds_with_match_id['ftr'] == 'A')
    odds_with_match_id['draw'] = (odds_with_match_id['ftr'] == 'D')

    odds_with_match_id['home_adj'] = odds_with_match_id['psch'] - ((odds_with_match_id['total_prob'] - 1) / 3)
    odds_with_match_id['away_adj'] = odds_with_match_id['psca'] - ((odds_with_match_id['total_prob'] - 1) / 3)
    odds_with_match_id['draw_adj'] = odds_with_match_id['pscd'] - ((odds_with_match_id['total_prob'] - 1) / 3)

    return odds_with_match_id



def generate_game_state_timeline(
    odds_with_match_id: pd.DataFrame,
    goals: pd.DataFrame,
    cards: pd.DataFrame,
    shots: pd.DataFrame
) -> pd.DataFrame:

    # Prepare goals and red card data
    goals_df = goals[['match_id', 'team_id', 'minute', 'period']].copy()
    goals_df['event_type'] = 'goal'

    red_cards = cards[cards['card_type'].isin(['red', 'second_yellow'])][['match_id', 'team_id', 'minute', 'period']].copy()
    red_cards['event_type'] = 'red_card'

    events = pd.concat([goals_df, red_cards], ignore_index=True)

    # Flag events
    events['goal'] = (events['event_type'] == 'goal').astype(int)
    events['red_card'] = (events['event_type'] == 'red_card').astype(int)
    events = events.drop(columns='event_type')

    # Create base table with all match-team-minute combinations (90 minutes)
    match_teams = pd.concat([
        odds_with_match_id[['match_id', 'home_team_id']].rename(columns={'home_team_id': 'team_id'}),
        odds_with_match_id[['match_id', 'away_team_id']].rename(columns={'away_team_id': 'team_id'})
    ]).drop_duplicates()

    full_minutes = pd.DataFrame({'minute': range(1, 91)})
    base = match_teams.assign(key=1).merge(full_minutes.assign(key=1), on='key').drop(columns='key')

    # Merge odds to get home/away identification
    base = base.merge(odds_with_match_id[['match_id', 'home_team_id', 'away_team_id']], on='match_id', how='left')
    base['is_home'] = base['team_id'] == base['home_team_id']
    base['is_away'] = base['team_id'] == base['away_team_id']

    # Prepare cumulative goals/red cards
    cumulative_records = []
    for minute in range(1, 91):
        if minute <= 45:
            relevant_events = events[(events['period'] == 1) & (events['minute'] < minute)]
        else:
            relevant_events = events[
                ((events['period'] == 1)) |
                ((events['period'] == 2) & (events['minute'] < minute))
            ]

        agg = (
            relevant_events
            .groupby(['match_id', 'team_id'])[['goal', 'red_card']]
            .sum()
            .reset_index()
            .assign(minute=minute)
        )

        cumulative_records.append(agg)

    cumulative_df = pd.concat(cumulative_records, ignore_index=True)

    # Merge cumulative data back to base
    result = base.merge(cumulative_df, on=['match_id', 'team_id', 'minute'], how='left').fillna(0)

    # Split into home and away
    home = result[result['is_home']][['match_id', 'minute', 'goal', 'red_card']].rename(
        columns={'goal': 'home_goals_before', 'red_card': 'home_red_cards_before'}
    )
    away = result[result['is_away']][['match_id', 'minute', 'goal', 'red_card']].rename(
        columns={'goal': 'away_goals_before', 'red_card': 'away_red_cards_before'}
    )

    final = home.merge(away, on=['match_id', 'minute'])

    final['home_lead_before'] = final['home_goals_before'] - final['away_goals_before']

    # Add match-level information
    match_info_cols = ['match_id', 'home_team_id', 'away_team_id', 'hometeam', 'awayteam', 'date', 'country',
                       'league', 'season', 'psch', 'pscd', 'psca']
    match_info = odds_with_match_id[match_info_cols].drop_duplicates()
    final = final.merge(match_info, on='match_id', how='left')

    # Add binary home_goal and away_goal indicators
    goals_home = goals_df.merge(odds_with_match_id[['match_id', 'home_team_id']], on='match_id')
    goals_home = goals_home[goals_home['team_id'] == goals_home['home_team_id']]
    goals_home = goals_home[
        ((goals_home['period'] == 1) & (goals_home['minute'].between(1, 45))) |
        ((goals_home['period'] == 2) & (goals_home['minute'].between(46, 90)))
    ][['match_id', 'minute']]
    goals_home['home_goal'] = 1

    goals_away = goals_df.merge(odds_with_match_id[['match_id', 'away_team_id']], on='match_id')
    goals_away = goals_away[goals_away['team_id'] == goals_away['away_team_id']]
    goals_away = goals_away[
        ((goals_away['period'] == 1) & (goals_away['minute'].between(1, 45))) |
        ((goals_away['period'] == 2) & (goals_away['minute'].between(46, 90)))
    ][['match_id', 'minute']]
    goals_away['away_goal'] = 1

    final = final.merge(goals_home, on=['match_id', 'minute'], how='left')
    final = final.merge(goals_away, on=['match_id', 'minute'], how='left')
    final['home_goal'] = final['home_goal'].fillna(0).astype(int)
    final['away_goal'] = final['away_goal'].fillna(0).astype(int)

    # ---------------------------------
    # Add home_xg, away_xg, home_shots, away_shots
    # ---------------------------------

    # Filter to valid shots (optional, if needed)
    shots = shots.copy()
    shots = shots[((shots['period'] == 1) & (shots['minute'] <= 45)) |
                ((shots['period'] == 2) & (shots['minute'] >= 46))]

    # Assign home/away team_id using home_team_shot
    shots['team_id'] = np.where(
        shots['home_team_shot'],
        shots['home_team_id'],
        shots['away_team_id']
    )

    # Aggregate xG and shot counts per match_id, team_id, minute
    agg_shots = shots.groupby(['match_id', 'team_id', 'minute']).agg(
        xg=('xg_shot', 'sum'),
        shots=('xg_shot', 'count')
    ).reset_index()

    # Add is_home and is_away from the base table
    agg_shots = agg_shots.merge(
        base[['match_id', 'team_id', 'is_home', 'is_away']].drop_duplicates(),
        on=['match_id', 'team_id'],
        how='left'
    )

    # Remove unmatched shots
    agg_shots = agg_shots.dropna(subset=['is_home', 'is_away'])

    # Separate home and away shot data
    home_shots = agg_shots[agg_shots['is_home']][['match_id', 'minute', 'xg', 'shots']].rename(
        columns={'xg': 'home_xg', 'shots': 'home_shots'}
    )
    away_shots = agg_shots[agg_shots['is_away']][['match_id', 'minute', 'xg', 'shots']].rename(
        columns={'xg': 'away_xg', 'shots': 'away_shots'}
    )

    # Merge into final game state timeline
    final = final.merge(home_shots, on=['match_id', 'minute'], how='left')
    final = final.merge(away_shots, on=['match_id', 'minute'], how='left')

    # Fill NaNs with 0
    final[['home_xg', 'home_shots', 'away_xg', 'away_shots']] = final[[
        'home_xg', 'home_shots', 'away_xg', 'away_shots'
    ]].fillna(0)

    return final

