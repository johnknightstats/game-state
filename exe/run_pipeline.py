
#####################################################################
### Pipeline to get required data from SQL database, process, and ###
### create a csv containing game states by minute                 ###
#####################################################################

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_load import *
from src.process_data import *

def main():

    print("Loading data...")
    shots = get_shots()
    cards = get_cards()
    xwalk = get_xwalk()
    odds = get_odds()
    goals = get_goals()

    print("Processing odds...")
    odds_processed = clean_and_merge_odds(odds, xwalk)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("Saving outputs...")
    shots.to_csv(output_dir / "shots.csv", index=False)
    cards.to_csv(output_dir / "cards.csv", index=False)
    goals.to_csv(output_dir / "goals.csv", index=False)

    print("Finding unmatched matches...")
    unmatched_shots, unmatched_odds = find_unmatched_matches(shots, odds_processed)

    print(f"Rows in shots not matched in odds: {len(unmatched_shots)}")
    print(f"Rows in odds not matched in shots: {len(unmatched_odds)}")

    unmatched_shots.to_csv(output_dir / "unmatched_from_shots.csv", index=False)
    unmatched_odds.to_csv(output_dir / "unmatched_from_odds.csv", index=False)

    odds_cleaned = remove_unmatched_odds(odds_processed, unmatched_odds)

    odds_with_match_id = add_match_id_to_odds(odds_cleaned, shots)
    odds_with_match_id = odds_to_probs(odds_with_match_id)
    odds_with_match_id.to_csv(output_dir / "matches.csv", index=False)

    print("Generating game state timeline...")
    game_state = generate_game_state_timeline(odds_with_match_id, goals, cards, shots)
    game_state.to_csv(output_dir / "game_state_timeline.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()
