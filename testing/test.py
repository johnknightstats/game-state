
import sys
import os
import polars as pl

# Add src directory to the system path
sys.path.append(os.path.abspath('../src'))

from data_load import *

testing_dir = Path(__file__).resolve().parents[1] / "testing"

if __name__ == "__main__":

    shots = get_shots()
    shots.write_csv(testing_dir / "shots_data.csv")
    cards = get_cards()
    cards.write_csv(testing_dir / "cards_data.csv")
    xwalk = get_xwalk()
    xwalk.write_csv(testing_dir / "xwalk_data.csv")
