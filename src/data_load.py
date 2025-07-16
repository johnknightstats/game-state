########################################################
### Script to get required data from sqlite database ###
########################################################

from pathlib import Path
import sqlite3
import pandas as pd
from contextlib import contextmanager

# Define the database path
project_root = Path(__file__).resolve().parents[2]
DB_PATH = Path("C:/Users/JohnK/unsynced_footballdb1/footballdb1.sqlite")

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        conn.commit()
        cursor.close()
        conn.close()

def get_shots():
    query = """
        SELECT 
            M.match_date, 
            C.competition_name, 
            M.home_team_id, 
            HT.team_name AS home_team, 
            M.away_team_id, 
            AT.team_name AS away_team, 
            S.*,
            CASE WHEN S.team_id = HT.team_id THEN 1 ELSE 0 END AS home_team_shot
        FROM Shot S
        JOIN "Match" M ON M.match_id = S.match_id
        JOIN Team HT ON HT.team_id = M.home_team_id
        JOIN Team AT ON AT.team_id = M.away_team_id
        JOIN Competition C ON C.competition_id = M.competition_id
        WHERE M.competition_id IN (9, 11, 12, 13, 20);
    """

    with get_db_connection() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    return df

def get_cards():
    query = """
        SELECT 
            M.match_date, 
            C.competition_name, 
            HT.team_name AS home_team, 
            M.home_team_id, 
            AT.team_name AS away_team, 
            M.away_team_id,
            Card.*, 
            CASE WHEN Card.team_id = HT.team_id THEN 1 ELSE 0 END AS home_team_card
        FROM Card
        JOIN "Match" M ON M.match_id = Card.match_id
        JOIN Team HT ON HT.team_id = M.home_team_id
        JOIN Team AT ON AT.team_id = M.away_team_id
        JOIN Competition C ON C.competition_id = M.competition_id
        WHERE M.competition_id IN (9, 11, 12, 13, 20);
    """

    with get_db_connection() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    return df

def get_goals():
    query = """
        SELECT 
            M.match_date, 
            C.competition_name, 
            HT.team_name AS home_team, 
            M.home_team_id, 
            AT.team_name AS away_team, 
            M.away_team_id,
            Goal.*, 
            CASE WHEN Goal.team_id = HT.team_id THEN 1 ELSE 0 END AS home_team_goal
        FROM Goal
        JOIN "Match" M ON M.match_id = Goal.match_id
        JOIN Team HT ON HT.team_id = M.home_team_id
        JOIN Team AT ON AT.team_id = M.away_team_id
        JOIN Competition C ON C.competition_id = M.competition_id
        WHERE M.competition_id IN (9, 11, 12, 13, 20);
    """

    with get_db_connection() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    return df

def get_xwalk():
    query = "SELECT * FROM Team_Crosswalk"

    with get_db_connection() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    return df

def get_odds():
    query = """
        SELECT * FROM Football_Data
        WHERE league IN ('Serie A', 'Premier League', 'Bundesliga', 'Primera Division', 'Ligue 1')
        AND country IN ('Italy', 'England', 'Germany', 'Spain', 'France')
        AND season IN ('2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025')
    """

    with get_db_connection() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

    return df


