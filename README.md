# game-state

This project analyzes how the probability of scoring a goal varies throughout a football match depending on the **game state** — including minutes elapsed, current score, and other situational factors. Using logistic regression, the project quantifies minute-by-minute goal likelihood under different match conditions.

📝 **Read the article**:  
📘 [How does scoring rate change in a football match based on time elapsed and game state?](https://johnknightstats.com/posts/game_state/)

---

## 🧠 Key Features

- Loads match data and aggregates events minute-by-minute
- Fits logistic regression models to assess goal probability under varying scorelines and odds
- Generates visualizations showing trends over time and by game state
- Outputs ready-to-use charts and regression results

---

## 📁 Folder Structure

```
├── exe/ # Pipeline runner and output files (ignored from versioning)
├── src/ # Core scripts and visualizations
│ ├── inspect_data.py # Exploratory plots and goal patterns
│ ├── regressions.py # Logistic regression models
├── viz/ # Final figures and regression outputs
├── testing/ # Development test data (ignored)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## ⚠️ Data Access Note

This repo does **not** include raw football match data, in accordance with data licensing restrictions.  
To replicate this analysis, you’ll need to acquire your own event-level match data (many tutorials and repos exist for this purpose).

Scripts for extracting and preparing data from my SQL database (`run_pipeline.py`, `data_load.py`, `process_data.py`) are **not included**.

---

## 🛠 Requirements

Install dependencies with:

`pip install -r requirements.txt`

📄 License

This project is licensed under the MIT License. See LICENSE for details.

✍️ Author

John Knight
https://johnknightstats.com