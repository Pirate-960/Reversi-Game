# Reversi-Game
Reversi is a strategy board game for two players, played on an 8×8 uncheckered board. It was invented in 1883. Othello, a variant with a fixed initial setup of the board, was patented in 1971.


```
# Othello Game and AI Performance Testing

This repository contains the implementation of the classic Othello (Reversi) game, along with AI players using heuristic evaluation and a performance testing framework.

## Features

- **Game Modes**:
  - Human vs Human
  - Human vs AI
  - AI vs AI
- **AI Player**:
  - Minimax with alpha-beta pruning
  - Multiple heuristics for evaluation:
    - Disc count
    - Positional strategy
    - Mobility and stability
- **Performance Testing**:
  - Evaluate AI performance across heuristics and depths
  - Save results for analysis

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Required Libraries
The following libraries are required for this project:
- `numpy`
- `opencv-python`
- `pyautogui`
- `pygetwindow`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

Alternatively, manually install:
```bash
pip install numpy opencv-python pyautogui pygetwindow
```

---

## Running the Game

### Main Script
To start the game, run:
```bash
python Othello_v1.py
```

Follow the prompts to select the game mode, AI settings, and make moves.

---

## AI Performance Testing

### Test Script
To run AI performance tests:
```bash
python test_v1.py
```

This script will:
1. Test all AI heuristics across depths.
2. Save results to a CSV file (e.g., `ai_performance_test_<timestamp>.csv`).
3. Analyze and display performance metrics.

### Analyzing Results
The test script automatically analyzes the most recent results. To manually analyze a specific file:
1. Place the CSV file in the project directory.
2. Modify the `filename` parameter in the `analyze_results` function inside `test_v1.py`.

---

## File Descriptions

- **`Othello_v1.py`**:
  - Contains the game implementation and AI logic.
- **`test_v1.py`**:
  - Script for testing and analyzing AI performance.

---

## Notes

- Ensure you have a compatible Python version and dependencies installed before running the scripts.
- Performance testing may take longer for higher depths. Adjust the depth range in `test_v1.py` if needed.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
