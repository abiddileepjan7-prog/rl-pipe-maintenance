# RL Pipe Maintenance

Reinforcement learning applied to a pipe thickness-loss / corrosion dataset to learn
**optimal maintenance decisions** — when to repair, monitor, or leave a pipe alone.

Two classic tabular RL algorithms, **Q-Learning** and **SARSA**, are trained on the
same environment and compared.

## Problem Framing

The raw dataset is static (one row per pipe with corrosion/condition features), so it's
reframed as a **sequential decision-making problem**:

- **State** (9 discrete states): `Condition level (Normal / Moderate / Critical)` ×
  `Thickness-loss bin (Low / Medium / High)`
- **Actions** (3):
  - `0` — Do Nothing
  - `1` — Monitor / Inspect
  - `2` — Repair / Replace
- **Reward**: designed to balance **safety vs. cost** — e.g. a large penalty for
  ignoring a Critical pipe, a reward for repairing it in time, and a small penalty for
  wasting money repairing a Normal pipe.

Each episode samples a random batch of pipes from the dataset; the agent picks an
action for each and receives a reward based on the pipe's true condition.

## Algorithms Compared

| Algorithm  | Type        |
|------------|-------------|
| Q-Learning | Off-policy  |
| SARSA      | On-policy   |

Both are trained with the same hyperparameters (learning rate, discount factor,
epsilon-greedy exploration schedule) so the comparison is fair.

## Results

The script outputs:
1. **Learning curves** — total reward per episode (raw + smoothed) for both algorithms
2. **Final learned policy** — best action per state, side-by-side for each algorithm

![RL Comparison](rl_comparison.png)

Both algorithms converge to a sensible policy (repair Critical pipes, monitor Moderate
ones, leave Normal pipes alone), with Q-Learning slightly outperforming SARSA in
average reward.

## Project Structure

```
rl-pipe-maintenance/
├── rl_pipe_maintenance.py       # Main script: env, training, plotting
├── pipe_thickness_loss_dataset.csv
├── rl_comparison.png            # Output plot
└── README.md
```

## Requirements

```
numpy
pandas
matplotlib
```

Install with:
```bash
pip install numpy pandas matplotlib
```

## Usage

```bash
python rl_pipe_maintenance.py
```

This trains both agents and saves `rl_comparison.png` with the comparison plots.

## Dataset

`pipe_thickness_loss_dataset.csv` — 1000 pipes with features including pipe size,
diameter, thickness, material, strength, max pressure, corrosion impact %, thickness
loss, material loss %, age (years), temperature, and a condition label
(Normal / Moderate / Critical).

## Future Improvements

- Add a third algorithm (e.g. Expected SARSA or a simple DQN) for comparison
- Use continuous state features instead of discretized bins
- Incorporate actual maintenance cost data if available
- Add a proper train/test split to evaluate policy generalization

## License

MIT
