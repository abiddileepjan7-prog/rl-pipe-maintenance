import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ----------------------------------------------------------------
# 1. Load & prepare data
# ----------------------------------------------------------------
df = pd.read_csv("pipe_thickness_loss_dataset.csv")

# Discretize the two key risk signals into simple bins -> defines the STATE
df["ThickLoss_bin"] = pd.cut(df["Thickness_Loss_mm"], bins=[-1, 3, 6, 100],
                              labels=[0, 1, 2]).astype(int)          # low/med/high loss
cond_map = {"Normal": 0, "Moderate": 1, "Critical": 2}
df["Cond_level"] = df["Condition"].map(cond_map)                     # risk level

# STATE = (condition_level, thickness_loss_bin)  -> 3 x 3 = 9 discrete states
df["state"] = df["Cond_level"] * 3 + df["ThickLoss_bin"]

N_STATES = 9
N_ACTIONS = 3   # 0=Do Nothing, 1=Monitor, 2=Repair

# ----------------------------------------------------------------
# 2. Reward function (domain logic: balance safety vs. cost)
# ----------------------------------------------------------------
def get_reward(cond_level, action):
    """cond_level: 0=Normal,1=Moderate,2=Critical"""
    if cond_level == 2:        # Critical pipe
        if action == 2:  return 10   # repaired in time -> big reward
        if action == 1:  return -5   # only monitored -> risky
        return -15                   # ignored -> failure risk, big penalty
    if cond_level == 1:        # Moderate pipe
        if action == 1:  return 5    # monitoring is the right call
        if action == 2:  return 2    # repaired early (safe but costs extra)
        return -4                    # ignored -> may worsen
    # Normal pipe
    if action == 0:  return 3        # correctly left alone
    if action == 1:  return 1        # unnecessary inspection cost
    return -3                        # unnecessary repair, wasted cost

# ----------------------------------------------------------------
# 3. Environment: one "episode" = a random shuffled batch of pipes
# ----------------------------------------------------------------
EPISODE_LEN = 40   # pipes seen per episode
N_EPISODES = 400
ALPHA, GAMMA, EPS_START, EPS_END = 0.1, 0.95, 1.0, 0.05

def eps_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[s]))

def sample_episode():
    rows = df.sample(EPISODE_LEN, replace=True).reset_index(drop=True)
    return rows["state"].values, rows["Cond_level"].values

# ----------------------------------------------------------------
# 4. Q-Learning
# ----------------------------------------------------------------
def train_qlearning():
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards_per_ep = []
    for ep in range(N_EPISODES):
        eps = EPS_START - (EPS_START - EPS_END) * (ep / N_EPISODES)
        states, conds = sample_episode()
        total_r = 0
        for t in range(len(states) - 1):
            s, s_next = states[t], states[t + 1]
            a = eps_greedy(Q, s, eps)
            r = get_reward(conds[t], a)
            Q[s, a] += ALPHA * (r + GAMMA * np.max(Q[s_next]) - Q[s, a])
            total_r += r
        rewards_per_ep.append(total_r)
    return Q, rewards_per_ep

# ----------------------------------------------------------------
# 5. SARSA
# ----------------------------------------------------------------
def train_sarsa():
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards_per_ep = []
    for ep in range(N_EPISODES):
        eps = EPS_START - (EPS_START - EPS_END) * (ep / N_EPISODES)
        states, conds = sample_episode()
        total_r = 0
        a = eps_greedy(Q, states[0], eps)
        for t in range(len(states) - 1):
            s, s_next = states[t], states[t + 1]
            r = get_reward(conds[t], a)
            a_next = eps_greedy(Q, s_next, eps)
            Q[s, a] += ALPHA * (r + GAMMA * Q[s_next, a_next] - Q[s, a])
            total_r += r
            a = a_next
        rewards_per_ep.append(total_r)
    return Q, rewards_per_ep

# ----------------------------------------------------------------
# 6. Train both agents
# ----------------------------------------------------------------
Q_ql, rewards_ql = train_qlearning()
Q_sarsa, rewards_sarsa = train_sarsa()

def smooth(x, w=15):
    return pd.Series(x).rolling(w, min_periods=1).mean()

# ----------------------------------------------------------------
# 7. Plot comparison
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(rewards_ql, alpha=0.25, color="tab:blue")
axes[0].plot(smooth(rewards_ql), color="tab:blue", label="Q-Learning (smoothed)")
axes[0].plot(rewards_sarsa, alpha=0.25, color="tab:orange")
axes[0].plot(smooth(rewards_sarsa), color="tab:orange", label="SARSA (smoothed)")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward per episode")
axes[0].set_title("Learning Curves: Q-Learning vs SARSA")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Final policy comparison (bar chart of best action per state)
state_labels = [f"{c}/{t}" for c in ["Normal", "Moderate", "Critical"] for t in ["LowLoss", "MedLoss", "HighLoss"]]
policy_ql = np.argmax(Q_ql, axis=1)
policy_sarsa = np.argmax(Q_sarsa, axis=1)
x = np.arange(N_STATES)
width = 0.35
axes[1].bar(x - width/2, policy_ql, width, label="Q-Learning", color="tab:blue")
axes[1].bar(x + width/2, policy_sarsa, width, label="SARSA", color="tab:orange")
axes[1].set_xticks(x)
axes[1].set_xticklabels(state_labels, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(["Do Nothing", "Monitor", "Repair"])
axes[1].set_title("Learned Policy per State (final)")
axes[1].legend()
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.show()
print("Saved plot to /mnt/user-data/outputs/rl_comparison.png")

# ----------------------------------------------------------------
# 8. Print summary
# ----------------------------------------------------------------
print("\nAverage reward (last 50 episodes):")
print(f"  Q-Learning : {np.mean(rewards_ql[-50:]):.2f}")
print(f"  SARSA      : {np.mean(rewards_sarsa[-50:]):.2f}")