"""Evaluate the trained Q-table greedily (ε = 0) and compare it with baseline policies.

Usage:
    python train.py          # writes q_table.npy
    python evaluate.py       # evaluates it; writes plots/d_policy_comparison.png
    python evaluate.py --q-table q_table.npy --n-episodes 200000 --seed 12345

Policies compared on the same number of hands:
    q_learning    greedy w.r.t. the learned Q-table
    sutton_barto  optimal policy from Sutton & Barto (2018, Fig. 5.2) — the target
    dealer_rule   hit below 17, stick on 17+ (mimic the dealer)
    always_stick  never take a card
    random        hit or stick with probability 1/2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent import DEALER_CARDS, HIT, PLAYER_SUMS, STICK, Obs, QLearningAgent, format_policy

Policy = Callable[[Obs], int]


def sutton_barto_policy_table() -> np.ndarray:
    """Optimal Blackjack policy from Sutton & Barto (2018, Fig. 5.2) as a (32, 11, 2) array.

    No usable ace: stick on 17+; on 13–16 stick vs. dealer 2–6; on 12 stick vs. 4–6.
    Usable ace:    stick on 19+; on 18 stick vs. dealer 2–8.
    Everything else (including all sums ≤ 11): hit.
    """
    table = np.full((32, 11, 2), HIT, dtype=np.int64)
    for d in DEALER_CARDS:
        table[17:, d, 0] = STICK
        if 2 <= d <= 6:
            table[13:17, d, 0] = STICK
        if 4 <= d <= 6:
            table[12, d, 0] = STICK
        table[19:, d, 1] = STICK
        if 2 <= d <= 8:
            table[18, d, 1] = STICK
    return table


def table_policy(table: np.ndarray) -> Policy:
    return lambda obs: int(table[obs])


def evaluate(policy: Policy, n_episodes: int, seed: int) -> dict[str, float]:
    """Run `policy` for n_episodes on Blackjack-v1 (sab=True); no learning happens."""
    env = gym.make("Blackjack-v1", sab=True)
    returns = np.empty(n_episodes)
    steps = np.empty(n_episodes)
    for i in range(n_episodes):
        obs, _info = env.reset(seed=seed if i == 0 else None)
        total, n = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _info = env.step(policy(obs))
            total += reward
            n += 1
            if terminated or truncated:
                break
        returns[i], steps[i] = total, n
    env.close()
    return {
        "mean": returns.mean(),
        "ci95": 1.96 * returns.std(ddof=1) / np.sqrt(n_episodes),
        "win": np.mean(returns > 0),
        "loss": np.mean(returns < 0),
        "draw": np.mean(returns == 0),
        "steps": steps.mean(),
    }


def decision_mask() -> np.ndarray:
    """True on the 200 decision states of Fig. 5.2 (player 12–21 × dealer 1–10 × ace 0/1)."""
    mask = np.zeros((32, 11, 2), dtype=bool)
    mask[12:22, 1:11, :] = True
    return mask


def plot_policy_comparison(learned: np.ndarray, optimal: np.ndarray, path: Path) -> Path:
    """Side-by-side heatmaps: learned vs. Sutton & Barto, with mismatches outlined."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    cmap = matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452"])  # stick, hit
    dealer_labels = ["A"] + [str(d) for d in range(2, 11)]
    sums = list(PLAYER_SUMS)

    for row, ace in enumerate((0, 1)):
        for col, (name, table) in enumerate((("Q-Learning (greedy)", learned),
                                             ("Sutton & Barto Fig. 5.2", optimal))):
            ax = axes[row, col]
            grid = table[12:22, 1:11, ace]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
            ax.set_xticks(range(10), dealer_labels)
            ax.set_yticks(range(10), sums)
            ax.set_xlabel("dealer showing")
            ax.set_ylabel("player sum")
            ax.set_title(f"{name}\n{'usable ace' if ace else 'no usable ace'}", fontsize=10)
            for i in range(10):
                for j in range(10):
                    ax.text(j, i, "H" if grid[i, j] == HIT else "S", ha="center",
                            va="center", color="white", fontsize=8)
                    if col == 0 and grid[i, j] != optimal[12 + i, 1 + j, ace]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                   edgecolor="black", lw=2))

    fig.suptitle("Greedy policy vs. optimal (H = hit, S = stick; outlined = mismatch)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy evaluation and baseline comparison.")
    parser.add_argument("--q-table", type=str, default="q_table.npy")
    parser.add_argument("--n-episodes", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=12345, help="differs from training seed")
    parser.add_argument("--plot-dir", type=str, default="plots")
    args = parser.parse_args()

    agent = QLearningAgent.load(args.q_table)
    learned = agent.policy_table()
    optimal = sutton_barto_policy_table()
    rng = np.random.default_rng(args.seed)

    policies: dict[str, Policy] = {
        "q_learning": table_policy(learned),
        "sutton_barto": table_policy(optimal),
        "dealer_rule": lambda obs: HIT if obs[0] < 17 else STICK,
        "always_stick": lambda obs: STICK,
        "random": lambda obs: int(rng.integers(2)),
    }

    print(f"Greedy evaluation: {args.n_episodes:,} hands per policy, seed {args.seed}, epsilon = 0\n")
    print(f"{'policy':<14}{'mean reward':>18}{'win':>8}{'loss':>8}{'draw':>8}{'steps':>8}")
    for name, policy in policies.items():
        r = evaluate(policy, args.n_episodes, args.seed)
        print(
            f"{name:<14}{r['mean']:>+10.4f} +/- {r['ci95']:.4f}"
            f"{r['win']:>8.3f}{r['loss']:>8.3f}{r['draw']:>8.3f}{r['steps']:>8.3f}"
        )

    mask = decision_mask()
    mismatch = (learned != optimal) & mask
    agree = mask.sum() - mismatch.sum()
    print(f"\nPolicy agreement with Sutton & Barto: {agree} / {mask.sum()} "
          f"({100 * agree / mask.sum():.1f}%)  [brackets = mismatch]")
    for ace in (0, 1):
        print(f"\nLearned greedy policy, {'usable ace' if ace else 'no usable ace'}")
        print(format_policy(learned, ace, mark=mismatch))

    path = plot_policy_comparison(learned, optimal, Path(args.plot_dir) / "d_policy_comparison.png")
    print(f"\nsaved plot: {path}")


if __name__ == "__main__":
    main()
