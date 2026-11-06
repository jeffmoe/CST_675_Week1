"""Training-curve plots for the Blackjack Q-Learning agent."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np

from q_learning import TrainingLog


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` episodes ('valid' mode: len(x) - window + 1 points)."""
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plot_training(log: TrainingLog, out_dir: str | Path, window: int = 10_000) -> list[Path]:
    """Save (a) cumulative reward, (b) steps per episode, (c) win/loss/draw moving averages.

    Returns the paths written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(log.episode_rewards)
    lengths = np.asarray(log.episode_lengths)
    outcomes = np.asarray(log.outcomes)
    epsilons = np.asarray(log.epsilons)
    episodes = np.arange(1, len(rewards) + 1)
    ma_x = episodes[window - 1 :] if len(rewards) >= window else episodes[-1:]
    paths: list[Path] = []

    def eps_axis(ax: plt.Axes) -> None:
        ax2 = ax.twinx()
        ax2.plot(episodes, epsilons, color="0.6", lw=1, ls="--", label="ε")
        ax2.set_ylabel("ε", color="0.4")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", colors="0.4")

    # (a) Cumulative reward vs. episode
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, np.cumsum(rewards), color="tab:blue", lw=1.5, label="cumulative reward")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward (sum of returns)")
    ax.set_title("(a) Cumulative reward vs. episode")
    eps_axis(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    paths.append(out / "a_cumulative_reward.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # (b) Steps vs. episode
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(episodes[:: max(1, len(episodes) // 5000)],
               lengths[:: max(1, len(episodes) // 5000)],
               s=2, alpha=0.15, color="tab:gray", label="steps (subsampled)")
    ax.plot(ma_x, moving_average(lengths, window), color="tab:orange", lw=2,
            label=f"moving average ({window:,} episodes)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode")
    ax.set_title("(b) Steps per episode")
    eps_axis(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    paths.append(out / "b_steps_per_episode.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # (c) Win-rate moving average (with loss and draw for context)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ma_x, moving_average(outcomes > 0, window), color="tab:green", lw=2, label="win rate")
    ax.plot(ma_x, moving_average(outcomes < 0, window), color="tab:red", lw=1.2, label="loss rate")
    ax.plot(ma_x, moving_average(outcomes == 0, window), color="tab:purple", lw=1.2, label="draw rate")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"rate (moving average, {window:,} episodes)")
    ax.set_ylim(0, 1)
    ax.set_title("(c) Win / loss / draw rate")
    eps_axis(ax)
    ax.legend(loc="center left")
    fig.tight_layout()
    paths.append(out / "c_win_rate.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    return paths
