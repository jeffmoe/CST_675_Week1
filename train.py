"""Train the tabular Q-Learning agent on gym.make("Blackjack-v1", sab=True).

Usage:
    python train.py                       # 500k episodes, saves q_table.npy and plots/
    python train.py --n-episodes 1000000 --alpha 0.005 --gamma 1.0 \
        --epsilon-start 1.0 --epsilon-final 0.05 --epsilon-decay-fraction 0.5 --seed 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np

from agent import HIT, STICK, QLearningAgent, format_policy


@dataclass
class TrainingLog:
    """Per-episode training statistics (one entry per episode unless noted)."""

    episode_rewards: list[float] = field(default_factory=list)  # return G of the episode
    episode_lengths: list[int] = field(default_factory=list)  # number of env.step() calls
    epsilons: list[float] = field(default_factory=list)  # ε used during the episode
    outcomes: list[int] = field(default_factory=list)  # +1 win, 0 draw, -1 loss
    td_errors: list[float] = field(default_factory=list)  # one entry per step
    wins: int = 0
    losses: int = 0
    draws: int = 0
    terminated_episodes: int = 0  # ended in a true terminal state (stick or bust)
    truncated_episodes: int = 0  # cut off by a time limit (never happens in Blackjack-v1)

    def record(self, total_reward: float, steps: int, epsilon: float, terminated: bool) -> None:
        outcome = int(np.sign(total_reward))
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilons.append(epsilon)
        self.outcomes.append(outcome)
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1
        if terminated:
            self.terminated_episodes += 1
        else:
            self.truncated_episodes += 1


def train(
    n_episodes: int = 500_000,
    alpha: float = 0.01,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
    epsilon_decay_fraction: float = 0.5,
    seed: int | None = 0,
) -> tuple[QLearningAgent, TrainingLog]:
    """Train an agent on Blackjack-v1 (sab=True).

    epsilon_decay_fraction: the fraction of n_episodes over which ε falls linearly from
    epsilon_start to epsilon_final. It then stays at epsilon_final.
    """
    decay_episodes = max(1, int(n_episodes * epsilon_decay_fraction))
    epsilon_decay = (epsilon_start - epsilon_final) / decay_episodes

    agent = QLearningAgent(alpha, gamma, epsilon_start, epsilon_final, epsilon_decay, seed)
    env = gym.make("Blackjack-v1", sab=True)
    log = TrainingLog()

    for episode in range(n_episodes):
        # Seed only the first reset; later resets continue the same RNG stream.
        obs, _info = env.reset(seed=seed if episode == 0 else None)
        total_reward, steps = 0.0, 0

        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            # Pass `terminated`, not `terminated or truncated`: only a true terminal
            # state has value 0. A truncated step still bootstraps from next_obs.
            log.td_errors.append(agent.update(obs, action, reward, next_obs, terminated))
            obs = next_obs
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        log.record(total_reward, steps, agent.epsilon, terminated)
        agent.decay_epsilon()

    env.close()
    return agent, log


# --------------------------------------------------------------------------- plots


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` episodes ('valid' mode: len(x) - window + 1 points)."""
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plot_training(log: TrainingLog, out_dir: str | Path, window: int = 10_000) -> list[Path]:
    """Save (a) cumulative reward, (b) steps per episode, (c) win/loss/draw moving averages."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(log.episode_rewards)
    lengths = np.asarray(log.episode_lengths)
    outcomes = np.asarray(log.outcomes)
    epsilons = np.asarray(log.epsilons)
    episodes = np.arange(1, len(rewards) + 1)
    window = max(1, min(window, len(rewards)))
    ma_x = episodes[window - 1 :]
    stride = max(1, len(episodes) // 5000)
    paths: list[Path] = []

    def eps_axis(ax: plt.Axes) -> None:
        ax2 = ax.twinx()
        ax2.plot(episodes, epsilons, color="0.6", lw=1, ls="--")
        ax2.set_ylabel("ε (dashed)", color="0.4")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", colors="0.4")

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        paths.append(out / name)
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    # (a) Cumulative reward vs. episode
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, np.cumsum(rewards), color="tab:blue", lw=1.5, label="cumulative reward")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward (sum of returns)")
    ax.set_title("(a) Cumulative reward vs. episode")
    eps_axis(ax)
    ax.legend(loc="lower left")
    save(fig, "a_cumulative_reward.png")

    # (b) Steps vs. episode
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(episodes[::stride], lengths[::stride], s=2, alpha=0.15, color="tab:gray",
               label="steps (subsampled)")
    ax.plot(ma_x, moving_average(lengths, window), color="tab:orange", lw=2,
            label=f"moving average ({window:,} episodes)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode")
    ax.set_title("(b) Steps per episode")
    eps_axis(ax)
    ax.legend(loc="upper left")
    save(fig, "b_steps_per_episode.png")

    # (c) Win-rate moving average (with loss and draw for context)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ma_x, moving_average(outcomes > 0, window), color="tab:green", lw=2, label="win rate")
    ax.plot(ma_x, moving_average(outcomes < 0, window), color="tab:red", lw=1.2, label="loss rate")
    ax.plot(ma_x, moving_average(outcomes == 0, window), color="tab:purple", lw=1.2,
            label="draw rate")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"rate (moving average, {window:,} episodes)")
    ax.set_ylim(0, 1)
    ax.set_title("(c) Win / loss / draw rate")
    eps_axis(ax)
    ax.legend(loc="center left")
    save(fig, "c_win_rate.png")

    return paths


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular Q-Learning on Blackjack-v1.")
    parser.add_argument("--n-episodes", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=0.01, help="learning rate α")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor γ")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="q_table.npy", help="Q-table output path")
    parser.add_argument("--plot-dir", type=str, default="plots", help="learning-curve directory")
    parser.add_argument("--window", type=int, default=10_000, help="moving-average window")
    args = parser.parse_args()

    agent, log = train(
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_fraction=args.epsilon_decay_fraction,
        seed=args.seed,
    )

    n = len(log.episode_rewards)
    rewards = np.asarray(log.episode_rewards)
    outcomes = np.asarray(log.outcomes)
    lengths = np.asarray(log.episode_lengths)
    window = min(args.window, n)
    print(f"episodes: {n:,}  final epsilon: {agent.epsilon:.3f}")
    print(f"terminated: {log.terminated_episodes:,}  truncated: {log.truncated_episodes:,}")
    print(
        f"overall  W/L/D: {log.wins:,} / {log.losses:,} / {log.draws:,}  "
        f"(win {log.wins / n:.3f}, loss {log.losses / n:.3f}, draw {log.draws / n:.3f})"
    )
    for label, sl in (("first", slice(None, window)), ("last ", slice(-window, None))):
        o = outcomes[sl]
        print(
            f"{label} {window:,}: mean reward {rewards[sl].mean():+.4f}  "
            f"win {np.mean(o > 0):.3f}  loss {np.mean(o < 0):.3f}  draw {np.mean(o == 0):.3f}  "
            f"mean steps {lengths[sl].mean():.3f}"
        )
    print(
        f"Q[(15, 10, 0)] = stick {agent.q[15, 10, 0, STICK]:+.4f}, "
        f"hit {agent.q[15, 10, 0, HIT]:+.4f}"
    )

    policy = agent.policy_table()
    print("\nGreedy policy, no usable ace (rows: player sum, cols: dealer card)")
    print(format_policy(policy, usable_ace=0))
    print("\nGreedy policy, usable ace")
    print(format_policy(policy, usable_ace=1))

    agent.save(args.save)
    print(f"\nsaved Q-table to {args.save}")
    for path in plot_training(log, args.plot_dir, window=args.window):
        print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
