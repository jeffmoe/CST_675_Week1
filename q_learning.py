"""Tabular Q-Learning agent for Gymnasium Blackjack-v1.

Usage:
    python q_learning.py --n-episodes 500000 --alpha 0.01 --gamma 1.0 \
        --epsilon-start 1.0 --epsilon-final 0.05 --epsilon-decay-fraction 0.5

References:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction
        (2nd ed.), Section 6.5 (Q-learning).
    Farama Foundation. gymnasium/envs/toy_text/blackjack.py (v1.4.0).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

# Observation space: Tuple(Discrete(32), Discrete(11), Discrete(2)); actions: Discrete(2).
Q_SHAPE = (32, 11, 2, 2)
STICK, HIT = 0, 1

Obs = tuple[int, int, int]


class QLearningAgent:
    """Off-policy TD control (Sutton & Barto, 2018, §6.5) with an ε-greedy behavior policy.

    The Q-table is a dense NumPy array indexed as q[player_sum, dealer_card, usable_ace, action].
    """

    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= epsilon_final <= epsilon_start <= 1.0:
            raise ValueError("need 0 <= epsilon_final <= epsilon_start <= 1")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(Q_SHAPE, dtype=np.float64)

    def greedy_action(self, obs: Obs) -> int:
        """argmax_a Q(obs, a), breaking ties uniformly at random.

        Random tie-breaking matters at the start: every Q-value is 0, and np.argmax
        would always return 0 (stick), so the agent would never try hitting.
        """
        q_values = self.q[obs]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def act(self, obs: Obs) -> int:
        """ε-greedy behavior policy: explore with probability ε, otherwise act greedily."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(Q_SHAPE[-1]))
        return self.greedy_action(obs)

    def update(
        self, obs: Obs, action: int, reward: float, next_obs: Obs, terminated: bool
    ) -> float:
        """Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ].

        When terminated is True, s' is a terminal state with value 0, so the target is
        just r. This matters in Blackjack: after a stick, next_obs is the same tuple as
        obs, so bootstrapping would feed Q(s, ·) back into its own target. A truncated
        (time-limited) step would still bootstrap, but Blackjack-v1 never truncates.

        Returns the TD error δ.
        """
        future = 0.0 if terminated else self.gamma * float(np.max(self.q[next_obs]))
        td_target = reward + future
        td_error = td_target - self.q[obs][action]
        self.q[obs][action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        """Linear decay per episode, floored at epsilon_final."""
        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)

    def policy_table(self) -> np.ndarray:
        """Greedy policy as a (32, 11, 2) array of actions (ties resolve to stick)."""
        return np.argmax(self.q, axis=-1)


def format_policy(policy: np.ndarray, usable_ace: int) -> str:
    """Text grid of the greedy policy: rows = player sum 21..12, cols = dealer A..10.

    H = hit, S = stick (the layout of Sutton & Barto, 2018, Fig. 5.2).
    """
    header = "      " + " ".join(f"{'A' if d == 1 else d:>2}" for d in range(1, 11))
    rows = [header]
    for player_sum in range(21, 11, -1):
        cells = " ".join(
            f"{'H' if policy[player_sum, d, usable_ace] == HIT else 'S':>2}" for d in range(1, 11)
        )
        rows.append(f"  {player_sum:>2}  {cells}")
    return "\n".join(rows)


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
        obs, info = env.reset(seed=seed if episode == 0 else None)
        total_reward, steps = 0.0, 0

        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-episodes", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=0.01, help="learning rate α")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor γ")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="path to save the Q-table (.npy)")
    parser.add_argument("--plot-dir", type=str, default=None, help="directory for training plots")
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
    print(
        f"terminated: {log.terminated_episodes:,}  truncated: {log.truncated_episodes:,}"
    )
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
    print(f"Q[(15, 10, 0)] = stick {agent.q[15, 10, 0, STICK]:+.4f}, hit {agent.q[15, 10, 0, HIT]:+.4f}")

    policy = agent.policy_table()
    print("\nGreedy policy, no usable ace (rows: player sum, cols: dealer card)")
    print(format_policy(policy, usable_ace=0))
    print("\nGreedy policy, usable ace")
    print(format_policy(policy, usable_ace=1))

    if args.save:
        np.save(args.save, agent.q)
        print(f"\nsaved Q-table to {args.save}")

    if args.plot_dir:
        from plot_training import plot_training  # matplotlib is only needed for plots

        for path in plot_training(log, args.plot_dir, window=args.window):
            print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
