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


@dataclass
class TrainingLog:
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


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
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        done, total_reward, length = False, 0.0, 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            log.td_errors.append(agent.update(obs, action, reward, next_obs, terminated))
            obs = next_obs
            total_reward += reward
            length += 1
            done = terminated or truncated

        log.episode_rewards.append(total_reward)
        log.episode_lengths.append(length)
        log.epsilons.append(agent.epsilon)
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

    rewards = np.asarray(log.episode_rewards)
    window = min(10_000, len(rewards))
    print(f"episodes: {len(rewards):,}  final epsilon: {agent.epsilon:.3f}")
    print(f"mean reward, first {window:,}: {rewards[:window].mean():+.4f}")
    print(f"mean reward, last  {window:,}: {rewards[-window:].mean():+.4f}")
    print(f"Q[(15, 10, 0)] = stick {agent.q[15, 10, 0, STICK]:+.4f}, hit {agent.q[15, 10, 0, HIT]:+.4f}")

    if args.save:
        np.save(args.save, agent.q)
        print(f"saved Q-table to {args.save}")


if __name__ == "__main__":
    main()
