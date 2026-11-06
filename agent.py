"""Tabular Q-Learning agent for Gymnasium Blackjack-v1.

The Q-table is a dense NumPy array of shape (32, 11, 2, 2), indexed as
q[player_sum, dealer_card, usable_ace, action]. It matches the observation space
Tuple(Discrete(32), Discrete(11), Discrete(2)) and the Discrete(2) action space.

References:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction
        (2nd ed.), §6.5 (Q-learning), Example 5.1 and Fig. 5.2 (Blackjack).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Q_SHAPE = (32, 11, 2, 2)
STICK, HIT = 0, 1
PLAYER_SUMS = range(12, 22)  # decision states shown in Sutton & Barto Fig. 5.2
DEALER_CARDS = range(1, 11)  # 1 = ace

Obs = tuple[int, int, int]


class QLearningAgent:
    """Off-policy TD control (Sutton & Barto, 2018, §6.5) with an ε-greedy behavior policy."""

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

    def save(self, path: str | Path) -> None:
        np.save(path, self.q)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> QLearningAgent:
        agent = cls(**kwargs)
        q = np.load(path)
        if q.shape != Q_SHAPE:
            raise ValueError(f"expected Q-table of shape {Q_SHAPE}, got {q.shape}")
        agent.q = q
        return agent


def format_policy(policy: np.ndarray, usable_ace: int, mark: np.ndarray | None = None) -> str:
    """Text grid of a (32, 11, 2) policy: rows = player sum 21..12, cols = dealer A..10.

    H = hit, S = stick (the layout of Sutton & Barto, 2018, Fig. 5.2). Cells where
    `mark` is True are wrapped in brackets.
    """
    header = "      " + "".join(f"{'A' if d == 1 else d:>4}" for d in DEALER_CARDS)
    rows = [header]
    for player_sum in reversed(PLAYER_SUMS):
        cells = []
        for d in DEALER_CARDS:
            c = "H" if policy[player_sum, d, usable_ace] == HIT else "S"
            flagged = mark is not None and mark[player_sum, d, usable_ace]
            cells.append(f"{'[' + c + ']' if flagged else c:>4}")
        rows.append(f"  {player_sum:>2}  " + "".join(cells))
    return "\n".join(rows)
