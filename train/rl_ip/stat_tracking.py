"""Per-prompt reward statistics tracking."""

import numpy as np


class PerPromptStatTracker:
    """Track reward statistics per prompt for advantage normalization."""

    def __init__(self):
        self.stats = {}

    def update(self, prompts, rewards) -> np.ndarray:
        """Update stats and return normalized advantages."""
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards)

        for prompt in np.unique(prompts):
            mask = prompts == prompt
            prompt_rewards = rewards[mask]

            # Update history
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())

            # Compute normalization
            history = np.array(self.stats[prompt])
            mean = history.mean()
            std = history.std() + 1e-4
            advantages[mask] = (prompt_rewards - mean) / std

        return advantages

    def clear(self):
        self.stats = {}


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)


if __name__ == "__main__":
    main()
