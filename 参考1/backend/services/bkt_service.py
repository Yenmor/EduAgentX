from __future__ import annotations


class BKTService:
    """Tiny Bayesian Knowledge Tracing implementation for the MVP."""

    def __init__(self, learn_rate: float = 0.12, guess: float = 0.2, slip: float = 0.1) -> None:
        self.learn_rate = learn_rate
        self.guess = guess
        self.slip = slip

    def update(self, prior: float, is_correct: bool, score: float = 1.0) -> float:
        prior = min(max(prior, 0.01), 0.99)
        score = min(max(score, 0.0), 1.0)
        effective_correct = is_correct and score >= 0.5

        if effective_correct:
            numerator = prior * (1.0 - self.slip)
            denominator = numerator + (1.0 - prior) * self.guess
        else:
            numerator = prior * self.slip
            denominator = numerator + (1.0 - prior) * (1.0 - self.guess)

        posterior = numerator / denominator if denominator else prior
        learn_boost = self.learn_rate * (0.5 + score / 2.0)
        updated = posterior + (1.0 - posterior) * learn_boost
        return round(min(max(updated, 0.01), 0.99), 4)
