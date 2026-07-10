from __future__ import annotations

from backend.services.bkt_service import BKTService
from backend.schemas.evaluation import EvaluationResult


class EvaluationAgent:
    """Updates mastery estimates from assessment outcomes."""

    def __init__(self) -> None:
        self.bkt = BKTService()

    def update_mastery(self, prior: float, is_correct: bool, score: float) -> float:
        return self.bkt.update(prior=prior, is_correct=is_correct, score=score)

    def evaluate_quiz(
        self,
        answers: dict[str, int],
        answer_key: dict[str, dict],
        mastery_map: dict[str, float] | None = None,
    ) -> EvaluationResult:
        mastery_map = dict(mastery_map or {})
        total = len(answer_key)
        correct_count = 0
        weak_concepts: list[str] = []

        for question_id, meta in answer_key.items():
            concept = str(meta.get("concept", question_id))
            expected = int(meta.get("answer_index", 0))
            actual = answers.get(question_id)
            is_correct = actual == expected
            if is_correct:
                correct_count += 1
            else:
                weak_concepts.append(concept)
            prior = float(mastery_map.get(concept, 0.45))
            mastery_map[concept] = round(self.update_mastery(prior, is_correct, 1.0 if is_correct else 0.2), 2)

        score = int(round((correct_count / total) * 100)) if total else 0
        suggestions = (
            [f"增加 {concept} 针对性练习" for concept in weak_concepts]
            + (["推迟 Agent 编排学习，先完成 FAISS 检索 CodeLab"] if weak_concepts else ["可以提前进入 LangChain LCEL 与 Agent 编排项目任务"])
        )
        return EvaluationResult(
            score=score,
            correct_count=correct_count,
            total_count=total,
            weak_concepts=list(dict.fromkeys(weak_concepts)),
            updated_mastery_map=mastery_map,
            replanning_suggestions=suggestions,
        )
