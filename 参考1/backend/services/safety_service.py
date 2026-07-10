from __future__ import annotations


class SafetyService:
    """Rule-based quality scoring used by JudgeAgent before real judging exists."""

    def score_content(self, content: str, required_terms: list[str] | None = None) -> tuple[int, list[str]]:
        score = 92
        suggestions: list[str] = []
        text = content.strip()

        if len(text) < 80:
            score -= 18
            suggestions.append("Add more concrete explanation and examples.")
        if required_terms:
            missing = [term for term in required_terms if term.lower() not in text.lower()]
            if missing:
                score -= min(20, 5 * len(missing))
                suggestions.append("Mention key terms: " + ", ".join(missing[:4]))
        if "Mock Spark" not in text and len(text) < 240:
            suggestions.append("Consider adding a short learning objective and checkpoint.")

        return max(score, 0), suggestions
