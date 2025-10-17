from dataclasses import dataclass
from typing import Dict, Any, List

from deepseek_handler import DeepSeekClient


SYSTEM_PROMPT = (
    "You are BestPracticeAdvisorAgent. Suggest improvements and modern practices."
    " Cite short references (e.g., PEP8, SOLID, language-specific guides)."
    " Return JSON with keys: suggestions (array of {suggestion, rationale, references: array<string>}),"
    " reasoning (short), confidence (0-1)."
)


@dataclass
class BestPracticeResult:
    suggestions: Any
    reasoning: str
    confidence: float


class BestPracticeAdvisorAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def run(self, code_text: str, filename: str, standards_text: str) -> Dict[str, Any]:
        user_prompt = (
            "Given the code and standards, propose best-practice improvements.\n\n"
            f"Filename: {filename}\n\n"
            "Standards and references (optional):\n" + standards_text[:8000] + "\n\n"
            "Code:\n" + code_text[:12000] + "\n\n"
            "Return strictly JSON with keys: suggestions (array of {suggestion, rationale, references}), reasoning, confidence."
        )
        resp = self.client.chat_complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            response_format="json",
        )
        data: Dict[str, Any] = resp.parsed_json or {
            "suggestions": [],
            "reasoning": "",
            "confidence": 0.75,
        }
        data.setdefault("suggestions", [])
        data.setdefault("reasoning", "")
        data.setdefault("confidence", 0.75)
        # Normalize references
        normalized = []
        for s in data.get("suggestions", []):
            suggestion = s.get("suggestion") or s.get("advice") or "Improvement suggestion"
            rationale = s.get("rationale") or s.get("why") or ""
            references = s.get("references")
            if not isinstance(references, list):
                references = [rationale[:60]] if rationale else []
            normalized.append(
                {
                    "suggestion": suggestion,
                    "rationale": rationale,
                    "references": references,
                }
            )
        data["suggestions"] = normalized
        return data
