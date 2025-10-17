from dataclasses import dataclass
from typing import Dict, Any

from deepseek_handler import DeepSeekClient


SYSTEM_PROMPT = (
    "You are BugDetectionAgent. Identify likely runtime errors, logic bugs, and anti-patterns."
    " Focus on concrete, actionable issues with specific locations."
    " Return JSON with keys: bugs (array of {title, location, explanation, severity in [low, medium, high], confidence 0-1}),"
    " reasoning (short), confidence (0-1 overall)."
)


@dataclass
class BugDetectionResult:
    bugs: Any
    reasoning: str
    confidence: float


class BugDetectionAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def run(self, code_text: str, filename: str, standards_text: str) -> Dict[str, Any]:
        user_prompt = (
            "Scan the following code and identify likely runtime errors, logic issues, and anti-patterns.\n\n"
            f"Filename: {filename}\n\n"
            "Contextual standards (optional):\n" + standards_text[:8000] + "\n\n"
            "Code:\n" + code_text[:12000] + "\n\n"
            "Return strictly JSON with keys: bugs (array of {title, location, explanation, severity, confidence}), reasoning, confidence."
        )
        resp = self.client.chat_complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format="json",
        )
        data: Dict[str, Any] = resp.parsed_json or {
            "bugs": [],
            "reasoning": "",
            "confidence": 0.7,
        }
        data.setdefault("bugs", [])
        data.setdefault("reasoning", "")
        data.setdefault("confidence", 0.7)
        # Normalize bug entries
        normalized_bugs = []
        for bug in data.get("bugs", []):
            title = bug.get("title") or bug.get("issue") or "Potential issue"
            location = bug.get("location") or filename
            explanation = bug.get("explanation") or bug.get("why") or ""
            severity = (bug.get("severity") or "medium").lower()
            if severity not in {"low", "medium", "high"}:
                severity = "medium"
            confidence = bug.get("confidence")
            try:
                confidence_val = float(confidence) if confidence is not None else 0.7
            except Exception:
                confidence_val = 0.7
            normalized_bugs.append(
                {
                    "title": title,
                    "location": location,
                    "explanation": explanation,
                    "severity": severity,
                    "confidence": confidence_val,
                }
            )
        data["bugs"] = normalized_bugs
        return data
