from dataclasses import dataclass
from typing import Dict, Any

from deepseek_handler import DeepSeekClient


SYSTEM_PROMPT = (
    "You are CodeAnalysisAgent. Analyze code structure, readability, naming, modularity,"
    " comments, and adherence to provided standards. Return JSON with keys: summary,"
    " style_issues (array of {issue, location, severity}), reasoning (short),"
    " confidence (0-1)."
)


@dataclass
class CodeAnalysisResult:
    summary: str
    style_issues: Any
    reasoning: str
    confidence: float


class CodeAnalysisAgent:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def run(self, code_text: str, filename: str, standards_text: str) -> Dict[str, Any]:
        user_prompt = (
            "Analyze the following code using the given standards.\n\n"
            f"Filename: {filename}\n\n"
            "Standards (extract key rules and compare):\n" + standards_text[:12000] + "\n\n"
            "Code:\n" + code_text[:12000] + "\n\n"
            "Return strictly JSON with keys: summary, style_issues, reasoning, confidence."
        )
        resp = self.client.chat_complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            response_format="json",
        )
        data: Dict[str, Any] = resp.parsed_json or {
            "summary": "",
            "style_issues": [],
            "reasoning": "",
            "confidence": 0.7,
        }
        # Normalize minimal schema
        data.setdefault("summary", "")
        data.setdefault("style_issues", [])
        data.setdefault("reasoning", "")
        data.setdefault("confidence", 0.7)
        return data
