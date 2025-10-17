from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deepseek_handler import DeepSeekClient
from agents.code_analysis import CodeAnalysisAgent
from agents.bug_detection import BugDetectionAgent
from agents.best_practice import BestPracticeAdvisorAgent


@dataclass
class ReviewSection:
    title: str
    content: str
    confidence: float


class ReviewManager:
    def __init__(self, client: Optional[DeepSeekClient] = None) -> None:
        self.client = client or DeepSeekClient()
        self.code_agent = CodeAnalysisAgent(self.client)
        self.bug_agent = BugDetectionAgent(self.client)
        self.practice_agent = BestPracticeAdvisorAgent(self.client)

    @staticmethod
    def infer_language_from_filename(filename: str) -> str:
        lower = (filename or "").lower()
        if lower.endswith(".py"):
            return "python"
        if lower.endswith(".js"):
            return "javascript"
        if lower.endswith(".ts"):
            return "typescript"
        if lower.endswith(".java"):
            return "java"
        return "unknown"

    def analyze(self, code_text: str, filename: str, standards_text: str) -> Dict[str, Any]:
        code_result = self.code_agent.run(code_text=code_text, filename=filename, standards_text=standards_text)
        bug_result = self.bug_agent.run(code_text=code_text, filename=filename, standards_text=standards_text)
        practice_result = self.practice_agent.run(code_text=code_text, filename=filename, standards_text=standards_text)

        score = self._compute_quality_score(code_result, bug_result, practice_result)

        report_md = self._build_markdown_report(
            filename=filename,
            code_result=code_result,
            bug_result=bug_result,
            practice_result=practice_result,
            score=score,
        )

        return {
            "language": self.infer_language_from_filename(filename),
            "code": code_result,
            "bugs": bug_result,
            "best_practices": practice_result,
            "score": score,
            "report_markdown": report_md,
        }

    def _compute_quality_score(self, code_result: Dict[str, Any], bug_result: Dict[str, Any], practice_result: Dict[str, Any]) -> int:
        score = 100.0

        # Deduct for style issues
        style_issues = code_result.get("style_issues", []) or []
        for issue in style_issues:
            severity = (issue.get("severity") or "minor").lower()
            if severity == "critical":
                score -= 5
            elif severity == "major":
                score -= 3
            else:
                score -= 1

        # Deduct for bugs
        bugs = bug_result.get("bugs", []) or []
        for bug in bugs:
            severity = (bug.get("severity") or "medium").lower()
            if severity == "high":
                score -= 25
            elif severity == "medium":
                score -= 10
            else:
                score -= 5

        # Small bonus for having concrete suggestions (cap at +10)
        suggestions = practice_result.get("suggestions", []) or []
        score += min(10, len(suggestions) * 2)

        # Clamp score to [0, 100]
        score = max(0.0, min(100.0, score))
        return int(round(score))

    def _build_markdown_report(
        self,
        filename: str,
        code_result: Dict[str, Any],
        bug_result: Dict[str, Any],
        practice_result: Dict[str, Any],
        score: int,
    ) -> str:
        summary = code_result.get("summary") or ""
        style_issues = code_result.get("style_issues", []) or []
        code_conf = float(code_result.get("confidence", 0.75))

        bugs = bug_result.get("bugs", []) or []
        bug_conf = float(bug_result.get("confidence", 0.75))

        suggestions = practice_result.get("suggestions", []) or []
        practice_conf = float(practice_result.get("confidence", 0.75))

        # Build sections
        lines: List[str] = []
        lines.append(f"# AI Code Review Report — {filename}")
        lines.append("")
        lines.append(f"**Code Quality Score:** {score}/100")
        lines.append("")
        lines.append("## 1. Summary")
        lines.append(summary or "No summary provided.")
        lines.append("")
        lines.append(f"AI Confidence: {code_conf:.2f}")
        lines.append("")

        # Style issues
        lines.append("## 2. Style and Standards Violations")
        if style_issues:
            for idx, issue in enumerate(style_issues, start=1):
                issue_text = issue.get("issue") or issue.get("message") or "Issue"
                location = issue.get("location") or "(unknown location)"
                severity = (issue.get("severity") or "minor").capitalize()
                lines.append(f"- {idx}. [{severity}] {issue_text} — at {location}")
        else:
            lines.append("- None detected.")
        lines.append("")
        lines.append(f"AI Confidence: {code_conf:.2f}")
        lines.append("")

        # Bugs
        lines.append("## 3. Potential Bugs and Runtime Risks")
        if bugs:
            for idx, bug in enumerate(bugs, start=1):
                title = bug.get("title") or "Potential issue"
                location = bug.get("location") or "(unknown location)"
                severity = (bug.get("severity") or "medium").capitalize()
                explanation = bug.get("explanation") or ""
                confidence = float(bug.get("confidence", bug_conf))
                lines.append(f"- {idx}. [{severity}] {title} — at {location}")
                if explanation:
                    lines.append(f"  - Why risky: {explanation}")
                lines.append(f"  - AI Confidence: {confidence:.2f}")
        else:
            lines.append("- None detected.")
        lines.append("")
        lines.append(f"AI Confidence: {bug_conf:.2f}")
        lines.append("")

        # Best practices
        lines.append("## 4. Suggested Improvements and Best Practices")
        if suggestions:
            for idx, s in enumerate(suggestions, start=1):
                suggestion = s.get("suggestion") or "Improvement suggestion"
                rationale = s.get("rationale") or ""
                refs = s.get("references", []) or []
                lines.append(f"- {idx}. {suggestion}")
                if rationale:
                    lines.append(f"  - Rationale: {rationale}")
                if refs:
                    lines.append(f"  - References: {', '.join(refs)}")
        else:
            lines.append("- No additional suggestions.")
        lines.append("")
        lines.append(f"AI Confidence: {practice_conf:.2f}")
        lines.append("")

        # Final notes
        lines.append("## 5. Final Notes")
        lines.append("This report was generated by multiple specialized AI agents using uploaded standards.")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def export_report_pdf(report_text: str) -> bytes:
        try:
            from fpdf import FPDF
        except Exception:
            # If PDF export is not available, return empty bytes
            return b""

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", size=11)

        # Naive word-wrap
        max_width = 180
        for line in report_text.splitlines():
            if not line:
                pdf.ln(5)
                continue
            # Basic handling of overly long lines
            words = line.split(" ")
            current = ""
            for w in words:
                test_line = (current + " " + w).strip()
                if pdf.get_string_width(test_line) > max_width:
                    pdf.multi_cell(0, 5, current)
                    current = w
                else:
                    current = test_line
            if current:
                pdf.multi_cell(0, 5, current)
        buffer = io.BytesIO()
        pdf.output(buffer)
        return buffer.getvalue()
