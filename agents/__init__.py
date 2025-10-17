try:
    from .code_analysis import CodeAnalysisAgent
    from .bug_detection import BugDetectionAgent
    from .best_practice import BestPracticeAdvisorAgent
except Exception:  # During initial scaffolding these may not exist yet
    CodeAnalysisAgent = object  # type: ignore
    BugDetectionAgent = object  # type: ignore
    BestPracticeAdvisorAgent = object  # type: ignore

__all__ = [
    "CodeAnalysisAgent",
    "BugDetectionAgent",
    "BestPracticeAdvisorAgent",
]

