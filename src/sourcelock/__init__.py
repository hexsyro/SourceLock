"""sourcelock — catch AI-summary claims the source posts never supported."""

from sourcelock.common import RESULT_SCHEMA_VERSION, ToolResult
from sourcelock.logic import SourceLockLogic, run

__version__ = "0.2.0"

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "SourceLockLogic",
    "ToolResult",
    "__version__",
    "run",
]
