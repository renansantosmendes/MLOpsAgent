"""Minimal package initializer for mlops_agent.

The module imports are intentionally lightweight to avoid heavy
dependencies during imports. Tests can import specific submodules
directly (e.g., mlops_agent.ingestion).
"""

__all__ = [
    "ingestion",
    "drift",
    "training",
    "deployment",
]
