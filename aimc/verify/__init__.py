"""Differential verification: interpreter vs. native execution vs. expectations."""
from .verifier import Check, Report, verify_module, verify_prompt, runnable_targets

__all__ = ["Check", "Report", "verify_module", "verify_prompt", "runnable_targets"]
