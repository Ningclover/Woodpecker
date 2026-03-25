"""Pipeline runner — resolves and executes processing steps in order."""

from __future__ import annotations

from typing import List

from woodpecker.core.exceptions import PipelineError
from woodpecker.core.registry import StepRegistry
from woodpecker.pipeline.context import PipelineContext


class PipelineRunner:
    """Run a sequence of named processing steps against a PipelineContext."""

    def __init__(self, step_names: List[str]) -> None:
        self.step_names = step_names

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for name in self.step_names:
            step_cls = StepRegistry.get(name)
            step = step_cls()
            print(f"[pipeline] running step: {name}")
            try:
                step.run(ctx)
            except Exception as exc:
                raise PipelineError(f"Step '{name}' failed: {exc}") from exc
        return ctx
