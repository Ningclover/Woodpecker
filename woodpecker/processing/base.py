"""Abstract base class for all processing steps."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProcessingStep(ABC):
    """A single step in the processing pipeline."""

    @abstractmethod
    def run(self, ctx) -> None:
        """Execute the step, modifying ctx in-place.

        Parameters
        ----------
        ctx : PipelineContext
            Carries FrameData, Selection, and an open outputs dict.
        """
