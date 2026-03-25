"""Simulation driver — future step to run wct-toolkit with extracted parameters.

Self-registers as "run_sim" in StepRegistry.

Reads from ctx:
    ctx.selection                          (Selection)
    ctx.outputs["track_directions"]        (from extract_tracks step)
    ctx.config.get("sim_jsonnet_template") — path to a jsonnet template
    ctx.config.get("wct_bin")             — path to wire-cell binary

Writes to ctx.outputs:
    ctx.outputs["sim_output_path"]  — path to simulation output archive
"""

from __future__ import annotations

from woodpecker.core.registry import StepRegistry
from woodpecker.processing.base import ProcessingStep


@StepRegistry.register("run_sim")
class SimDriver(ProcessingStep):
    """Run wct-toolkit simulation using selection and track directions."""

    def run(self, ctx) -> None:
        if "track_directions" not in ctx.outputs:
            raise ValueError("run_sim requires ctx.outputs['track_directions'] from extract_tracks.")

        # TODO: implement simulation driver
        # 1. Read jsonnet template from ctx.config["sim_jsonnet_template"]
        # 2. Fill in tick range, channel ranges, and track directions
        # 3. Write parameterised jsonnet to a temp file
        # 4. Invoke wire-cell subprocess:
        #    wire-cell -c <tmp.jsonnet> ...
        # 5. Store output path in ctx.outputs["sim_output_path"]

        raise NotImplementedError(
            "sim_driver is not yet implemented. "
            "See processing/sim_driver.py for the planned algorithm."
        )
