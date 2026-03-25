"""Track direction extractor — derives dominant direction and length per cluster.

Self-registers as "extract_tracks" in StepRegistry.

Algorithm
---------
For each cluster (a set of 3D points):

  1. Centre the points by subtracting their centroid.
  2. Run SVD on the centred point matrix.
       X_centred = U · S · Vt
     The first row of Vt (= V[:, 0]) is the direction of maximum variance —
     the dominant axis, equivalent to the first PCA component.
  3. Project every point onto that axis:
       t_i = dot(point_i - centroid, direction)
  4. Track length = max(t) - min(t).
  5. Endpoints = centroid + t_min * direction,  centroid + t_max * direction.
  6. The start/end are ordered so that the direction vector points from start
     to end (always positive).

Why SVD instead of sklearn PCA?
  numpy.linalg.svd is always available in this environment; no extra dependency.
  For a line-like cluster the result is identical.

Reads from ctx:
    ctx.cluster_data   (ClusterData)

Writes to ctx.outputs:
    ctx.outputs["track_results"]  — list of TrackResult (one per cluster)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from woodpecker.core.registry import StepRegistry
from woodpecker.processing.base import ProcessingStep


@dataclass
class TrackResult:
    """PCA-derived track properties for one cluster."""

    cluster_id: int
    n_points: int

    centroid: np.ndarray      # shape (3,)  [x, y, z] cm
    direction: np.ndarray     # shape (3,)  unit vector along dominant axis

    length: float             # cm, extent along dominant axis
    start: np.ndarray         # shape (3,)  endpoint with smaller projection
    end: np.ndarray           # shape (3,)  endpoint with larger projection
    total_charge: float       # sum of q for all points in the cluster

    # Fraction of total variance explained by dominant axis (0–1).
    # Close to 1 → looks like a straight track.
    # Close to 0 → isotropic blob.
    linearity: float

    source_file: str = ""

    def direction_angles_deg(self):
        """Return (theta, phi) in degrees.

        theta — polar angle from +z axis  (0° = along z)
        phi   — azimuthal angle in x-y plane from +x axis
        """
        dx, dy, dz = self.direction
        theta = float(np.degrees(np.arccos(np.clip(dz, -1.0, 1.0))))
        phi   = float(np.degrees(np.arctan2(dy, dx)))
        return theta, phi

    def __repr__(self) -> str:
        theta, phi = self.direction_angles_deg()
        return (
            f"TrackResult(cluster_id={self.cluster_id}, "
            f"n={self.n_points}, "
            f"length={self.length:.1f} cm, "
            f"total_q={self.total_charge:.1f}, "
            f"linearity={self.linearity:.3f}, "
            f"theta={theta:.1f}°, phi={phi:.1f}°)"
        )


def _pca_track(points: np.ndarray) -> tuple:
    """Return (centroid, direction, length, start, end, linearity).

    points : (N, 3) float array
    """
    centroid = points.mean(axis=0)
    centred  = points - centroid

    # SVD: centred = U S Vt
    # Vt rows are principal directions, sorted by descending singular value.
    _, s, vt = np.linalg.svd(centred, full_matrices=False)

    direction = vt[0]                    # dominant axis, unit vector
    direction = direction / np.linalg.norm(direction)   # ensure unit length

    # Project points onto dominant axis
    projections = centred @ direction    # shape (N,)

    t_min, t_max = float(projections.min()), float(projections.max())
    length = t_max - t_min
    start  = centroid + t_min * direction
    end    = centroid + t_max * direction

    # Linearity: fraction of variance in dominant direction
    variances  = s ** 2
    linearity  = float(variances[0] / variances.sum()) if variances.sum() > 0 else 0.0

    return centroid, direction, length, start, end, linearity


def extract_tracks(cluster_data) -> List[TrackResult]:
    """Run PCA on every cluster and return a list of TrackResult.

    This is the pure-computation function, usable without the pipeline framework.
    """
    results = []
    for cl in cluster_data.clusters:
        pts = cl.points
        total_q = float(cl.charge.sum())

        if len(pts) < 2:
            # Degenerate cluster — can't fit a line
            centroid = pts.mean(axis=0) if len(pts) == 1 else np.zeros(3)
            results.append(TrackResult(
                cluster_id=cl.cluster_id,
                n_points=len(pts),
                centroid=centroid,
                direction=np.array([1.0, 0.0, 0.0]),
                length=0.0,
                start=centroid.copy(),
                end=centroid.copy(),
                linearity=0.0,
                total_charge=total_q,
                source_file=cl.source_file,
            ))
            continue

        centroid, direction, length, start, end, linearity = _pca_track(pts)
        results.append(TrackResult(
            cluster_id=cl.cluster_id,
            n_points=len(pts),
            centroid=centroid,
            direction=direction,
            length=length,
            start=start,
            end=end,
            linearity=linearity,
            total_charge=total_q,
            source_file=cl.source_file,
        ))
    return results


@StepRegistry.register("extract_tracks")
class TrackExtractor(ProcessingStep):
    """Extract dominant track directions from 3D imaging cluster data."""

    def run(self, ctx) -> None:
        if ctx.cluster_data is None:
            raise ValueError("extract_tracks requires ctx.cluster_data to be set.")

        results = extract_tracks(ctx.cluster_data)
        ctx.outputs["track_results"] = results

        print(f"\n{'='*55}")
        print(f"Track extraction results ({len(results)} clusters):")
        for r in results:
            print(f"  {r}")
        print(f"{'='*55}\n")
