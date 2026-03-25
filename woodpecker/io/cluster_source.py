"""Cluster point-cloud source — loads WireCell 3D imaging cluster files.

Self-registers as "clusters" in SourceRegistry.

Supported file formats
----------------------
zip   — upload.zip produced by the WCP viewer, contains one or more JSON files
        under data/<eventNo>/<eventNo>-clustering-*.json.
        Each JSON has parallel arrays: x, y, z, cluster_id, q, real_cluster_id,
        plus metadata keys: eventNo, runNo, subRunNo, geom, type.

The loader merges all JSON files found in the archive into a single ClusterData,
with cluster IDs made unique across files by prepending the filename as a scope.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from woodpecker.core.exceptions import LoadError
from woodpecker.core.registry import SourceRegistry
from woodpecker.io.base import DataSource


@dataclass
class ClusterPoints:
    """3D point cloud for a single cluster."""

    cluster_id: int                       # integer ID from the file
    points: np.ndarray                    # shape (N, 3)  columns: x, y, z  [cm]
    charge: np.ndarray                    # shape (N,)    integrated charge
    source_file: str = ""                 # which JSON file this came from


@dataclass
class ClusterData:
    """All clusters loaded from one archive."""

    source_path: str
    clusters: List[ClusterPoints] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)  # eventNo, runNo, geom …

    def cluster_ids(self) -> List[int]:
        return [c.cluster_id for c in self.clusters]

    def total_points(self) -> int:
        return sum(len(c.points) for c in self.clusters)


def _load_json_flat(obj: dict, source_file: str) -> List[ClusterPoints]:
    """Parse one flat-array JSON object into a list of ClusterPoints."""
    try:
        x   = np.asarray(obj["x"],          dtype=np.float32)
        y   = np.asarray(obj["y"],          dtype=np.float32)
        z   = np.asarray(obj["z"],          dtype=np.float32)
        cid = np.asarray(obj["cluster_id"], dtype=np.int32)
        q   = np.asarray(obj.get("q", np.zeros(len(x))), dtype=np.float32)
    except KeyError as e:
        raise LoadError(f"Missing expected key {e} in {source_file}") from e

    xyz = np.stack([x, y, z], axis=1)   # (N, 3)

    clusters = []
    for uid in np.unique(cid):
        mask = cid == uid
        clusters.append(ClusterPoints(
            cluster_id=int(uid),
            points=xyz[mask],
            charge=q[mask],
            source_file=source_file,
        ))
    return clusters


@SourceRegistry.register("clusters")
class ClusterSource(DataSource):
    """Load a WireCell 3D imaging cluster archive (zip with JSON arrays)."""

    def load(self, path: str, **kwargs) -> ClusterData:
        print(f"Loading clusters from {path} ...")

        if not os.path.exists(path):
            raise LoadError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".zip":
            return self._load_zip(path)
        raise LoadError(f"Unsupported cluster file format: {ext}  (expected .zip)")

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_zip(self, path: str) -> ClusterData:
        meta: Dict[str, object] = {}
        all_clusters: List[ClusterPoints] = []

        with zipfile.ZipFile(path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_names:
                raise LoadError(f"No JSON files found inside {path}")

            for name in json_names:
                with zf.open(name) as f:
                    obj = json.load(f)

                # Collect metadata from first file
                if not meta:
                    for key in ("eventNo", "runNo", "subRunNo", "geom", "type"):
                        if key in obj:
                            meta[key] = obj[key]

                clusters = _load_json_flat(obj, source_file=name)
                all_clusters.extend(clusters)

        print(f"  Loaded {len(all_clusters)} clusters, "
              f"{sum(len(c.points) for c in all_clusters)} points total")
        return ClusterData(source_path=path, clusters=all_clusters, meta=meta)
