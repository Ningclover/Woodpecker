# woodpecker

WireCell targeted region selection and debugging tool.

## Install

```bash
cd woodpecker
pip install -e .
```

Requires Python ≥ 3.9, numpy, matplotlib.

After install, activate the environment (e.g. `direnv allow`) so that
`woodpecker` is on your PATH.  Without installing you can also run:

```bash
python -m woodpecker <subcommand> ...
```

---

## Typical workflow

```
protodune-sp-frames-anode<N>.tar.bz2   (WireCell FrameFileSink output)
          │
          ▼
   woodpecker select        → masked  anode<N>.tar.bz2    in woodpecker_data/
          │
          ▼
   woodpecker run-img       → imaging clusters             in woodpecker_data/
          │
          ▼
   woodpecker run-clustering → bee upload.zip + tracks-*.json  in woodpecker_data/
          │
          ├─── woodpecker extract-tracks   → inspect track list
          │
          ├─── woodpecker run-sim-check    → simulated raw frames (longest track)
          │                                  woodpecker_data/protodune-sp-frames-sim-anode<N>.tar.bz2
          │
          └─── woodpecker plot-frames      → U/V/W wire-plane image (PNG)
```

---

## Commands

### `select` — interactive frame selection GUI

Displays the three wire-plane (U/V/W) gauss frame images and walks you through
four sequential selection steps.

```bash
woodpecker select protodune-sp-frames-anode0.tar.bz2
woodpecker select protodune-sp-frames-anode0.tar.bz2 --vmax 1000
woodpecker select protodune-sp-frames-anode0.tar.bz2 --out my_output.tar.bz2
woodpecker select protodune-sp-frames-anode0.tar.bz2 --save-selection sel.json
```

Selection workflow:

| Step | Action | Gesture |
|------|--------|---------|
| 1 | Tick range | drag UP/DOWN on any plot |
| 2 | U channel range | drag LEFT/RIGHT on plane U |
| 3 | V channel range | drag LEFT/RIGHT on plane V |
| 4 | W channel range | drag LEFT/RIGHT on plane W |

- Press **ENTER** to confirm each step and advance.
- Press **r** to restart from Step 1.
- Click **[Save selection]** when all four steps are done.

The output is a new `.tar.bz2` with the same file/array structure as the
input. Data outside the selected tick and channel ranges is zeroed out; all
shapes and dtypes are preserved so the file is a drop-in replacement for
`img.jsonnet`.

Output is written to `woodpecker_data/<basename>` by default.

---

### `mask` — non-interactive (batch) masking

Apply a previously saved selection to an archive without opening a GUI.

```bash
# Step 1 — save selection during a GUI session:
woodpecker select anode0.tar.bz2 --save-selection sel.json

# Step 2 — apply it (can be scripted):
woodpecker mask anode0.tar.bz2 --selection sel.json --out anode0-masked.tar.bz2
```

---

### `run-img` — run WireCell imaging on masked frames

Invokes `wire-cell` with the imaging jsonnet on the masked frame files found
in `woodpecker_data/`.

```bash
woodpecker run-img
woodpecker run-img --datadir woodpecker_data
woodpecker run-img --anode-indices '[2]'
woodpecker run-img --dry-run
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--datadir` | `woodpecker_data` | Directory containing masked `*-anode<N>.tar.bz2` files |
| `--anode-indices` | auto-detect | JSON list of anode indices, e.g. `'[1,2]'` |
| `--output-prefix` | `<datadir>/protodune-sp-frames-img` | Prefix for output files |
| `--jsonnet` | auto-search | Path to imaging jsonnet |
| `--wct-base` | `/nfs/data/1/xning/wirecell-working` | WCT_BASE directory |
| `--log-level` | `debug` | Wire-cell log level |
| `--dry-run` | false | Print command without executing |

---

### `run-clustering` — run WireCell clustering and upload to bee

Invokes `wire-cell` with the clustering jsonnet, then packages the output as
a bee-compatible upload zip.

```bash
woodpecker run-clustering
woodpecker run-clustering --datadir woodpecker_data
woodpecker run-clustering --anode-indices '[2]'
woodpecker run-clustering --dry-run
```

Options mirror `run-img`; see `--help` for details.

Output: `woodpecker_data/upload.zip` (bee viewer) and `woodpecker_data/tracks-<N>.json`
(track info for `extract-tracks` and `run-sim-check`).

---

### `extract-tracks` — derive track directions from 3D imaging clusters

Loads a WireCell 3D imaging cluster file (zip with flat-array JSON), runs
PCA on each cluster's 3D point cloud, and reports the dominant direction,
length, total charge, and endpoints.

```bash
woodpecker extract-tracks upload.zip
woodpecker extract-tracks upload.zip --out tracks.json
woodpecker extract-tracks upload.zip --out tracks.json --min-points 5
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--out` | (none) | Save results as JSON |
| `--min-points` | 2 | Skip clusters with fewer points than this |

#### Input format

A zip file containing one or more JSON files, each with parallel arrays:

```
x, y, z         — 3D position [cm]
cluster_id      — integer cluster label
q               — charge at each point
```

This is the format produced by the WCP viewer (`upload.zip`).

#### Output fields (per cluster)

| Field | Description |
|-------|-------------|
| `cluster_id` | Integer ID from the input file |
| `n_points` | Number of 3D points in the cluster |
| `total_charge` | Sum of `q` over all points |
| `centroid` | Mean position `[x, y, z]` cm |
| `direction` | Unit vector along dominant axis (PCA first component) |
| `length_cm` | Extent along dominant axis (max − min projection) |
| `start`, `end` | 3D endpoints along the dominant axis |
| `linearity` | Fraction of variance in dominant direction (1=line, 0=blob) |
| `theta_deg` | Polar angle from +z axis |
| `phi_deg` | Azimuthal angle in x-y plane from +x axis |

#### Algorithm

1. Subtract the centroid from all points.
2. Run SVD: the first right-singular vector is the direction of maximum variance
   (equivalent to PCA component 1), using only `numpy` — no extra dependencies.
3. Project all points onto that axis to get scalar coordinates `t`.
4. `length = max(t) − min(t)`;  endpoints = `centroid ± t·direction`.
5. `linearity = s[0]² / sum(s²)` where `s` are the singular values.

---

### `run-sim-check` — simulate longest extracted track

Picks the longest track from a `woodpecker extract-tracks` JSON output,
simulates it with `wire-cell` (full sim + noise filtering), and saves the
resulting raw NF frames as `woodpecker_data/protodune-sp-frames-sim-anode<N>.tar.bz2`.

These files have the same `tar.bz2` format as the input frames and can be
inspected with `woodpecker plot-frames`.

```bash
woodpecker run-sim-check
woodpecker run-sim-check --tracks-file woodpecker_data/tracks-upload.json
woodpecker run-sim-check --anode-indices '[2]'
woodpecker run-sim-check --dry-run
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tracks-file` | auto-detect `woodpecker_data/tracks-*.json` | Track JSON from `extract-tracks` |
| `--datadir` | `woodpecker_data` | Used for auto-detecting anode indices |
| `--anode-indices` | auto-detect from masked files | JSON list e.g. `'[2]'` |
| `--output-prefix` | `<datadir>/protodune-sp-frames-sim` | Prefix for output tar.bz2 files |
| `--jsonnet` | auto-search for `wcp-porting-img/pdvd/wct-sim-check-track.jsonnet` | Simulation jsonnet |
| `--wct-base` | `/nfs/data/1/xning/wirecell-working` | WCT_BASE directory |
| `--log-level` | `debug` | Wire-cell log level |
| `--dry-run` | false | Print command without executing |

The command prints the selected track's properties before running:

```
--- longest track selected for simulation ---
  cluster_id : 12
  source     : data/0/0-clustering-apa2-face0.json
  length_cm  : 137.12
  linearity  : 0.9975
  theta_deg  : 70.07
  phi_deg    : 20.71
  start (cm) : [-190.44, 279.93, -0.44]
  end   (cm) : [-69.86, 325.51, 46.30]
--------------------------------------------
```

---

### `plot-frames` — draw U/V/W wire plane views from a tar.bz2

Reads a `FrameFileSink` tar.bz2 archive and produces a PNG with one subplot
per wire plane (U, V, W).  The x-axis is channel number, y-axis is tick
(0-based relative index).

```bash
woodpecker plot-frames woodpecker_data/protodune-sp-frames-sim-anode2.tar.bz2
woodpecker plot-frames data.tar.bz2 --tag raw2
woodpecker plot-frames data.tar.bz2 --out frames.png
woodpecker plot-frames data.tar.bz2 --tick-range 1000 3000
woodpecker plot-frames data.tar.bz2 --zrange -50 50
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tag` | auto-detect (raw > gauss > wiener > \*) | Frame tag to display |
| `--out` | `<input>.png` | Output PNG path |
| `--tick-range T0 T1` | full range | Restrict to tick indices T0..T1 |
| `--zrange ZMIN ZMAX` | ±3 × RMS | ADC color scale range |
| `--dpi` | 150 | Output image DPI |

Color scale uses a diverging RdBu_r colormap centered at zero.

---

## Architecture

```
woodpecker/
├── core/
│   ├── selection.py       # Selection dataclass (tick_range + ch_ranges)
│   ├── registry.py        # SourceRegistry / StepRegistry — plugin system
│   └── exceptions.py      # Exception hierarchy
├── io/
│   ├── base.py            # DataSource ABC
│   ├── frame_data.py      # FrameData / PlaneData dataclasses
│   ├── frame_source.py    # Gauss/wiener tar.bz2 loader   [registered as "frames"]
│   └── cluster_source.py  # WCP zip cluster loader         [registered as "clusters"]
├── gui/
│   ├── app.py             # Figure assembly and event wiring
│   ├── controller.py      # Step-machine state (no matplotlib dependency)
│   ├── widgets.py         # SpanSelector / Button / text bar factories
│   └── overlays.py        # Highlight band helpers
├── processing/
│   ├── base.py            # ProcessingStep ABC
│   ├── masker.py          # Write masked archive            [registered as "mask_frames"]
│   └── track_extractor.py # PCA track direction extraction  [registered as "extract_tracks"]
├── pipeline/
│   ├── context.py         # PipelineContext carrier dataclass
│   └── runner.py          # Resolve steps by name and run in sequence
└── cli/
    ├── main.py                # Top-level argparse with subcommands
    ├── cmd_select.py          # `select` subcommand (GUI + mask)
    ├── cmd_mask.py            # `mask` subcommand (non-interactive)
    ├── cmd_extract.py         # `extract-tracks` subcommand
    ├── cmd_run_img.py         # `run-img` subcommand (wire-cell imaging)
    ├── cmd_run_clustering.py  # `run-clustering` subcommand (wire-cell clustering)
    ├── cmd_run_sim_check.py   # `run-sim-check` subcommand (track simulation)
    └── cmd_plot_frames.py     # `plot-frames` subcommand (U/V/W PNG)
```

### How the plugin system works

Sources and processing steps register themselves by name using decorators:

```python
@SourceRegistry.register("clusters")
class ClusterSource(DataSource): ...

@StepRegistry.register("extract_tracks")
class TrackExtractor(ProcessingStep): ...
```

CLI commands look them up by name at runtime.  Adding a new source or step
requires no changes to any existing file — only the new file and a one-line
import in the relevant CLI command.
