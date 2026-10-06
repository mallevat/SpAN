# SpAN: Spatial-Anchored Niche Trajectory Analysis

A computational framework for placing Visium HD bins on a continuous tumor-to-lymphoid axis.

## Overview

SpAN (Spatial-Anchored Niche trajectory) combines:
1. **ONTraC** - graph neural network niche trajectory scoring from cell-type composition
2. **Spatial Gradient** - physical distance-based positioning relative to tumor and lymphoid anchor bins
3. **SpAN Score** - the average of the two, so each bin carries both its molecular and its physical position

SpAN was developed for Visium HD sections of head and neck squamous cell carcinoma (HNSCC) and accompanies the manuscript listed under Citation, where it is used to resolve the tumor-immune interface.

## Citation

If you use this code, please cite:
> Allevato MM, Krishnan SN, et al. Convergent spatial analyses define a prognostic tumor-immune interface niche in head and neck cancer. Manuscript under review.

## System requirements

- Python 3.10 with the packages listed under Installation.
- Developed and run on Linux on a SLURM cluster (see `slurm/submit_ontrac.sh`).
- ONTraC training was run on one NVIDIA GPU (`--device cuda:0`); a CPU run is possible but slower.

## Installation

### Dependencies
```bash
# Create conda environment
conda create -n span_analysis python=3.10
conda activate span_analysis

# Core dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install scanpy anndata

# ONTraC (for trajectory analysis)
pip install ONTraC

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Clone repository
```bash
git clone https://github.com/mallevat/SpAN.git
cd SpAN
```

## Demo

A Code Ocean capsule (DOI 10.24433/CO.5296569.v1, made public when the manuscript is published) runs SpAN and CLiP end to end on a representative Visium HD sample with one "Reproducible Run" click and reproduces the interface-versus-lymphoid separation reported in the manuscript. Reviewers receive access through the journal.

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Visium HD     │     │  CARD Deconv +   │     │     ONTraC      │
│   8μm Data      │────▶│  NicheCompass    │────▶│   NT Scores     │
└─────────────────┘     │  Niche Labels    │     └────────┬────────┘
                        └──────────────────┘              │
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Spatial Gradient │     │  NT Oriented    │
                        │     Score        │     │  (Tumor=0)      │
                        └────────┬─────────┘     └────────┬────────┘
                                 │                        │
                                 └──────────┬─────────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │   SpAN Score    │
                                   │  (NT + SG) / 2  │
                                   └─────────────────┘
```

## Quick Start

### 1. Prepare metadata for ONTraC
```bash
python scripts/01_prepare_ontrac_metadata.py \
    --h5ad data/sample.h5ad \
    --card_results data/card_deconv.csv \
    --output ontrac_input/metadata.csv
```

### 2. Run ONTraC
```bash
python scripts/02_run_ontrac.py \
    --metadata ontrac_input/metadata.csv \
    --output_dir results/ontrac \
    --device cuda:0 \
    --epochs 1000
```

### 3. Calculate Spatial Gradient
```bash
python scripts/03_calculate_spatial_gradient.py \
    --h5ad data/sample.h5ad \
    --niche_labels data/niche_labels.csv \
    --output results/spatial_gradient.csv
```

### 4. Compute SpAN Score
```bash
python scripts/04_compute_span_score.py \
    --nt_scores results/ontrac/NTScore.csv \
    --spatial_gradient results/spatial_gradient.csv \
    --niche_labels data/niche_labels.csv \
    --output results/span_scores.csv
```

### 5. Full pipeline (all samples)
```bash
python scripts/run_full_pipeline.py \
    --config config/samples.yaml \
    --output_dir results/
```

Niche labels are the per-bin NicheCompass niche assignments (tumor, interface, lymphoid, stromal) described in the manuscript Methods.

## File Structure

```
SpAN/
├── README.md
├── LICENSE
├── config/
│   └── samples.yaml           # Sample configuration and parameters
├── scripts/
│   ├── 01_prepare_ontrac_metadata.py
│   ├── 02_run_ontrac.py
│   ├── 03_calculate_spatial_gradient.py
│   ├── 04_compute_span_score.py
│   ├── run_full_pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── scoring_utils.py
├── analysis/
│   └── visualization.py
└── slurm/
    └── submit_ontrac.sh
```

## Methodology

### ONTraC NT Score
ONTraC uses a graph neural network to learn niche trajectories from cell-type composition. Each spot receives an NT score (0-1) based on its neighborhood's cell-type proportions.

### Spatial Gradient Score
Physical distance-based positioning using KD-tree nearest-neighbor search:

```
spatial_gradient = dist_to_tumor / (dist_to_tumor + dist_to_lymphoid)
```

Where distances are calculated to the nearest Tumor and Lymphoid anchor spots (defined by the NicheCompass niche labels).

### SpAN Score
Simple average of molecular and spatial components:

```
SpAN = (NT_oriented + spatial_gradient) / 2
```

**Rationale**: Equal weighting captures complementary information - ONTraC reflects compositional similarity while spatial gradient captures physical tissue architecture.

## Expected Results

| Niche     | NT Score | Spatial Gradient | SpAN Score |
|-----------|----------|------------------|------------|
| Tumor     | 0.13     | 0.00             | 0.07       |
| Interface | 0.59     | 0.71             | 0.65       |
| Lymphoid  | 0.61     | 1.00             | 0.80       |

**Key finding**: NT score alone cannot distinguish Interface from Lymphoid (P = 0.82), whereas SpAN separates neighboring niches (Mann-Whitney U, P < 0.003).

## Parameters

### ONTraC
| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 1000 | Training iterations |
| k-neighbors | 50 | Spatial neighborhood size |
| hidden-feats | 4 | GNN hidden dimensions |
| batch-size | 5 | Training batch size |
| seed | 42 | Random seed |

### Spatial Gradient
| Parameter | Description |
|-----------|-------------|
| Anchor niches | Tumor, Lymphoid (from NicheCompass niche labels) |
| Distance metric | Euclidean (8μm Visium HD) |
| Normalization | d_tumor / (d_tumor + d_lymphoid) |

## License

MIT License - see LICENSE file for details.

## Contact

Michael Allevato. Questions and bug reports: https://github.com/mallevat/SpAN/issues
