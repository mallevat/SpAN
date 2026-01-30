# SpAN: Spatial-Anchored Niche Trajectory Analysis

A computational framework for analyzing tumor microenvironment spatial organization using HD Visium data.

## Overview

SpAN (Spatial-Anchored Niche trajectory) combines:
1. **ONTraC** - Neural network-based niche trajectory scoring based on cell-type composition
2. **Spatial Gradient** - Physical distance-based positioning relative to tissue compartments
3. **SpAN Score** - Integrated molecular + spatial trajectory positioning

This framework was developed for analyzing Interface zones in head and neck squamous cell carcinoma (HNSCC), revealing that tumor-immune boundaries confer significant survival benefits (p=0.0008, validated in 488 TCGA patients).

## Citation

If you use this code, please cite:
> [Your paper citation here]

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
git clone https://github.com/[your-username]/SpAN_analysis.git
cd SpAN_analysis
```

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Visium HD     │     │  CARD Deconv +   │     │     ONTraC      │
│   8μm Data      │────▶│  CellCompass     │────▶│   NT Scores     │
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
    --niche_labels data/cellcompass_niches.csv \
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
    --niche_labels data/cellcompass_niches.csv \
    --output results/spatial_gradient.csv
```

### 4. Compute SpAN Score
```bash
python scripts/04_compute_span_score.py \
    --nt_scores results/ontrac/NTScore.csv \
    --spatial_gradient results/spatial_gradient.csv \
    --niche_labels data/cellcompass_niches.csv \
    --output results/span_scores.csv
```

### 5. Full pipeline (all samples)
```bash
python scripts/run_full_pipeline.py \
    --config config/samples.yaml \
    --output_dir results/
```

## File Structure

```
SpAN_analysis/
├── README.md
├── LICENSE
├── config/
│   └── samples.yaml           # Sample configuration
├── scripts/
│   ├── 01_prepare_ontrac_metadata.py
│   ├── 02_run_ontrac.py
│   ├── 03_calculate_spatial_gradient.py
│   ├── 04_compute_span_score.py
│   ├── run_full_pipeline.py
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py
│       └── scoring_utils.py
├── analysis/
│   ├── trajectory_analysis.py
│   ├── gene_program_analysis.py
│   └── visualization.py
└── slurm/
    ├── submit_ontrac.sh
    └── submit_pipeline.sh
```

## Methodology

### ONTraC NT Score
ONTraC uses a graph neural network to learn niche trajectories from cell-type composition. Each spot receives an NT score (0-1) based on its neighborhood's cell-type proportions.

### Spatial Gradient Score
Physical distance-based positioning using KD-tree nearest-neighbor search:

```
spatial_gradient = dist_to_tumor / (dist_to_tumor + dist_to_lymphoid)
```

Where distances are calculated to the nearest Tumor and Lymphoid anchor spots (defined by CellCompass niche labels).

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

**Key finding**: NT score alone cannot distinguish Interface from Lymphoid (p=0.82), but SpAN provides clear separation (p=0.002).

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
| Anchor niches | Tumor, Lymphoid (from CellCompass) |
| Distance metric | Euclidean (8μm Visium HD) |
| Normalization | d_tumor / (d_tumor + d_lymphoid) |

## License

MIT License - see LICENSE file for details.

## Contact

[Your contact information]
