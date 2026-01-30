#!/usr/bin/env python3
"""
03_calculate_spatial_gradient.py

Calculate spatial gradient scores based on physical distance to anchor niches.

The spatial gradient provides a continuous measure of position along the
Tumor → Interface → Lymphoid axis based purely on physical tissue architecture,
independent of molecular composition.

Formula:
    spatial_gradient = dist_to_tumor / (dist_to_tumor + dist_to_lymphoid)

Where:
    - Tumor anchor spots → 0.0
    - Lymphoid anchor spots → 1.0  
    - Interface spots → intermediate values (~0.7)

Usage:
    python 03_calculate_spatial_gradient.py \
        --h5ad data/sample.h5ad \
        --niche_labels data/cellcompass_niches.csv \
        --output results/spatial_gradient.csv
"""

import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import logging
from pathlib import Path
from scipy.spatial import cKDTree

# Import from local utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.scoring_utils import (
    calculate_spatial_gradient,
    calculate_spatial_gradient_geometric,
    validate_inputs
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_niche_labels(
    niche_path: str,
    spot_col: str = 'spot_id',
    niche_col: str = 'Niche'
) -> pd.DataFrame:
    """Load niche labels from CSV."""
    df = pd.read_csv(niche_path)
    
    # Standardize column names
    if spot_col in df.columns:
        df = df.rename(columns={spot_col: 'Cell_ID'})
    elif 'Cell_ID' not in df.columns:
        # Assume first column is spot ID
        df = df.rename(columns={df.columns[0]: 'Cell_ID'})
    
    if niche_col not in df.columns:
        raise ValueError(f"Niche column '{niche_col}' not found in {niche_path}")
    
    return df[['Cell_ID', niche_col]]


def calculate_spatial_gradient_from_files(
    h5ad_path: str,
    niche_path: str,
    output_path: str,
    coord_key: str = 'spatial',
    tumor_label: str = 'Tumor',
    lymphoid_label: str = 'Lymphoid',
    sample_name: str = None,
    include_geometric: bool = False
):
    """
    Calculate spatial gradient from h5ad and niche labels.
    
    Parameters
    ----------
    h5ad_path : str
        Path to h5ad file with spatial coordinates
    niche_path : str
        Path to CSV with niche labels
    output_path : str
        Output CSV path
    coord_key : str
        Key for coordinates in adata.obsm
    tumor_label : str
        Label for tumor niche (anchor at 0)
    lymphoid_label : str
        Label for lymphoid niche (anchor at 1)
    sample_name : str, optional
        Sample name to add
    include_geometric : bool
        Also calculate geometric mean version
        
    Returns
    -------
    pd.DataFrame
        DataFrame with spatial gradient scores
    """
    # Load data
    logger.info(f"Loading h5ad from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    logger.info(f"Loading niche labels from {niche_path}")
    niche_df = load_niche_labels(niche_path)
    
    # Get coordinates
    coords = adata.obsm[coord_key]
    spot_ids = adata.obs_names.values
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Cell_ID': spot_ids,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    
    if sample_name:
        results['Sample'] = sample_name
    
    # Merge niche labels
    results['Cell_ID'] = results['Cell_ID'].astype(str)
    niche_df['Cell_ID'] = niche_df['Cell_ID'].astype(str)
    results = results.merge(niche_df, on='Cell_ID', how='left')
    
    # Check for missing niches
    n_missing = results['Niche'].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} spots missing niche labels")
        results = results.dropna(subset=['Niche'])
    
    logger.info(f"Niche distribution:\n{results['Niche'].value_counts()}")
    
    # Calculate spatial gradient
    niches = results['Niche'].values
    spot_coords = results[['x', 'y']].values
    
    validate_inputs(spot_coords, niches)
    
    logger.info("Calculating spatial gradient (arithmetic mean normalization)...")
    sg, dist_tumor, dist_lymphoid = calculate_spatial_gradient(
        coords=spot_coords,
        niches=niches,
        tumor_label=tumor_label,
        lymphoid_label=lymphoid_label
    )
    
    results['dist_to_tumor'] = dist_tumor
    results['dist_to_lymphoid'] = dist_lymphoid
    results['spatial_gradient'] = sg
    
    # Optional: geometric mean version for comparison
    if include_geometric:
        logger.info("Calculating spatial gradient (geometric mean normalization)...")
        sg_geom = calculate_spatial_gradient_geometric(
            coords=spot_coords,
            niches=niches,
            tumor_label=tumor_label,
            lymphoid_label=lymphoid_label
        )
        results['spatial_gradient_geometric'] = sg_geom
        
        # Report correlation
        corr = np.corrcoef(sg, sg_geom)[0, 1]
        logger.info(f"Arithmetic vs Geometric correlation: r={corr:.4f}")
    
    # Summary statistics by niche
    logger.info("\nSpatial gradient by niche:")
    for niche in [tumor_label, 'Interface', lymphoid_label]:
        if niche in results['Niche'].values:
            niche_sg = results[results['Niche'] == niche]['spatial_gradient']
            logger.info(f"  {niche}: {niche_sg.mean():.3f} ± {niche_sg.std():.3f}")
    
    # Check niche balance
    n_tumor = (niches == tumor_label).sum()
    n_lymphoid = (niches == lymphoid_label).sum()
    ratio = n_lymphoid / n_tumor if n_tumor > 0 else np.inf
    logger.info(f"\nLymphoid:Tumor ratio: {ratio:.2f}")
    
    if ratio < 0.5 or ratio > 2.0:
        logger.warning(
            f"Niche imbalance detected (ratio={ratio:.2f}). "
            "Consider balanced sampling for sensitivity analysis."
        )
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Saved spatial gradient results to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Calculate spatial gradient scores from tissue architecture'
    )
    parser.add_argument(
        '--h5ad', required=True,
        help='Path to h5ad file with spatial coordinates'
    )
    parser.add_argument(
        '--niche_labels', required=True,
        help='Path to CSV with niche labels'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output CSV path'
    )
    parser.add_argument(
        '--coord_key', default='spatial',
        help='Key for coordinates in adata.obsm (default: spatial)'
    )
    parser.add_argument(
        '--tumor_label', default='Tumor',
        help='Label for tumor niche (default: Tumor)'
    )
    parser.add_argument(
        '--lymphoid_label', default='Lymphoid',
        help='Label for lymphoid niche (default: Lymphoid)'
    )
    parser.add_argument(
        '--sample_name', default=None,
        help='Sample name to add'
    )
    parser.add_argument(
        '--include_geometric', action='store_true',
        help='Also calculate geometric mean version'
    )
    
    args = parser.parse_args()
    
    calculate_spatial_gradient_from_files(
        h5ad_path=args.h5ad,
        niche_path=args.niche_labels,
        output_path=args.output,
        coord_key=args.coord_key,
        tumor_label=args.tumor_label,
        lymphoid_label=args.lymphoid_label,
        sample_name=args.sample_name,
        include_geometric=args.include_geometric
    )


if __name__ == '__main__':
    main()
