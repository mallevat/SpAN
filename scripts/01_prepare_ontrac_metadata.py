#!/usr/bin/env python3
"""
01_prepare_ontrac_metadata.py

Prepare metadata CSV for ONTraC input from CARD deconvolution results
and spatial coordinates.

Usage:
    python 01_prepare_ontrac_metadata.py \
        --h5ad data/sample.h5ad \
        --card_results data/card_proportions.csv \
        --output ontrac_input/metadata.csv

The output metadata.csv contains:
    - Cell_ID: unique spot identifier
    - x, y: spatial coordinates
    - Cell type columns: proportion values from CARD deconvolution
"""

import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_ontrac_metadata(
    h5ad_path: str,
    card_results_path: str,
    output_path: str,
    coord_key: str = 'spatial',
    sample_name: str = None
):
    """
    Prepare ONTraC metadata from h5ad and CARD results.
    
    Parameters
    ----------
    h5ad_path : str
        Path to AnnData h5ad file with spatial coordinates
    card_results_path : str
        Path to CARD deconvolution results CSV
    output_path : str
        Output path for metadata CSV
    coord_key : str
        Key in adata.obsm for spatial coordinates
    sample_name : str, optional
        Sample name to add as prefix to Cell_ID
    """
    logger.info(f"Loading h5ad from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    logger.info(f"Loading CARD results from {card_results_path}")
    card_df = pd.read_csv(card_results_path, index_col=0)
    
    # Extract spatial coordinates
    if coord_key not in adata.obsm:
        raise ValueError(f"Coordinate key '{coord_key}' not found in adata.obsm")
    
    coords = adata.obsm[coord_key]
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'Cell_ID': adata.obs_names,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    
    if sample_name:
        metadata['Cell_ID'] = sample_name + '_' + metadata['Cell_ID'].astype(str)
        metadata['Sample'] = sample_name
    
    # Merge with CARD proportions
    # Ensure index alignment
    card_df.index = card_df.index.astype(str)
    metadata = metadata.set_index('Cell_ID')
    
    # Get cell type columns from CARD
    cell_type_cols = card_df.columns.tolist()
    logger.info(f"Cell types from CARD: {cell_type_cols}")
    
    # Join CARD proportions
    metadata = metadata.join(card_df, how='left')
    
    # Check for missing values
    n_missing = metadata[cell_type_cols].isna().any(axis=1).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} spots missing CARD proportions")
        # Fill missing with zeros (rare/edge cases)
        metadata[cell_type_cols] = metadata[cell_type_cols].fillna(0)
    
    # Normalize proportions to sum to 1 (in case of floating point issues)
    row_sums = metadata[cell_type_cols].sum(axis=1)
    metadata[cell_type_cols] = metadata[cell_type_cols].div(row_sums, axis=0)
    
    # Reset index for output
    metadata = metadata.reset_index()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    metadata.to_csv(output_path, index=False)
    logger.info(f"Saved metadata with {len(metadata)} spots to {output_path}")
    
    # Print summary statistics
    logger.info("\nCell type proportion summary (mean across spots):")
    for ct in cell_type_cols:
        mean_prop = metadata[ct].mean()
        logger.info(f"  {ct}: {mean_prop:.3f}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Prepare metadata for ONTraC from CARD deconvolution results'
    )
    parser.add_argument(
        '--h5ad', required=True,
        help='Path to h5ad file with spatial coordinates'
    )
    parser.add_argument(
        '--card_results', required=True,
        help='Path to CARD deconvolution results CSV'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for metadata CSV'
    )
    parser.add_argument(
        '--coord_key', default='spatial',
        help='Key for spatial coordinates in adata.obsm (default: spatial)'
    )
    parser.add_argument(
        '--sample_name', default=None,
        help='Sample name to prefix Cell_IDs'
    )
    
    args = parser.parse_args()
    
    prepare_ontrac_metadata(
        h5ad_path=args.h5ad,
        card_results_path=args.card_results,
        output_path=args.output,
        coord_key=args.coord_key,
        sample_name=args.sample_name
    )


if __name__ == '__main__':
    main()
