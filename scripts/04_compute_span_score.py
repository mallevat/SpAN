#!/usr/bin/env python3
"""
04_compute_span_score.py

Compute SpAN (Spatial-Anchored Niche trajectory) scores by combining
ONTraC NT scores with spatial gradient scores.

SpAN integrates:
1. NT scores - compositional/molecular similarity from GNN
2. Spatial gradient - physical tissue architecture

Formula: SpAN = (NT_oriented + spatial_gradient) / 2

This combination provides robust trajectory positioning that captures
both molecular neighborhood composition and physical tissue structure.

Key insight: NT score alone cannot distinguish Interface from Lymphoid
(p=0.82), but SpAN provides clear separation (p=0.002).

Usage:
    python 04_compute_span_score.py \
        --nt_scores results/ontrac/NT/NTScore.csv \
        --spatial_gradient results/spatial_gradient.csv \
        --output results/span_scores.csv
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

# Import from local utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.scoring_utils import (
    orient_nt_scores,
    compute_span_score,
    compare_niche_scores,
    save_span_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_span_from_files(
    nt_scores_path: str,
    spatial_gradient_path: str,
    output_path: str,
    niche_labels_path: str = None,
    nt_weight: float = 0.5,
    nt_col: str = 'Cell_NTScore',
    spot_col: str = 'Cell_ID'
):
    """
    Compute SpAN scores from ONTraC and spatial gradient results.
    
    Parameters
    ----------
    nt_scores_path : str
        Path to ONTraC NTScore.csv
    spatial_gradient_path : str
        Path to spatial gradient CSV
    output_path : str
        Output path for combined results
    niche_labels_path : str, optional
        Path to niche labels (if not in spatial gradient file)
    nt_weight : float
        Weight for NT component (default 0.5)
    nt_col : str
        Column name for NT scores
    spot_col : str
        Column name for spot IDs
        
    Returns
    -------
    pd.DataFrame
        Combined results with SpAN scores
    """
    # Load NT scores
    logger.info(f"Loading NT scores from {nt_scores_path}")
    nt_df = pd.read_csv(nt_scores_path)
    
    # Load spatial gradient
    logger.info(f"Loading spatial gradient from {spatial_gradient_path}")
    sg_df = pd.read_csv(spatial_gradient_path)
    
    # Standardize ID columns
    if spot_col not in nt_df.columns:
        # Try common alternatives
        for alt in ['Cell_ID', 'spot_id', 'barcode']:
            if alt in nt_df.columns:
                nt_df = nt_df.rename(columns={alt: spot_col})
                break
    
    if spot_col not in sg_df.columns:
        for alt in ['Cell_ID', 'spot_id', 'barcode']:
            if alt in sg_df.columns:
                sg_df = sg_df.rename(columns={alt: spot_col})
                break
    
    # Merge datasets
    logger.info("Merging NT scores with spatial gradient...")
    nt_df[spot_col] = nt_df[spot_col].astype(str)
    sg_df[spot_col] = sg_df[spot_col].astype(str)
    
    # Use spatial gradient as base (has coordinates and niche labels)
    results = sg_df.merge(
        nt_df[[spot_col, nt_col]],
        on=spot_col,
        how='inner'
    )
    
    logger.info(f"Merged {len(results)} spots")
    
    # Load separate niche labels if provided
    if niche_labels_path and 'Niche' not in results.columns:
        niche_df = pd.read_csv(niche_labels_path)
        niche_df[spot_col] = niche_df[spot_col].astype(str)
        results = results.merge(niche_df[[spot_col, 'Niche']], on=spot_col, how='left')
    
    # Check for niche column
    if 'Niche' not in results.columns:
        raise ValueError("Niche labels not found. Provide --niche_labels or ensure they're in spatial_gradient file.")
    
    # Orient NT scores (Tumor=0, Lymphoid=1)
    logger.info("Orienting NT scores...")
    results = orient_nt_scores(
        results,
        nt_col=nt_col,
        niche_col='Niche'
    )
    
    # Compute SpAN score
    logger.info(f"Computing SpAN score (NT weight: {nt_weight}, SG weight: {1-nt_weight})...")
    results['SpAN'] = compute_span_score(
        nt_oriented=results['NT_oriented'].values,
        spatial_gradient=results['spatial_gradient'].values,
        nt_weight=nt_weight
    )
    
    # Statistical summary
    logger.info("\n" + "="*60)
    logger.info("SCORE SUMMARY BY NICHE")
    logger.info("="*60)
    
    for score_name in ['NT_oriented', 'spatial_gradient', 'SpAN']:
        if score_name in results.columns:
            logger.info(f"\n{score_name}:")
            stats = compare_niche_scores(results, score_name)
            
            for niche in ['Tumor', 'Interface', 'Lymphoid']:
                if niche in stats['means']:
                    logger.info(f"  {niche}: {stats['means'][niche]:.3f} ± {stats['stds'][niche]:.3f}")
            
            logger.info("  Pairwise p-values:")
            for comp, pval in stats['pairwise_pvalues'].items():
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                logger.info(f"    {comp}: p={pval:.4f} {sig}")
    
    # Correlation between NT and spatial gradient
    r, p = spearmanr(results['NT_oriented'], results['spatial_gradient'])
    logger.info(f"\nNT vs Spatial Gradient correlation: r={r:.3f}, p={p:.2e}")
    
    # Key insight check
    logger.info("\n" + "="*60)
    logger.info("KEY VALIDATION: Interface vs Lymphoid discrimination")
    logger.info("="*60)
    
    interface_mask = results['Niche'] == 'Interface'
    lymphoid_mask = results['Niche'] == 'Lymphoid'
    
    for score_name in ['NT_oriented', 'SpAN']:
        if score_name in results.columns:
            _, pval = mannwhitneyu(
                results.loc[interface_mask, score_name],
                results.loc[lymphoid_mask, score_name]
            )
            sig_status = "SIGNIFICANT" if pval < 0.05 else "NOT SIGNIFICANT"
            logger.info(f"  {score_name}: p={pval:.4f} ({sig_status})")
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define output columns
    output_cols = [
        spot_col, 'x', 'y', 'Niche',
        nt_col, 'NT_oriented',
        'dist_to_tumor', 'dist_to_lymphoid', 'spatial_gradient',
        'SpAN'
    ]
    
    # Add Sample column if present
    if 'Sample' in results.columns:
        output_cols.insert(1, 'Sample')
    
    # Filter to existing columns
    output_cols = [c for c in output_cols if c in results.columns]
    
    results[output_cols].to_csv(output_path, index=False)
    logger.info(f"\nSaved SpAN results to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute SpAN scores from ONTraC NT and spatial gradient'
    )
    parser.add_argument(
        '--nt_scores', required=True,
        help='Path to ONTraC NTScore.csv'
    )
    parser.add_argument(
        '--spatial_gradient', required=True,
        help='Path to spatial gradient CSV'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for SpAN results'
    )
    parser.add_argument(
        '--niche_labels', default=None,
        help='Path to niche labels (if not in spatial gradient file)'
    )
    parser.add_argument(
        '--nt_weight', type=float, default=0.5,
        help='Weight for NT component (default: 0.5 for equal weighting)'
    )
    parser.add_argument(
        '--nt_col', default='Cell_NTScore',
        help='Column name for NT scores (default: Cell_NTScore)'
    )
    
    args = parser.parse_args()
    
    compute_span_from_files(
        nt_scores_path=args.nt_scores,
        spatial_gradient_path=args.spatial_gradient,
        output_path=args.output,
        niche_labels_path=args.niche_labels,
        nt_weight=args.nt_weight,
        nt_col=args.nt_col
    )


if __name__ == '__main__':
    main()
