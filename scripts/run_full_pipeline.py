#!/usr/bin/env python3
"""
run_full_pipeline.py

Run complete SpAN analysis pipeline for multiple samples.

This script orchestrates:
1. Metadata preparation for ONTraC
2. ONTraC trajectory analysis
3. Spatial gradient calculation
4. SpAN score computation
5. Cross-sample validation

Usage:
    python scripts/run_full_pipeline.py \
        --config config/samples.yaml \
        --output_dir results/
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_single_sample(
    sample_name: str,
    h5ad_path: str,
    card_path: str,
    niche_path: str,
    output_dir: str,
    device: str = 'cpu',
    skip_ontrac: bool = False
) -> pd.DataFrame:
    """
    Process a single sample through the pipeline.
    
    Parameters
    ----------
    sample_name : str
        Sample identifier
    h5ad_path : str
        Path to h5ad file
    card_path : str
        Path to CARD results
    niche_path : str
        Path to niche labels
    output_dir : str
        Output directory for this sample
    device : str
        Compute device for ONTraC
    skip_ontrac : bool
        Skip ONTraC if results exist
        
    Returns
    -------
    pd.DataFrame
        SpAN results for this sample
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing sample: {sample_name}")
    
    # Import pipeline components
    from scripts import (
        prepare_ontrac_metadata,
        run_ontrac,
        calculate_spatial_gradient_from_files,
        compute_span_from_files
    )
    
    # Step 1: Prepare ONTraC metadata
    metadata_path = output_dir / 'metadata.csv'
    if not metadata_path.exists():
        logger.info(f"  Preparing metadata...")
        prepare_ontrac_metadata(
            h5ad_path=h5ad_path,
            card_results_path=card_path,
            output_path=str(metadata_path),
            sample_name=sample_name
        )
    
    # Step 2: Run ONTraC
    nt_score_path = output_dir / 'ontrac' / 'NT' / 'NTScore.csv'
    if not nt_score_path.exists() and not skip_ontrac:
        logger.info(f"  Running ONTraC...")
        run_ontrac(
            metadata_path=str(metadata_path),
            output_dir=str(output_dir / 'ontrac'),
            device=device
        )
    elif skip_ontrac and not nt_score_path.exists():
        raise FileNotFoundError(f"NT scores not found and skip_ontrac=True: {nt_score_path}")
    
    # Step 3: Calculate spatial gradient
    sg_path = output_dir / 'spatial_gradient.csv'
    if not sg_path.exists():
        logger.info(f"  Calculating spatial gradient...")
        calculate_spatial_gradient_from_files(
            h5ad_path=h5ad_path,
            niche_path=niche_path,
            output_path=str(sg_path),
            sample_name=sample_name
        )
    
    # Step 4: Compute SpAN score
    span_path = output_dir / 'span_scores.csv'
    logger.info(f"  Computing SpAN scores...")
    results = compute_span_from_files(
        nt_scores_path=str(nt_score_path),
        spatial_gradient_path=str(sg_path),
        output_path=str(span_path)
    )
    
    results['Sample'] = sample_name
    logger.info(f"  Completed: {len(results)} spots")
    
    return results


def combine_sample_results(
    sample_results: List[pd.DataFrame],
    output_path: str
):
    """
    Combine results from multiple samples.
    
    Parameters
    ----------
    sample_results : List[pd.DataFrame]
        List of per-sample result DataFrames
    output_path : str
        Output path for combined CSV
    """
    combined = pd.concat(sample_results, ignore_index=True)
    
    logger.info(f"\nCombined results: {len(combined)} total spots")
    logger.info(f"Samples: {combined['Sample'].nunique()}")
    
    # Summary by sample and niche
    summary = combined.groupby(['Sample', 'Niche']).agg({
        'NT_oriented': ['mean', 'std'],
        'spatial_gradient': ['mean', 'std'],
        'SpAN': ['mean', 'std']
    }).round(3)
    
    logger.info(f"\nSummary by sample and niche:\n{summary}")
    
    # Save combined results
    combined.to_csv(output_path, index=False)
    logger.info(f"\nSaved combined results to {output_path}")
    
    return combined


def cross_sample_validation(
    combined_df: pd.DataFrame,
    output_dir: str,
    min_samples: int = 4
):
    """
    Perform cross-sample validation of SpAN scores.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined results from all samples
    output_dir : str
        Output directory for validation results
    min_samples : int
        Minimum samples required for consensus (default: 4/6)
    """
    from scipy.stats import mannwhitneyu
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("CROSS-SAMPLE VALIDATION")
    logger.info("="*60)
    
    samples = combined_df['Sample'].unique()
    n_samples = len(samples)
    
    # Per-sample statistics
    sample_stats = []
    for sample in samples:
        sample_df = combined_df[combined_df['Sample'] == sample]
        
        for score_name in ['NT_oriented', 'spatial_gradient', 'SpAN']:
            for niche in ['Tumor', 'Interface', 'Lymphoid']:
                niche_scores = sample_df[sample_df['Niche'] == niche][score_name]
                if len(niche_scores) > 0:
                    sample_stats.append({
                        'Sample': sample,
                        'Score': score_name,
                        'Niche': niche,
                        'Mean': niche_scores.mean(),
                        'Std': niche_scores.std(),
                        'N': len(niche_scores)
                    })
    
    stats_df = pd.DataFrame(sample_stats)
    stats_df.to_csv(output_dir / 'per_sample_statistics.csv', index=False)
    
    # Check Interface vs Lymphoid discrimination per sample
    logger.info("\nInterface vs Lymphoid discrimination by sample:")
    discrimination_results = []
    
    for sample in samples:
        sample_df = combined_df[combined_df['Sample'] == sample]
        interface_span = sample_df[sample_df['Niche'] == 'Interface']['SpAN']
        lymphoid_span = sample_df[sample_df['Niche'] == 'Lymphoid']['SpAN']
        
        if len(interface_span) > 0 and len(lymphoid_span) > 0:
            _, pval = mannwhitneyu(interface_span, lymphoid_span)
            sig = pval < 0.05
            
            discrimination_results.append({
                'Sample': sample,
                'Interface_mean': interface_span.mean(),
                'Lymphoid_mean': lymphoid_span.mean(),
                'p_value': pval,
                'Significant': sig
            })
            
            status = "✓" if sig else "✗"
            logger.info(f"  {sample}: p={pval:.4f} {status}")
    
    disc_df = pd.DataFrame(discrimination_results)
    disc_df.to_csv(output_dir / 'discrimination_validation.csv', index=False)
    
    # Consensus check
    n_significant = disc_df['Significant'].sum()
    logger.info(f"\nConsensus: {n_significant}/{n_samples} samples show significant discrimination")
    
    if n_significant >= min_samples:
        logger.info(f"✓ PASSED: ≥{min_samples} samples required")
    else:
        logger.warning(f"✗ FAILED: <{min_samples} samples show significant discrimination")
    
    return disc_df


def main():
    parser = argparse.ArgumentParser(
        description='Run complete SpAN analysis pipeline'
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for all results'
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Compute device for ONTraC (default: cpu)'
    )
    parser.add_argument(
        '--skip_ontrac', action='store_true',
        help='Skip ONTraC (use existing results)'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Process samples in parallel (CPU-only mode)'
    )
    parser.add_argument(
        '--n_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    samples = config['samples']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(samples)} samples")
    
    # Process samples
    all_results = []
    
    if args.parallel and not args.skip_ontrac:
        logger.warning("Parallel mode with ONTraC not recommended (GPU memory). Switching to sequential.")
        args.parallel = False
    
    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {}
            for sample in samples:
                sample_name = sample['name']
                sample_output = output_dir / sample_name
                
                future = executor.submit(
                    process_single_sample,
                    sample_name=sample_name,
                    h5ad_path=sample['h5ad'],
                    card_path=sample['card'],
                    niche_path=sample['niches'],
                    output_dir=str(sample_output),
                    device=args.device,
                    skip_ontrac=args.skip_ontrac
                )
                futures[future] = sample_name
            
            for future in as_completed(futures):
                sample_name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed processing {sample_name}: {e}")
    else:
        for sample in samples:
            sample_name = sample['name']
            sample_output = output_dir / sample_name
            
            try:
                result = process_single_sample(
                    sample_name=sample_name,
                    h5ad_path=sample['h5ad'],
                    card_path=sample['card'],
                    niche_path=sample['niches'],
                    output_dir=str(sample_output),
                    device=args.device,
                    skip_ontrac=args.skip_ontrac
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed processing {sample_name}: {e}")
    
    # Combine results
    if all_results:
        combined_path = output_dir / 'combined_span_scores.csv'
        combined = combine_sample_results(all_results, str(combined_path))
        
        # Cross-sample validation
        cross_sample_validation(
            combined,
            output_dir=str(output_dir / 'validation'),
            min_samples=4
        )
    else:
        logger.error("No samples processed successfully")


if __name__ == '__main__':
    main()
