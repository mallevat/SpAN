#!/usr/bin/env python3
"""
02_run_ontrac.py

Run ONTraC (Ordered Niche TRAjectory Construction) analysis.

ONTraC uses graph neural networks to learn continuous niche trajectories
from cell-type composition data. Each spot receives an NT (Niche Trajectory)
score reflecting its position along the learned trajectory.

Usage:
    python 02_run_ontrac.py \
        --metadata ontrac_input/metadata.csv \
        --output_dir results/ontrac \
        --device cuda:0 \
        --epochs 1000

References:
    Wang et al. ONTraC: https://github.com/gyyang23/ONTraC
"""

import argparse
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default ONTraC parameters (optimized for Visium HD)
DEFAULT_PARAMS = {
    'epochs': 1000,
    'batch_size': 5,
    'patience': 100,
    'min_delta': 0.001,
    'hidden_feats': 4,
    'k_neighbors': 50,
    'n_cpu': 8,
    'seed': 42,
    'lr': 0.03,
    'modularity_loss_weight': 0.3,
    'regularization_loss_weight': 0.1,
}


def run_ontrac(
    metadata_path: str,
    output_dir: str,
    device: str = 'cuda:0',
    **kwargs
):
    """
    Run ONTraC analysis.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV (from 01_prepare_ontrac_metadata.py)
    output_dir : str
        Output directory for ONTraC results
    device : str
        Compute device ('cuda:0', 'cuda:1', or 'cpu')
    **kwargs
        Override default ONTraC parameters
        
    Returns
    -------
    str
        Path to NT score output file
    """
    # Merge default params with overrides
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nn_dir = output_dir / 'NN'
    gnn_dir = output_dir / 'GNN'
    nt_dir = output_dir / 'NT'
    
    # Build ONTraC command
    cmd = [
        'ONTraC',
        '--meta-input', str(metadata_path),
        '--NN-dir', str(nn_dir),
        '--GNN-dir', str(gnn_dir),
        '--NT-dir', str(nt_dir),
        '--device', device,
        '--epochs', str(params['epochs']),
        '--batch-size', str(params['batch_size']),
        '--patience', str(params['patience']),
        '--min-delta', str(params['min_delta']),
        '--hidden-feats', str(params['hidden_feats']),
        '--k-neighbors', str(params['k_neighbors']),
        '--n-cpu', str(params['n_cpu']),
        '--seed', str(params['seed']),
        '--lr', str(params['lr']),
        '--modularity-loss-weight', str(params['modularity_loss_weight']),
        '--regularization-loss-weight', str(params['regularization_loss_weight']),
    ]
    
    logger.info(f"Running ONTraC with device: {device}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run ONTraC
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("ONTraC completed successfully")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ONTraC failed with return code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise
    
    # Check for output
    nt_score_path = nt_dir / 'NTScore.csv'
    if not nt_score_path.exists():
        raise FileNotFoundError(f"Expected NT score output not found: {nt_score_path}")
    
    logger.info(f"NT scores saved to {nt_score_path}")
    
    return str(nt_score_path)


def run_ontrac_python_api(
    metadata_path: str,
    output_dir: str,
    device: str = 'cuda:0',
    **kwargs
):
    """
    Run ONTraC using Python API (alternative to CLI).
    
    This provides more control and better error handling.
    """
    try:
        from ONTraC import ONTraC
        from ONTraC.model import ONTraCModel
    except ImportError:
        raise ImportError("ONTraC not installed. Run: pip install ONTraC")
    
    # Merge params
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run ONTraC
    logger.info(f"Initializing ONTraC model...")
    
    ontrac = ONTraC(
        meta_input=metadata_path,
        NN_dir=str(output_dir / 'NN'),
        GNN_dir=str(output_dir / 'GNN'),
        NT_dir=str(output_dir / 'NT'),
        device=device,
        **params
    )
    
    logger.info("Training ONTraC model...")
    ontrac.train()
    
    logger.info("Computing NT scores...")
    ontrac.NTScore()
    
    nt_score_path = output_dir / 'NT' / 'NTScore.csv'
    logger.info(f"NT scores saved to {nt_score_path}")
    
    return str(nt_score_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run ONTraC niche trajectory analysis'
    )
    parser.add_argument(
        '--metadata', required=True,
        help='Path to metadata CSV'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for ONTraC results'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Compute device (default: cuda:0)'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='Training epochs (default: 1000)'
    )
    parser.add_argument(
        '--k_neighbors', type=int, default=50,
        help='Number of spatial neighbors (default: 50)'
    )
    parser.add_argument(
        '--hidden_feats', type=int, default=4,
        help='GNN hidden features (default: 4)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=5,
        help='Batch size (default: 5)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--use_python_api', action='store_true',
        help='Use Python API instead of CLI'
    )
    
    args = parser.parse_args()
    
    run_func = run_ontrac_python_api if args.use_python_api else run_ontrac
    
    run_func(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        k_neighbors=args.k_neighbors,
        hidden_feats=args.hidden_feats,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
