"""
Utility functions for SpAN analysis pipeline.

This module provides core scoring and I/O utilities for:
- ONTraC NT score orientation
- Spatial gradient calculation
- SpAN score computation
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# NT Score Orientation
# =============================================================================

def orient_nt_scores(
    df: pd.DataFrame,
    nt_col: str = 'Cell_NTScore',
    niche_col: str = 'Niche',
    tumor_label: str = 'Tumor',
    lymphoid_label: str = 'Lymphoid'
) -> pd.DataFrame:
    """
    Orient ONTraC NT scores so Tumor=low (0) and Lymphoid=high (1).
    
    ONTraC's unsupervised trajectory direction varies by run/sample.
    This function standardizes direction based on mean niche scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NT scores and niche labels
    nt_col : str
        Column name for raw NT scores
    niche_col : str
        Column name for niche assignments
    tumor_label : str
        Label for tumor niche
    lymphoid_label : str
        Label for lymphoid/immune niche
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'NT_oriented' column added
        
    Example
    -------
    >>> df = orient_nt_scores(df, nt_col='Cell_NTScore', niche_col='Niche')
    >>> df['NT_oriented'].groupby(df['Niche']).mean()
    Niche
    Tumor       0.13
    Interface   0.59
    Lymphoid    0.61
    """
    df = df.copy()
    
    # Calculate mean NT score for anchor niches
    tumor_nt = df[df[niche_col] == tumor_label][nt_col].mean()
    lymphoid_nt = df[df[niche_col] == lymphoid_label][nt_col].mean()
    
    logger.info(f"Mean NT scores - Tumor: {tumor_nt:.3f}, Lymphoid: {lymphoid_nt:.3f}")
    
    # Flip if tumor has higher NT than lymphoid
    if tumor_nt > lymphoid_nt:
        df['NT_oriented'] = 1 - df[nt_col]
        logger.info("Flipped NT scores (original had Tumor > Lymphoid)")
    else:
        df['NT_oriented'] = df[nt_col]
        logger.info("NT scores already oriented correctly")
    
    return df


# =============================================================================
# Spatial Gradient Calculation
# =============================================================================

def calculate_spatial_gradient(
    coords: np.ndarray,
    niches: np.ndarray,
    tumor_label: str = 'Tumor',
    lymphoid_label: str = 'Lymphoid',
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate spatial gradient score based on physical distance to anchor niches.
    
    Uses KD-tree for efficient nearest-neighbor queries to find distance
    to closest Tumor and Lymphoid spots, then normalizes to [0, 1].
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n_spots, 2)
    niches : np.ndarray
        Niche labels for each spot
    tumor_label : str
        Label identifying tumor niche (anchored to 0)
    lymphoid_label : str
        Label identifying lymphoid niche (anchored to 1)
    eps : float
        Small constant to avoid division by zero
        
    Returns
    -------
    spatial_gradient : np.ndarray
        Gradient score [0=Tumor, 1=Lymphoid] for each spot
    dist_to_tumor : np.ndarray
        Euclidean distance to nearest tumor spot
    dist_to_lymphoid : np.ndarray
        Euclidean distance to nearest lymphoid spot
        
    Notes
    -----
    Formula: spatial_gradient = d_tumor / (d_tumor + d_lymphoid)
    
    Anchor spots are forced to exact values:
    - Tumor spots → 0.0
    - Lymphoid spots → 1.0
    
    Example
    -------
    >>> coords = adata.obsm['spatial']
    >>> niches = adata.obs['Niche'].values
    >>> sg, d_t, d_l = calculate_spatial_gradient(coords, niches)
    """
    # Identify anchor spots
    tumor_mask = niches == tumor_label
    lymphoid_mask = niches == lymphoid_label
    
    n_tumor = tumor_mask.sum()
    n_lymphoid = lymphoid_mask.sum()
    
    if n_tumor == 0:
        raise ValueError(f"No spots found with niche label '{tumor_label}'")
    if n_lymphoid == 0:
        raise ValueError(f"No spots found with niche label '{lymphoid_label}'")
    
    logger.info(f"Anchor spots - Tumor: {n_tumor}, Lymphoid: {n_lymphoid}")
    logger.info(f"Lymphoid:Tumor ratio: {n_lymphoid/n_tumor:.2f}")
    
    # Get anchor coordinates
    tumor_coords = coords[tumor_mask]
    lymphoid_coords = coords[lymphoid_mask]
    
    # Build KD-trees for fast nearest-neighbor search
    tumor_tree = cKDTree(tumor_coords)
    lymphoid_tree = cKDTree(lymphoid_coords)
    
    # Query distances for all spots
    dist_to_tumor, _ = tumor_tree.query(coords, k=1)
    dist_to_lymphoid, _ = lymphoid_tree.query(coords, k=1)
    
    # Calculate spatial gradient (normalized distance ratio)
    spatial_gradient = dist_to_tumor / (dist_to_tumor + dist_to_lymphoid + eps)
    
    # Force exact values for anchor niches
    spatial_gradient[tumor_mask] = 0.0
    spatial_gradient[lymphoid_mask] = 1.0
    
    logger.info(f"Spatial gradient range: [{spatial_gradient.min():.3f}, {spatial_gradient.max():.3f}]")
    
    return spatial_gradient, dist_to_tumor, dist_to_lymphoid


def calculate_spatial_gradient_geometric(
    coords: np.ndarray,
    niches: np.ndarray,
    tumor_label: str = 'Tumor',
    lymphoid_label: str = 'Lymphoid',
    eps: float = 1e-10
) -> np.ndarray:
    """
    Alternative spatial gradient using geometric mean normalization.
    
    Formula: sqrt(d_tumor * d_lymphoid) based normalization.
    Note: Highly correlated with arithmetic version (r > 0.98),
    but compresses extreme values toward 0.5.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates
    niches : np.ndarray
        Niche labels
    tumor_label : str
        Tumor niche label
    lymphoid_label : str
        Lymphoid niche label
    eps : float
        Small constant for numerical stability
        
    Returns
    -------
    np.ndarray
        Geometric-mean normalized spatial gradient
    """
    tumor_mask = niches == tumor_label
    lymphoid_mask = niches == lymphoid_label
    
    tumor_tree = cKDTree(coords[tumor_mask])
    lymphoid_tree = cKDTree(coords[lymphoid_mask])
    
    dist_to_tumor, _ = tumor_tree.query(coords, k=1)
    dist_to_lymphoid, _ = lymphoid_tree.query(coords, k=1)
    
    # Geometric mean normalization
    geom_mean = np.sqrt(dist_to_tumor * dist_to_lymphoid + eps)
    spatial_gradient = dist_to_tumor / (geom_mean + dist_to_tumor + eps)
    
    # Force anchor values
    spatial_gradient[tumor_mask] = 0.0
    spatial_gradient[lymphoid_mask] = 1.0
    
    return spatial_gradient


# =============================================================================
# SpAN Score Computation
# =============================================================================

def compute_span_score(
    nt_oriented: np.ndarray,
    spatial_gradient: np.ndarray,
    nt_weight: float = 0.5
) -> np.ndarray:
    """
    Compute SpAN (Spatial-Anchored Niche trajectory) score.
    
    Combines ONTraC NT scores (molecular/compositional) with spatial
    gradient scores (physical distance) for robust trajectory positioning.
    
    Parameters
    ----------
    nt_oriented : np.ndarray
        Oriented NT scores [0=Tumor, 1=Lymphoid]
    spatial_gradient : np.ndarray
        Spatial gradient scores [0=Tumor, 1=Lymphoid]
    nt_weight : float
        Weight for NT component (default 0.5 for equal weighting)
        
    Returns
    -------
    np.ndarray
        SpAN score for each spot
        
    Notes
    -----
    Default formula: SpAN = (NT_oriented + spatial_gradient) / 2
    
    Equal weighting rationale:
    - NT captures compositional/molecular similarity
    - Spatial gradient captures physical tissue architecture
    - Both contribute complementary information
    - No a priori reason to favor one dimension
    """
    sg_weight = 1.0 - nt_weight
    span = (nt_weight * nt_oriented) + (sg_weight * spatial_gradient)
    return span


# =============================================================================
# Statistical Utilities
# =============================================================================

def compare_niche_scores(
    df: pd.DataFrame,
    score_col: str,
    niche_col: str = 'Niche',
    niches: List[str] = ['Tumor', 'Interface', 'Lymphoid']
) -> Dict:
    """
    Calculate statistics and pairwise comparisons for scores by niche.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores and niche labels
    score_col : str
        Column containing scores to analyze
    niche_col : str
        Column containing niche labels
    niches : List[str]
        Niche labels to compare (in order)
        
    Returns
    -------
    Dict
        Statistics including means, stds, and pairwise p-values
    """
    from scipy.stats import mannwhitneyu
    
    results = {
        'means': {},
        'stds': {},
        'pairwise_pvalues': {}
    }
    
    # Calculate summary statistics
    for niche in niches:
        scores = df[df[niche_col] == niche][score_col]
        results['means'][niche] = scores.mean()
        results['stds'][niche] = scores.std()
    
    # Pairwise comparisons
    for i, n1 in enumerate(niches):
        for n2 in niches[i+1:]:
            s1 = df[df[niche_col] == n1][score_col]
            s2 = df[df[niche_col] == n2][score_col]
            _, pval = mannwhitneyu(s1, s2, alternative='two-sided')
            results['pairwise_pvalues'][f'{n1}_vs_{n2}'] = pval
    
    return results


# =============================================================================
# I/O Utilities
# =============================================================================

def load_ontrac_results(
    nt_score_path: str,
    sample_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ONTraC NT score results.
    
    Parameters
    ----------
    nt_score_path : str
        Path to NTScore.csv from ONTraC output
    sample_name : str, optional
        Sample identifier to add as column
        
    Returns
    -------
    pd.DataFrame
        NT scores with Cell_ID index
    """
    df = pd.read_csv(nt_score_path)
    
    if sample_name:
        df['Sample'] = sample_name
    
    return df


def save_span_results(
    df: pd.DataFrame,
    output_path: str,
    columns: Optional[List[str]] = None
):
    """
    Save SpAN analysis results to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    output_path : str
        Output file path
    columns : List[str], optional
        Specific columns to save (default: all)
    """
    if columns:
        df = df[columns]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path} ({len(df)} spots)")


def validate_inputs(
    coords: np.ndarray,
    niches: np.ndarray,
    nt_scores: Optional[np.ndarray] = None
):
    """
    Validate input arrays for consistency.
    
    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates
    niches : np.ndarray
        Niche labels
    nt_scores : np.ndarray, optional
        NT scores if available
        
    Raises
    ------
    ValueError
        If array dimensions don't match
    """
    n_spots = len(niches)
    
    if coords.shape[0] != n_spots:
        raise ValueError(f"Coords ({coords.shape[0]}) and niches ({n_spots}) length mismatch")
    
    if coords.shape[1] != 2:
        raise ValueError(f"Coords should be 2D, got shape {coords.shape}")
    
    if nt_scores is not None and len(nt_scores) != n_spots:
        raise ValueError(f"NT scores ({len(nt_scores)}) and niches ({n_spots}) length mismatch")
    
    logger.info(f"Validated inputs: {n_spots} spots")
