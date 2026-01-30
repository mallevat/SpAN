#!/usr/bin/env python3
"""
visualization.py

Publication-ready visualizations for SpAN analysis.

Generates:
1. Spatial maps colored by SpAN score
2. Boxplots comparing scores across niches
3. Rolling Gaussian trajectory plots
4. Cross-sample validation heatmaps

Usage:
    python analysis/visualization.py \
        --span_results results/combined_span_scores.csv \
        --output_dir figures/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, gaussian_kde
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # Editable text in PDFs
    'ps.fonttype': 42
})

# Color schemes
NICHE_COLORS = {
    'Tumor': '#2166AC',      # Blue
    'Interface': '#7B3294',  # Purple
    'Lymphoid': '#D53E4F'    # Red/Orange
}

# Custom colormap for SpAN (Tumor=Blue, Interface=Purple, Lymphoid=Red)
SPAN_CMAP = LinearSegmentedColormap.from_list(
    'SpAN',
    ['#2166AC', '#7B3294', '#D53E4F']
)


def plot_spatial_span(
    df: pd.DataFrame,
    sample: str,
    output_path: str,
    score_col: str = 'SpAN',
    figsize: Tuple[float, float] = (8, 8),
    point_size: float = 1.0,
    show_colorbar: bool = True
):
    """
    Plot spatial map colored by SpAN score.
    
    Parameters
    ----------
    df : pd.DataFrame
        SpAN results with x, y coordinates
    sample : str
        Sample name to plot
    output_path : str
        Output file path
    score_col : str
        Column to use for coloring
    figsize : tuple
        Figure size
    point_size : float
        Point size multiplier
    show_colorbar : bool
        Show colorbar
    """
    sample_df = df[df['Sample'] == sample].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        sample_df['x'],
        sample_df['y'],
        c=sample_df[score_col],
        cmap=SPAN_CMAP,
        s=point_size,
        vmin=0, vmax=1,
        rasterized=True
    )
    
    ax.set_aspect('equal')
    ax.set_xlabel('X coordinate (μm)')
    ax.set_ylabel('Y coordinate (μm)')
    ax.set_title(f'{sample} - {score_col} Score')
    
    # Invert y-axis for standard histology orientation
    ax.invert_yaxis()
    
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(score_col)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Tumor', 'Interface', 'Lymphoid'])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()
    
    logger.info(f"Saved spatial plot to {output_path}")


def plot_niche_boxplots(
    df: pd.DataFrame,
    output_path: str,
    scores: List[str] = ['NT_oriented', 'spatial_gradient', 'SpAN'],
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot boxplots comparing scores across niches.
    
    Parameters
    ----------
    df : pd.DataFrame
        SpAN results
    output_path : str
        Output file path
    scores : list
        Score columns to plot
    figsize : tuple
        Figure size
    """
    n_scores = len(scores)
    fig, axes = plt.subplots(1, n_scores, figsize=figsize)
    
    if n_scores == 1:
        axes = [axes]
    
    niche_order = ['Tumor', 'Interface', 'Lymphoid']
    
    for ax, score in zip(axes, scores):
        # Boxplot
        sns.boxplot(
            data=df,
            x='Niche',
            y=score,
            order=niche_order,
            palette=NICHE_COLORS,
            ax=ax,
            width=0.6,
            fliersize=2
        )
        
        ax.set_xlabel('')
        ax.set_ylabel(score.replace('_', ' ').title())
        
        # Add statistical annotations
        y_max = df[score].max()
        
        # Tumor vs Interface
        t_scores = df[df['Niche'] == 'Tumor'][score]
        i_scores = df[df['Niche'] == 'Interface'][score]
        _, p1 = mannwhitneyu(t_scores, i_scores)
        
        # Interface vs Lymphoid
        l_scores = df[df['Niche'] == 'Lymphoid'][score]
        _, p2 = mannwhitneyu(i_scores, l_scores)
        
        # Add significance stars
        def sig_stars(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        ax.text(0.5, y_max * 1.05, sig_stars(p1), ha='center', fontsize=10)
        ax.text(1.5, y_max * 1.05, sig_stars(p2), ha='center', fontsize=10)
        
        ax.set_ylim(0, y_max * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()
    
    logger.info(f"Saved boxplots to {output_path}")


def plot_rolling_gaussian(
    df: pd.DataFrame,
    output_path: str,
    score_col: str = 'SpAN',
    value_cols: List[str] = None,
    sigma: float = 0.05,
    n_points: int = 150,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Plot rolling Gaussian window analysis along trajectory.
    
    Parameters
    ----------
    df : pd.DataFrame
        SpAN results
    output_path : str
        Output file path
    score_col : str
        Trajectory score column
    value_cols : list
        Columns to analyze along trajectory
    sigma : float
        Gaussian window width
    n_points : int
        Number of evaluation points
    figsize : tuple
        Figure size
    """
    if value_cols is None:
        value_cols = ['NT_oriented', 'spatial_gradient']
    
    # Create evaluation points
    x_eval = np.linspace(0, 1, n_points)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(value_cols)))
    
    for col, color in zip(value_cols, colors):
        rolling_means = []
        rolling_cis = []
        
        for x in x_eval:
            # Gaussian weights
            weights = np.exp(-((df[score_col] - x) ** 2) / (2 * sigma ** 2))
            weights = weights / weights.sum()
            
            # Weighted mean
            mean = (df[col] * weights).sum()
            rolling_means.append(mean)
            
            # Bootstrap CI (simplified)
            var = ((df[col] - mean) ** 2 * weights).sum()
            se = np.sqrt(var / len(weights))
            rolling_cis.append(1.96 * se)
        
        rolling_means = np.array(rolling_means)
        rolling_cis = np.array(rolling_cis)
        
        ax.plot(x_eval, rolling_means, color=color, linewidth=2, label=col)
        ax.fill_between(
            x_eval,
            rolling_means - rolling_cis,
            rolling_means + rolling_cis,
            color=color, alpha=0.2
        )
    
    # Add niche regions
    ax.axvspan(0, 0.2, alpha=0.1, color=NICHE_COLORS['Tumor'], label='Tumor zone')
    ax.axvspan(0.4, 0.8, alpha=0.1, color=NICHE_COLORS['Interface'], label='Interface zone')
    ax.axvspan(0.8, 1.0, alpha=0.1, color=NICHE_COLORS['Lymphoid'], label='Lymphoid zone')
    
    ax.set_xlabel(f'{score_col} Score')
    ax.set_ylabel('Value')
    ax.set_xlim(0, 1)
    ax.legend(loc='best')
    ax.set_title(f'Rolling Gaussian Analysis (σ={sigma})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()
    
    logger.info(f"Saved rolling Gaussian plot to {output_path}")


def plot_sample_comparison(
    df: pd.DataFrame,
    output_path: str,
    score_col: str = 'SpAN',
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Plot comparison across samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        SpAN results
    output_path : str
        Output file path
    score_col : str
        Score column to plot
    figsize : tuple
        Figure size
    """
    samples = df['Sample'].unique()
    n_samples = len(samples)
    
    fig, axes = plt.subplots(1, n_samples, figsize=figsize, sharey=True)
    
    if n_samples == 1:
        axes = [axes]
    
    niche_order = ['Tumor', 'Interface', 'Lymphoid']
    
    for ax, sample in zip(axes, samples):
        sample_df = df[df['Sample'] == sample]
        
        sns.boxplot(
            data=sample_df,
            x='Niche',
            y=score_col,
            order=niche_order,
            palette=NICHE_COLORS,
            ax=ax,
            width=0.6,
            fliersize=1
        )
        
        ax.set_xlabel('')
        ax.set_title(sample)
        
        if ax != axes[0]:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(score_col)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()
    
    logger.info(f"Saved sample comparison to {output_path}")


def generate_all_figures(
    span_results_path: str,
    output_dir: str
):
    """
    Generate all publication figures.
    
    Parameters
    ----------
    span_results_path : str
        Path to combined SpAN results CSV
    output_dir : str
        Output directory for figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading SpAN results from {span_results_path}")
    df = pd.read_csv(span_results_path)
    
    # 1. Boxplots across niches
    plot_niche_boxplots(
        df,
        str(output_dir / 'Fig_SpAN_Boxplots.png'),
        scores=['NT_oriented', 'spatial_gradient', 'SpAN']
    )
    
    # 2. Sample comparison
    plot_sample_comparison(
        df,
        str(output_dir / 'Fig_Sample_Comparison.png'),
        score_col='SpAN'
    )
    
    # 3. Rolling Gaussian analysis
    plot_rolling_gaussian(
        df,
        str(output_dir / 'Fig_Rolling_Gaussian.png'),
        score_col='SpAN'
    )
    
    # 4. Spatial maps for each sample
    samples = df['Sample'].unique()
    for sample in samples:
        plot_spatial_span(
            df,
            sample,
            str(output_dir / f'Fig_Spatial_{sample}.png'),
            score_col='SpAN'
        )
    
    logger.info(f"All figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for SpAN analysis'
    )
    parser.add_argument(
        '--span_results', required=True,
        help='Path to combined SpAN results CSV'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    generate_all_figures(
        span_results_path=args.span_results,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
