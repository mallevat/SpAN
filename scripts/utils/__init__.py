"""
SpAN Analysis Utilities

Core functions for:
- ONTraC NT score orientation
- Spatial gradient calculation
- SpAN score computation
- Statistical validation
"""

from .scoring_utils import (
    orient_nt_scores,
    calculate_spatial_gradient,
    calculate_spatial_gradient_geometric,
    compute_span_score,
    compare_niche_scores,
    load_ontrac_results,
    save_span_results,
    validate_inputs
)

__all__ = [
    'orient_nt_scores',
    'calculate_spatial_gradient',
    'calculate_spatial_gradient_geometric',
    'compute_span_score',
    'compare_niche_scores',
    'load_ontrac_results',
    'save_span_results',
    'validate_inputs'
]
