"""Decision-focused learning for portfolio optimisation (AI in Finance, Project 4)."""

from .layers import PortfolioLayer, mv_closed_form, solve_box_qp
from .models import SharedLinear, SharedMLP, build_model
from .train import Split, TrainConfig, fit_decision, fit_mse

__all__ = [
    "PortfolioLayer",
    "solve_box_qp",
    "mv_closed_form",
    "SharedLinear",
    "SharedMLP",
    "build_model",
    "Split",
    "TrainConfig",
    "fit_mse",
    "fit_decision",
]
__version__ = "1.0.0"
