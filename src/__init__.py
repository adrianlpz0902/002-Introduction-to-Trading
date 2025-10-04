"""
Trading Strategy Project 002
A modular algorithmic trading system with multi-indicator confirmation.
"""

__version__ = "1.0.0"
__author__ = "Trading Strategy Team"

from . import data_loader
from . import indicators
from . import signals
from . import backtest
from . import portfolio
from . import metrics
from . import optimizer
from . import visualization

__all__ = [
    'data_loader',
    'indicators',
    'signals',
    'backtest',
    'portfolio',
    'metrics',
    'optimizer',
    'visualization'
]
