#!/usr/bin/env python3
"""
MÓDULO CORE
===========

Componentes principais do sistema de trading.
"""

from .api_manager import APIManager
from .model_manager import ModelManager
from .signal_processor import SignalProcessor

__all__ = [
    'APIManager',
    'ModelManager',
    'SignalProcessor'
] 