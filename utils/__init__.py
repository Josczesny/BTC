#!/usr/bin/env python3
"""
MÓDULO UTILITÁRIOS
==================

Utilitários do sistema de trading.
"""

from .terminal_colors import TerminalColors
from .time_sync import sync_system_time, is_admin, run_as_admin

# Carregamento automático do .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Se não houver python-dotenv, ignora (mas recomenda instalar para produção)

__all__ = [
    'TerminalColors',
    'sync_system_time',
    'is_admin',
    'run_as_admin'
] 