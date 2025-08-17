 # executors/__init__.py
"""
Módulo de Execução de Ordens

Este módulo contém:
- Integração com APIs de exchanges
- Execução de ordens de compra/venda
- Gestão de conectividade e rate limits
"""

from .exchange_api import ExchangeAPI

__all__ = ['ExchangeAPI']