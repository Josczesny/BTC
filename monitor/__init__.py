 # monitor/__init__.py
"""
Módulo de Monitoramento e Performance

Este módulo contém:
- Logging de performance de trades
- Métricas de lucro/prejuízo
- Relatórios de desempenho
- Alertas e notificações
"""

from .performance_logger import PerformanceLogger

__all__ = ['PerformanceLogger']