# models/__init__.py
"""
Módulo de Modelos Preditivos

Este módulo contém:
- Treinamento de modelos de IA
- Modelos LSTM, XGBoost, Prophet
- Gestão de versões de modelos
- Avaliação de performance
"""

from .train_predictor import PricePredictorTrainer

__all__ = ['PricePredictorTrainer']