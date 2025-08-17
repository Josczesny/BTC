"""
Sistema de Aprendizado Contínuo 2025
Implementa as técnicas mais avançadas baseadas nas pesquisas de 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam

from utils.logger import setup_logger

logger = setup_logger("continuous-learning")

class ContinuousLearningSystem:
    """
    Sistema de Aprendizado Contínuo para Trading de Bitcoin 2025
    """
    
    def __init__(self, models_dir="models", learning_rate=0.001):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.learning_rate = learning_rate
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        
        # Configurações
        self.update_frequency = 100
        self.trade_counter = 0
        self.min_samples_for_update = 50
        
        # Históricos
        self.sharpe_history = []
        self.accuracy_history = []
        self.profit_history = []
        
        # Sistema IMCA
        self.imca_weights = np.array([])
        
        logger.info("Sistema de Aprendizado Contínuo 2025 inicializado")
        
    def add_model(self, name: str, model: Any, initial_weight: float = 1.0):
        """Adiciona modelo ao sistema"""
        self.models[name] = model
        self.model_weights[name] = initial_weight
        self.performance_history[name] = []
        
        # Atualiza arrays IMCA
        self._update_imca_arrays()
        
        logger.info(f"Modelo {name} adicionado")
        
    def _update_imca_arrays(self):
        """Atualiza arrays do sistema IMCA"""
        n_models = len(self.models)
        if n_models > 0:
            if len(self.imca_weights) != n_models:
                self.imca_weights = np.ones(n_models) / n_models
                
    def register_trade_feedback(self, trade_data: Dict[str, Any]) -> None:
        """Registra feedback de trade executado"""
        try:
            self.trade_counter += 1
            
            # Extrai dados
            profit = trade_data.get('profit', 0)
            signals = trade_data.get('signals', {})
            market_data = trade_data.get('market_data', {})
            
            success = profit > 0
            
            # Atualiza histórico
            self.profit_history.append(profit)
            
            # Atualiza pesos usando IMCA
            self._update_model_weights_imca(signals, success, profit)
            
            # Retreina se necessário
            if self.trade_counter % self.update_frequency == 0:
                self._trigger_model_retraining()
                
            # Salva estado periodicamente
            if self.trade_counter % (self.update_frequency * 2) == 0:
                self.save_state()
                
            logger.info(f"Feedback registrado - Trade {self.trade_counter}, Profit: {profit:.4f}")
            
        except Exception as e:
            logger.error(f"Erro ao registrar feedback: {e}")
            
    def _update_model_weights_imca(self, signals: Dict, success: bool, profit: float):
        """Atualiza pesos usando algoritmo IMCA"""
        try:
            if not self.models or not signals:
                return
                
            errors = []
            model_names = list(self.models.keys())
            
            for name in model_names:
                model_signal = signals.get(name, 0.5)
                
                # Converte sinal para numérico
                if isinstance(model_signal, str):
                    signal_map = {'BUY': 1.0, 'SELL': 0.0, 'HOLD': 0.5}
                    predicted = signal_map.get(model_signal.upper(), 0.5)
                else:
                    predicted = float(model_signal)
                    
                # Valor real baseado no sucesso
                actual = 1.0 if success else 0.0
                
                # Calcula erro
                error = abs(predicted - actual)
                errors.append(error)
                
                # Atualiza histórico
                self.performance_history[name].append({
                    'predicted': predicted,
                    'actual': actual,
                    'error': error,
                    'profit': profit,
                    'timestamp': datetime.now()
                })
                
                # Limita histórico
                if len(self.performance_history[name]) > 1000:
                    self.performance_history[name] = self.performance_history[name][-1000:]
                    
            # Atualiza pesos IMCA
            if errors and len(errors) == len(self.imca_weights):
                errors = np.array(errors)
                
                # Novos pesos (inverso do erro)
                new_weights = 1.0 / (errors + 1e-8)
                new_weights = new_weights / np.sum(new_weights)
                
                # Atualização suave
                alpha = self.learning_rate
                self.imca_weights = (1 - alpha) * self.imca_weights + alpha * new_weights
                
                # Atualiza pesos individuais
                for i, name in enumerate(model_names):
                    self.model_weights[name] = self.imca_weights[i]
                    
                logger.debug(f"Pesos IMCA atualizados")
                
        except Exception as e:
            logger.error(f"Erro ao atualizar pesos IMCA: {e}")
            
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calcula métricas de risco"""
        try:
            if len(self.profit_history) < 10:
                return {
                    'volatility': 0.1,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
                
            returns = np.array(self.profit_history[-100:])
            
            volatility = np.std(returns)
            mean_return = np.mean(returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
            
            # Maximum Drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max)
            max_drawdown = np.min(drawdown)
            
            metrics = {
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            }
            
            self.sharpe_history.append(sharpe_ratio)
            if len(self.sharpe_history) > 100:
                self.sharpe_history = self.sharpe_history[-100:]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}
            
    def _trigger_model_retraining(self):
        """Dispara retreinamento dos modelos"""
        try:
            logger.info(f"Iniciando retreinamento após {self.trade_counter} trades")
            
            # Verifica dados suficientes
            total_samples = sum(len(hist) for hist in self.performance_history.values())
            if total_samples < self.min_samples_for_update:
                logger.warning("[WARNING] Dados insuficientes para retreinamento")
                return
                
            # Calcula performance atual
            current_performance = self._calculate_current_performance()
            
            # Identifica modelos para retreinar
            models_to_retrain = self._identify_models_for_retraining(current_performance)
            
            # Executa retreinamento
            for model_name in models_to_retrain:
                self._retrain_model(model_name)
                
            logger.info(f"Retreinamento concluído para {len(models_to_retrain)} modelos")
            
        except Exception as e:
            logger.error(f"Erro no retreinamento: {e}")
            
    def _calculate_current_performance(self) -> Dict[str, float]:
        """Calcula performance atual de cada modelo"""
        performance = {}
        
        for name, history in self.performance_history.items():
            if len(history) >= 10:
                recent_errors = [h['error'] for h in history[-50:]]
                recent_profits = [h['profit'] for h in history[-50:]]
                
                avg_error = np.mean(recent_errors)
                avg_profit = np.mean(recent_profits)
                
                # Score combinado
                score = (1.0 - avg_error) + avg_profit * 0.1
                performance[name] = score
            else:
                performance[name] = 0.5
                
        return performance
        
    def _identify_models_for_retraining(self, performance: Dict[str, float]) -> List[str]:
        """Identifica modelos que precisam retreinamento"""
        models_to_retrain = []
        adaptation_threshold = 0.8
        
        for name, score in performance.items():
            if score < adaptation_threshold:
                models_to_retrain.append(name)
                logger.info(f"Modelo {name} marcado para retreinamento (score: {score:.3f})")
                
        return models_to_retrain
        
    def _retrain_model(self, model_name: str):
        """Retreina modelo específico"""
        try:
            if model_name not in self.models:
                return
                
            model = self.models[model_name]
            history = self.performance_history[model_name]
            
            if len(history) < self.min_samples_for_update:
                return
                
            # Prepara dados
            X, y = self._prepare_training_data(history)
            
            if len(X) == 0:
                return
                
            # Retreina baseado no tipo
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X, y)
            elif hasattr(model, 'fit'):
                model.fit(X, y)
            elif hasattr(model, 'compile'):
                # TEMPORARIAMENTE DESABILITADO: LSTM retreinamento precisa de correção de shape
                logger.warning(f"[SKIP] Retreinamento de {model_name} temporariamente desabilitado")
                return
                # Para LSTM, precisa reshape para (samples, timesteps, features)
                if len(X.shape) == 2 and X.shape[1] > 8:
                    # Limita para 8 features
                    X = X[:, :8]
                # Reshape para sequências de 20 timesteps
                n_samples = len(X) - 20 + 1
                if n_samples > 0:
                    X_seq = []
                    y_seq = []
                    for i in range(n_samples):
                        X_seq.append(X[i:i+20])
                        y_seq.append(y[i+19])
                    X = np.array(X_seq)
                    y = np.array(y_seq)
                model.fit(X, y, epochs=10, verbose=0, batch_size=32)
                
            logger.info(f"Modelo {model_name} retreinado com {len(X)} amostras")
            
        except Exception as e:
            logger.error(f"Erro ao retreinar modelo {model_name}: {e}")
            
    def _prepare_training_data(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para retreinamento"""
        try:
            if len(history) < 2:
                return np.array([]), np.array([])
                
            X = []
            y = []
            
            for record in history[-100:]:
                feature = [record['predicted'], record['profit']]
                target = record['actual']
                
                X.append(feature)
                y.append(target)
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            return np.array([]), np.array([])
            
    def get_model_predictions(self, market_data: Dict) -> Dict[str, Any]:
        """Obtém predições com pesos atualizados"""
        predictions = {}
        
        try:
            for name, model in self.models.items():
                weight = self.model_weights.get(name, 1.0)
                
                try:
                    if hasattr(model, 'predict_next_move'):
                        pred = model.predict_next_move(market_data)
                    elif hasattr(model, 'predict'):
                        pred = model.predict([list(market_data.values())])
                    else:
                        pred = 0.5
                        
                    predictions[name] = {
                        'prediction': pred,
                        'weight': weight,
                        'confidence': self._calculate_model_confidence(name)
                    }
                    
                except Exception as e:
                    logger.warning(f"Erro na predição do modelo {name}: {e}")
                    predictions[name] = {
                        'prediction': 0.5,
                        'weight': weight * 0.5,
                        'confidence': 0.1
                    }
                    
        except Exception as e:
            logger.error(f"Erro ao obter predições: {e}")
            
        return predictions
        
    def _calculate_model_confidence(self, model_name: str) -> float:
        """Calcula confiança do modelo"""
        try:
            if model_name not in self.performance_history:
                return 0.5
                
            history = self.performance_history[model_name]
            if len(history) < 10:
                return 0.5
                
            # Accuracy nos últimos 20 trades
            recent_predictions = history[-20:]
            correct = sum(1 for h in recent_predictions if abs(h['predicted'] - h['actual']) < 0.3)
            confidence = correct / len(recent_predictions)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Erro ao calcular confiança: {e}")
            return 0.5
            
    def save_state(self):
        """Salva estado do sistema"""
        try:
            state = {
                'model_weights': self.model_weights,
                'imca_weights': self.imca_weights.tolist(),
                'trade_counter': self.trade_counter,
                'sharpe_history': self.sharpe_history,
                'profit_history': self.profit_history[-1000:],
                'timestamp': datetime.now().isoformat()
            }
            
            state_file = self.models_dir / 'continuous_learning_state.json'
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info("Estado salvo")
            
        except Exception as e:
            logger.error(f"Erro ao salvar: {e}")
            
    def load_state(self):
        """Carrega estado do sistema"""
        try:
            state_file = self.models_dir / 'continuous_learning_state.json'
            if not state_file.exists():
                logger.info("Arquivo de estado não encontrado")
                return
                
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.model_weights = state.get('model_weights', {})
            imca_weights = state.get('imca_weights', [])
            if imca_weights:
                self.imca_weights = np.array(imca_weights)
                
            self.trade_counter = state.get('trade_counter', 0)
            self.sharpe_history = state.get('sharpe_history', [])
            self.profit_history = state.get('profit_history', [])
            
            logger.info(f"Estado carregado - {self.trade_counter} trades")
            
        except Exception as e:
            logger.error(f"Erro ao carregar: {e}")
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do sistema"""
        try:
            risk_metrics = self._calculate_risk_metrics()
            
            metrics = {
                'trades_processed': self.trade_counter,
                'active_models': len(self.models),
                'current_weights': self.model_weights.copy(),
                'imca_weights': self.imca_weights.tolist() if len(self.imca_weights) > 0 else [],
                'risk_metrics': risk_metrics,
                'recent_sharpe': np.mean(self.sharpe_history[-10:]) if self.sharpe_history else 0.0,
                'total_profit': sum(self.profit_history) if self.profit_history else 0.0,
                'win_rate': len([p for p in self.profit_history if p > 0]) / max(len(self.profit_history), 1)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {e}")
            return {} 