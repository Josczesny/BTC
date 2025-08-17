#!/usr/bin/env python3
"""
SISTEMA CENTRAL DE ENSEMBLE
===========================

Consolida TODOS os sistemas de ensemble em um sistema centralizado:
- Integra TODOS os modelos e agentes
- Pesos adaptativos inteligentes
- Performance tracking centralizado
- Confiança calculation unificado
- Elimina redundâncias de ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CentralEnsembleSystem:
    """
    Sistema centralizado de ensemble
    - Integra TODOS os modelos e agentes
    - Pesos adaptativos inteligentes
    - Performance tracking centralizado
    - Confiança calculation unificado
    """
    
    def __init__(self):
        self.model_registry = {}
        self.performance_tracker = {}
        self.weight_optimizer = {}
        self.ensemble_history = []
        self.market_regime_detector = None
        
        # Configurações
        self.min_confidence_threshold = 0.3
        self.max_confidence_threshold = 0.9
        self.performance_window = 100
        self.weight_update_frequency = 50
        
        print("🚀 Central Ensemble System inicializado")
    
    def register_model(self, model_name: str, model_type: str = 'unknown', 
                      initial_weight: float = 1.0, initial_confidence: float = 0.5):
        """
        Registra um modelo no sistema central
        
        Args:
            model_name: Nome único do modelo
            model_type: Tipo do modelo (prediction, technical, sentiment, etc.)
            initial_weight: Peso inicial
            initial_confidence: Confiança inicial
        """
        self.model_registry[model_name] = {
            'type': model_type,
            'weight': initial_weight,
            'confidence': initial_confidence,
            'performance_history': [],
            'last_update': datetime.now(),
            'total_predictions': 0,
            'correct_predictions': 0,
            'profit_history': []
        }
        
        print(f"📝 Modelo registrado: {model_name} ({model_type})")
    
    def get_ensemble_prediction(self, signals: Dict[str, float], 
                              market_conditions: Optional[Dict[str, Any]] = None,
                              model_confidences: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Retorna predição ensemble unificada
        
        Args:
            signals: Dicionário com sinais dos modelos
            market_conditions: Condições de mercado
            model_confidences: Confianças dos modelos
            
        Returns:
            Tuple[consensus, confidence]
        """
        try:
            if not signals:
                return 0.5, 0.1
            
            # ===== 1. PESOS BASEADOS EM PERFORMANCE HISTÓRICA =====
            performance_weights = self._calculate_performance_weights(signals.keys())
            
            # ===== 2. PESOS BASEADOS EM REGIME DE MERCADO =====
            regime_weights = self._adjust_weights_for_regime(signals.keys(), market_conditions)
            
            # ===== 3. PESOS BASEADOS EM CONFIANÇA DOS MODELOS =====
            confidence_weights = self._calculate_confidence_weights(signals, model_confidences)
            
            # ===== 4. COMBINAÇÃO PONDERADA =====
            final_weights = self._combine_weights(performance_weights, regime_weights, confidence_weights)
            
            # ===== 5. CALCULA CONSENSO PONDERADO =====
            consensus = self._calculate_weighted_consensus(signals, final_weights)
            
            # ===== 6. CALCULA CONFIANÇA DO ENSEMBLE =====
            confidence = self._calculate_ensemble_confidence(signals, final_weights, market_conditions)
            
            # ===== 7. ATUALIZA HISTÓRICOS =====
            self._update_ensemble_history(signals, consensus, confidence, final_weights)
            
            return consensus, confidence
            
        except Exception as e:
            print(f"❌ Erro no Central Ensemble System: {e}")
            return 0.5, 0.1
    
    def update_model_performance(self, model_name: str, prediction: float, 
                               actual_outcome: float, profit: float = 0.0):
        """
        Atualiza performance de um modelo específico
        
        Args:
            model_name: Nome do modelo
            prediction: Predição feita
            actual_outcome: Resultado real
            profit: Lucro/prejuízo gerado
        """
        if model_name not in self.model_registry:
            return
        
        model_data = self.model_registry[model_name]
        
        # Calcula acurácia
        prediction_direction = 1 if prediction > 0.5 else 0
        actual_direction = 1 if actual_outcome > 0.5 else 0
        is_correct = prediction_direction == actual_direction
        
        # Atualiza estatísticas
        model_data['total_predictions'] += 1
        if is_correct:
            model_data['correct_predictions'] += 1
        
        # Calcula acurácia
        accuracy = model_data['correct_predictions'] / model_data['total_predictions']
        
        # Adiciona à história de performance
        performance_entry = {
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'profit': profit,
            'prediction': prediction,
            'actual': actual_outcome
        }
        
        model_data['performance_history'].append(performance_entry)
        model_data['profit_history'].append(profit)
        
        # Mantém apenas histórico recente
        if len(model_data['performance_history']) > self.performance_window:
            model_data['performance_history'] = model_data['performance_history'][-self.performance_window:]
            model_data['profit_history'] = model_data['profit_history'][-self.performance_window:]
        
        # Atualiza peso baseado na performance
        self._update_model_weight(model_name, accuracy, profit)
        
        model_data['last_update'] = datetime.now()
    
    def _calculate_performance_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calcula pesos baseados em performance histórica"""
        weights = {}
        
        for model_name in model_names:
            if model_name in self.model_registry:
                model_data = self.model_registry[model_name]
                
                # Performance baseada em acurácia recente
                recent_performance = self._get_recent_performance(model_name)
                recent_profit = self._get_recent_profit(model_name)
                
                # Score combinado
                performance_score = (recent_performance * 0.7) + (min(recent_profit, 1.0) * 0.3)
                weights[model_name] = max(0.1, performance_score)
            else:
                # Peso inicial neutro
                weights[model_name] = 0.5
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _adjust_weights_for_regime(self, model_names: List[str], 
                                 market_conditions: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Ajusta pesos baseado no regime de mercado"""
        weights = {}
        
        if market_conditions is None:
            # Pesos neutros se não há informações de regime
            for model_name in model_names:
                weights[model_name] = 1.0
            return weights
        
        regime = market_conditions.get('regime', 'normal')
        volatility = market_conditions.get('volatility', 0.5)
        
        # Pesos específicos por regime
        regime_adjustments = {
            'high_vol_bull': {
                'prediction_agent': 1.2,  # Modelos preditivos mais importantes
                'news_agent': 0.8,        # Sentimento menos importante
                'vision_agent': 1.1,      # Padrões visuais importantes
                'xgboost': 1.3,           # XGBoost bom em alta volatilidade
                'lstm': 1.1,              # LSTM adaptativo
                'random_forest': 0.9      # RF mais conservador
            },
            'high_vol_bear': {
                'prediction_agent': 1.1,
                'news_agent': 1.0,        # Sentimento importante em bear market
                'vision_agent': 1.2,      # Padrões de reversão importantes
                'xgboost': 1.2,
                'lstm': 1.0,
                'random_forest': 1.1      # RF mais estável
            },
            'low_vol_bull': {
                'prediction_agent': 1.0,
                'news_agent': 0.9,
                'vision_agent': 0.8,      # Menos padrões em baixa volatilidade
                'xgboost': 1.0,
                'lstm': 1.2,              # LSTM bom em tendências
                'random_forest': 1.0
            },
            'low_vol_bear': {
                'prediction_agent': 0.9,
                'news_agent': 1.1,
                'vision_agent': 0.9,
                'xgboost': 0.9,
                'lstm': 1.0,
                'random_forest': 1.2      # RF mais confiável
            },
            'sideways': {
                'prediction_agent': 1.0,
                'news_agent': 1.0,
                'vision_agent': 1.0,
                'xgboost': 1.0,
                'lstm': 1.0,
                'random_forest': 1.0
            }
        }
        
        # Aplica ajustes do regime
        adjustments = regime_adjustments.get(regime, {})
        
        for model_name in model_names:
            # Ajuste baseado no regime
            regime_adjustment = adjustments.get(model_name, 1.0)
            
            # Ajuste baseado na volatilidade
            vol_adjustment = 1.0 + (volatility - 0.5) * 0.2
            
            # Peso final
            weights[model_name] = regime_adjustment * vol_adjustment
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_confidence_weights(self, signals: Dict[str, float], 
                                    model_confidences: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calcula pesos baseados na confiança dos modelos"""
        weights = {}
        
        if model_confidences is None:
            # Confiança baseada na consistência dos sinais
            signal_values = list(signals.values())
            signal_std = np.std(signal_values) if len(signal_values) > 1 else 0.5
            
            for model_name in signals.keys():
                # Menor desvio = maior confiança
                confidence = max(0.1, 1.0 - signal_std)
                weights[model_name] = confidence
        else:
            # Usa confianças fornecidas
            for model_name in signals.keys():
                confidence = model_confidences.get(model_name, 0.5)
                weights[model_name] = max(0.1, confidence)
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _combine_weights(self, performance_weights: Dict[str, float],
                        regime_weights: Dict[str, float],
                        confidence_weights: Dict[str, float]) -> Dict[str, float]:
        """Combina os diferentes tipos de pesos"""
        combined_weights = {}
        
        for model_name in performance_weights.keys():
            perf_weight = performance_weights.get(model_name, 0.5)
            regime_weight = regime_weights.get(model_name, 1.0)
            conf_weight = confidence_weights.get(model_name, 0.5)
            
            # Combinação ponderada
            combined_weight = (perf_weight * 0.4 + regime_weight * 0.4 + conf_weight * 0.2)
            combined_weights[model_name] = combined_weight
        
        # Normaliza pesos finais
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v / total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def _calculate_weighted_consensus(self, signals: Dict[str, float], 
                                    weights: Dict[str, float]) -> float:
        """Calcula consenso ponderado"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, signal in signals.items():
            weight = weights.get(model_name, 0.0)
            weighted_sum += signal * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.mean(list(signals.values())) if signals else 0.5
    
    def _calculate_ensemble_confidence(self, signals: Dict[str, float],
                                     weights: Dict[str, float],
                                     market_conditions: Optional[Dict[str, Any]]) -> float:
        """Calcula confiança do ensemble baseada em múltiplos fatores"""
        
        # ===== 1. CONSISTÊNCIA DOS SINAIS =====
        signal_values = list(signals.values())
        signal_std = np.std(signal_values) if len(signal_values) > 1 else 0.5
        consistency_score = max(0.1, 1.0 - signal_std)
        
        # ===== 2. CONFIABILIDADE DOS PESOS =====
        weight_entropy = -sum(w * np.log(w + 1e-8) for w in weights.values())
        max_entropy = np.log(len(weights)) if weights else 1.0
        weight_reliability = 1.0 - (weight_entropy / max_entropy) if max_entropy > 0 else 0.5
        
        # ===== 3. CONDIÇÕES DE MERCADO =====
        volatility = market_conditions.get('volatility', 0.5) if market_conditions else 0.5
        regime = market_conditions.get('regime', 'normal') if market_conditions else 'normal'
        
        # Confiança reduzida em alta volatilidade
        volatility_factor = 1.0 - (volatility * 0.3)
        
        # Confiança baseada no regime
        regime_confidence = {
            'high_vol_bull': 0.7,
            'high_vol_bear': 0.6,
            'low_vol_bull': 0.8,
            'low_vol_bear': 0.8,
            'sideways': 0.9
        }.get(regime, 0.7)
        
        # ===== 4. COMBINAÇÃO FINAL =====
        final_confidence = (
            consistency_score * 0.4 +
            weight_reliability * 0.3 +
            volatility_factor * 0.2 +
            regime_confidence * 0.1
        )
        
        # Limita confiança entre thresholds
        final_confidence = max(self.min_confidence_threshold, 
                             min(self.max_confidence_threshold, final_confidence))
        
        return final_confidence
    
    def _update_ensemble_history(self, signals: Dict[str, float], 
                               consensus: float, confidence: float, weights: Dict[str, float]):
        """Atualiza histórico do ensemble"""
        timestamp = datetime.now()
        
        self.ensemble_history.append({
            'timestamp': timestamp,
            'consensus': consensus,
            'confidence': confidence,
            'signals': signals.copy(),
            'weights': weights.copy(),
            'num_models': len(signals)
        })
        
        # Mantém apenas últimos 100 registros
        if len(self.ensemble_history) > 100:
            self.ensemble_history = self.ensemble_history[-100:]
    
    def _get_recent_performance(self, model_name: str, window: int = 20) -> float:
        """Obtém performance recente de um modelo"""
        if model_name not in self.model_registry:
            return 0.5
        
        history = self.model_registry[model_name]['performance_history']
        if not history:
            return 0.5
        
        recent_history = history[-window:] if len(history) >= window else history
        return np.mean([entry['accuracy'] for entry in recent_history])
    
    def _get_recent_profit(self, model_name: str, window: int = 20) -> float:
        """Obtém lucro recente de um modelo"""
        if model_name not in self.model_registry:
            return 0.0
        
        history = self.model_registry[model_name]['profit_history']
        if not history:
            return 0.0
        
        recent_history = history[-window:] if len(history) >= window else history
        return np.mean(recent_history)
    
    def _update_model_weight(self, model_name: str, accuracy: float, profit: float):
        """Atualiza peso de um modelo baseado na performance"""
        if model_name not in self.model_registry:
            return
        
        model_data = self.model_registry[model_name]
        
        # Calcula novo peso baseado em performance
        performance_score = (accuracy * 0.7) + (min(profit, 1.0) * 0.3)
        
        # Atualiza peso com suavização
        current_weight = model_data['weight']
        new_weight = 0.9 * current_weight + 0.1 * performance_score
        
        # Limita peso entre 0.1 e 2.0
        model_data['weight'] = max(0.1, min(2.0, new_weight))
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de todos os modelos"""
        stats = {}
        
        for model_name, model_data in self.model_registry.items():
            stats[model_name] = {
                'type': model_data['type'],
                'weight': model_data['weight'],
                'confidence': model_data['confidence'],
                'total_predictions': model_data['total_predictions'],
                'accuracy': model_data['correct_predictions'] / max(model_data['total_predictions'], 1),
                'recent_performance': self._get_recent_performance(model_name),
                'recent_profit': self._get_recent_profit(model_name),
                'last_update': model_data['last_update']
            }
        
        return stats
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do ensemble"""
        if not self.ensemble_history:
            return {'total_predictions': 0, 'avg_confidence': 0.0}
        
        recent_history = self.ensemble_history[-50:]  # Últimos 50 registros
        
        return {
            'total_predictions': len(self.ensemble_history),
            'avg_confidence': np.mean([h['confidence'] for h in recent_history]),
            'avg_consensus': np.mean([h['consensus'] for h in recent_history]),
            'avg_models_used': np.mean([h['num_models'] for h in recent_history]),
            'recent_consensus_history': [h['consensus'] for h in recent_history[-10:]]
        }
    
    def get_best_models(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Retorna os melhores modelos baseado em performance"""
        model_stats = self.get_model_statistics()
        
        # Ordena por performance recente
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: x[1]['recent_performance'],
            reverse=True
        )
        
        return [(name, stats['recent_performance']) for name, stats in sorted_models[:top_n]]
    
    def reset_model_performance(self, model_name: str):
        """Reseta performance de um modelo"""
        if model_name in self.model_registry:
            model_data = self.model_registry[model_name]
            model_data['performance_history'] = []
            model_data['profit_history'] = []
            model_data['total_predictions'] = 0
            model_data['correct_predictions'] = 0
            model_data['weight'] = 1.0
            print(f"🔄 Performance resetada para: {model_name}")
    
    def clear_all_history(self):
        """Limpa todo o histórico"""
        self.ensemble_history.clear()
        for model_name in self.model_registry:
            self.reset_model_performance(model_name)
        print("🗑️ Todo histórico limpo") 