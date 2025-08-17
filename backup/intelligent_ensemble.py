#!/usr/bin/env python3
"""
SISTEMA DE ENSEMBLE INTELIGENTE V2.0
====================================

Implementa ensemble avançado com:
- Pesos adaptativos baseados em performance
- Ajuste dinâmico por regime de mercado
- Confiança baseada em consistência
- Seleção inteligente de modelos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IntelligentEnsemble:
    """
    Sistema de ensemble inteligente com pesos adaptativos
    """
    
    def __init__(self):
        self.model_weights = {}
        self.model_performance = {}
        self.performance_history = {}
        self.regime_weights = {}
        self.confidence_history = []
        self.min_confidence_threshold = 0.3
        self.max_confidence_threshold = 0.9
        
    def calculate_intelligent_ensemble(self, signals: Dict[str, float], 
                                     market_conditions: Dict[str, Any],
                                     model_confidences: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Calcula ensemble inteligente com múltiplos fatores
        
        Args:
            signals: Dicionário com sinais dos modelos
            market_conditions: Condições de mercado
            model_confidences: Confiança de cada modelo
            
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
            self._update_performance_history(signals, consensus, confidence)
            
            return consensus, confidence
            
        except Exception as e:
            print(f"❌ Erro no ensemble inteligente: {e}")
            return 0.5, 0.1
    
    def _calculate_performance_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calcula pesos baseados em performance histórica"""
        weights = {}
        
        for model_name in model_names:
            if model_name in self.model_performance:
                # Performance baseada em acurácia recente
                recent_performance = self.model_performance[model_name].get('recent_accuracy', 0.5)
                recent_profit = self.model_performance[model_name].get('recent_profit', 0.0)
                
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
                                 market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Ajusta pesos baseado no regime de mercado"""
        weights = {}
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
                                     market_conditions: Dict[str, Any]) -> float:
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
        volatility = market_conditions.get('volatility', 0.5)
        regime = market_conditions.get('regime', 'normal')
        
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
    
    def _update_performance_history(self, signals: Dict[str, float], 
                                  consensus: float, confidence: float):
        """Atualiza histórico de performance"""
        timestamp = datetime.now()
        
        # Registra confiança
        self.confidence_history.append({
            'timestamp': timestamp,
            'confidence': confidence,
            'consensus': consensus,
            'signals': signals.copy()
        })
        
        # Mantém apenas últimos 100 registros
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]
    
    def update_model_performance(self, model_name: str, accuracy: float, profit: float):
        """Atualiza performance de um modelo específico"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'recent_accuracy': [],
                'recent_profit': [],
                'overall_accuracy': 0.0,
                'overall_profit': 0.0
            }
        
        # Adiciona métricas recentes
        self.model_performance[model_name]['recent_accuracy'].append(accuracy)
        self.model_performance[model_name]['recent_profit'].append(profit)
        
        # Mantém apenas últimos 20 registros
        if len(self.model_performance[model_name]['recent_accuracy']) > 20:
            self.model_performance[model_name]['recent_accuracy'] = \
                self.model_performance[model_name]['recent_accuracy'][-20:]
            self.model_performance[model_name]['recent_profit'] = \
                self.model_performance[model_name]['recent_profit'][-20:]
        
        # Atualiza médias
        self.model_performance[model_name]['recent_accuracy'] = \
            np.mean(self.model_performance[model_name]['recent_accuracy'])
        self.model_performance[model_name]['recent_profit'] = \
            np.mean(self.model_performance[model_name]['recent_profit'])
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Retorna resumo do ensemble"""
        return {
            'model_performance': self.model_performance,
            'confidence_history': len(self.confidence_history),
            'avg_confidence': np.mean([h['confidence'] for h in self.confidence_history]) if self.confidence_history else 0.0,
            'recent_consensus': [h['consensus'] for h in self.confidence_history[-10:]] if self.confidence_history else []
        } 