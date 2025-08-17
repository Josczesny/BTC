#!/usr/bin/env python3
"""
PROCESSADOR DE SINAIS
=====================

Módulo responsável pelo processamento e análise de sinais de trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors
from utils.logger import log_trade_info
from utils.technical_indicators import TechnicalIndicators

class SignalProcessor:
    """Processador de sinais de trading"""
    
    def __init__(self, central_ensemble_system=None, central_market_regime_system=None, central_feature_engine=None):
        """Inicializa o processador de sinais"""
        self.signal_history = []
        self.consensus_thresholds = {
            'STRONG_BUY': 0.75,
            'BUY': 0.60,
            'NEUTRAL': 0.45,
            'SELL': 0.40,
            'STRONG_SELL': 0.25
        }
        self.central_ensemble_system = central_ensemble_system
        self.central_market_regime_system = central_market_regime_system
        self.central_feature_engine = central_feature_engine
    
    def get_consensus_signal(self, data, price, model_predictions):
        """Obtém sinal de consenso usando sistema centralizado"""
        try:
            if self.central_ensemble_system:
                regime = self._detect_market_regime(data)
                signals = {
                    'xgboost': float(model_predictions.get('xgboost', 0.0)),
                    'random_forest': float(model_predictions.get('random_forest', 0.0)),
                    'lstm': float(model_predictions.get('lstm', 0.0)),
                    'technical': float(self.calculate_technical_score(data, price)),
                    'market_regime': float(self._regime_to_score(regime))
                }
                consensus, confidence = self.central_ensemble_system.get_ensemble_prediction(signals, market_conditions=None)
                return consensus, confidence
            else:
                return 0.5, 0.1
        except Exception as e:
            print(f"⚠️ Erro no sistema de ensemble centralizado: {e}")
            return 0.5, 0.1
    
    def calculate_technical_score(self, data, price):
        """Calcula score técnico público (antes era _calculate_technical_score)"""
        try:
            if data is None or data.empty:
                return 0.5
            score = 0.5  # Neutro
            if len(data) >= 20:
                ma_short = data['close'].rolling(window=5).mean().iloc[-1]
                ma_long = data['close'].rolling(window=20).mean().iloc[-1]
                if ma_short > ma_long:
                    score += 0.1
                else:
                    score -= 0.1
            if len(data) >= 14:
                rsi = TechnicalIndicators.calculate_rsi(data['close'].values)
                if len(rsi) > 0:
                    rsi_value = rsi[-1]
                    if rsi_value < 30:
                        score += 0.1
                    elif rsi_value > 70:
                        score -= 0.1
            volatility = data['close'].pct_change().std()
            if volatility > 0.02:
                score *= 0.9
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro na análise técnica: {e}"))
            return 0.5
    
    def _detect_market_regime(self, data):
        """Detecta regime de mercado usando sistema centralizado"""
        try:
            if self.central_market_regime_system:
                regime_info = self.central_market_regime_system.get_current_regime(data)
                return regime_info['regime']
            else:
                return 'sideways'
        except Exception as e:
            print(f"⚠️ Erro no sistema de market regime centralizado: {e}")
            return 'sideways'
    
    def _prepare_advanced_features(self, data):
        """Prepara features avançadas usando sistema centralizado"""
        try:
            if hasattr(self, 'central_feature_engine') and self.central_feature_engine and data is not None and len(data) >= 50:
                try:
                    return self.central_feature_engine.get_all_features(data, 'basic')
                except Exception as e:
                    print(f"⚠️ Erro no sistema de features centralizado: {e}")
                    return self._prepare_basic_features_fallback(data)
            else:
                return self._prepare_basic_features_fallback(data)
            
        except Exception as e:
            print(f"⚠️ Erro na preparação de features: {e}")
            return pd.DataFrame()
    
    def _prepare_basic_features_fallback(self, data):
        """Fallback básico para features se sistema central não estiver disponível"""
        try:
            if data is None or len(data) < 20:
                return pd.DataFrame()
            
            # Usa apenas as últimas 50 linhas para evitar problemas de length
            df = data.tail(50).copy()
            
            # Calcula indicadores básicos com sistema centralizado
            rsi_values = TechnicalIndicators.calculate_rsi(df['close'].values)
            # Garante que RSI tenha o mesmo tamanho que o DataFrame
            if len(rsi_values) < len(df):
                # Preenche com valores padrão se necessário
                rsi_padded = np.full(len(df), 50.0)
                rsi_padded[-len(rsi_values):] = rsi_values
                df['rsi'] = rsi_padded
            else:
                df['rsi'] = pd.Series(rsi_values[-len(df):], index=df.index)
            
            df['volatility'] = df['close'].rolling(20).std().fillna(df['close'].std())
            df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'].mean())
            df['sma_50'] = df['close'].rolling(50).mean().fillna(df['close'].mean())
            
            # Garante que todas as colunas tenham o mesmo tamanho
            df = df.dropna()  # Remove linhas com NaN
            
            # Seleciona features básicas (8 features como esperado pelos modelos)
            basic_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_features = [col for col in basic_features if col in df.columns]
            
            # Garante que temos pelo menos algumas features
            if len(available_features) < 3:
                print("⚠️ Poucas features disponíveis, usando fallback")
                return pd.DataFrame()
            
            # Garante que o DataFrame tenha exatamente o tamanho esperado
            result_df = df[available_features].copy()
            
            # Se ainda houver problemas de tamanho, usa apenas a última linha
            if len(result_df) > 0:
                return result_df.iloc[-1:].copy()
            else:
                return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠️ Erro no fallback de features: {e}")
            return pd.DataFrame()
    
    def _calculate_weighted_confidence(self, predictions, technical_score, market_regime):
        """Calcula confiança ponderada"""
        try:
            # Peso dos modelos (60%)
            model_weight = 0.6
            avg_prediction = np.mean(predictions)
            
            # Peso da análise técnica (30%)
            technical_weight = 0.3
            
            # Peso do regime de mercado (10%)
            regime_weight = 0.1
            regime_score = self._regime_to_score(market_regime)
            
            # Confiança ponderada
            confidence = (
                avg_prediction * model_weight +
                technical_score * technical_weight +
                regime_score * regime_weight
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro no cálculo de confiança: {e}"))
            return 0.5
    
    def _regime_to_score(self, regime):
        regime_scores = {
            'BULLISH': 0.7,
            'NEUTRAL': 0.5,
            'SIDEWAYS': 0.5,
            'BEARISH': 0.3,
            'TRENDING_UP': 0.7,
            'TRENDING_DOWN': 0.3,
            'VOLATILE': 0.4,
            'SIDEWAYS': 0.5
        }
        return regime_scores.get(str(regime).upper(), 0.5)
    
    def _determine_signal(self, confidence):
        """Determina sinal baseado na confiança"""
        if confidence >= self.consensus_thresholds['STRONG_BUY']:
            return 'STRONG_BUY'
        elif confidence >= self.consensus_thresholds['BUY']:
            return 'BUY'
        elif confidence >= self.consensus_thresholds['NEUTRAL']:
            return 'NEUTRAL'
        elif confidence >= self.consensus_thresholds['SELL']:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _update_model_performance(self, model_predictions, consensus_signal, confidence):
        """Atualiza performance dos modelos no ensemble centralizado após cada consenso"""
        try:
            # Simula resultado real (em produção seria baseado no resultado real do trade)
            # Por enquanto, usa o consenso como proxy do resultado
            actual_outcome = 0.5  # Neutro como base
            
            # Converte sinal para valor numérico
            signal_values = {
                'STRONG_BUY': 0.8,
                'BUY': 0.6,
                'NEUTRAL': 0.5,
                'SELL': 0.4,
                'STRONG_SELL': 0.2
            }
            actual_outcome = signal_values.get(consensus_signal, 0.5)
            
            # Atualiza performance de cada modelo
            for model_name, prediction in model_predictions.items():
                # Calcula "lucro" baseado na precisão da predição
                prediction_accuracy = 1.0 - abs(prediction - actual_outcome)
                simulated_profit = (prediction_accuracy - 0.5) * 2.0  # Normaliza para -1 a 1
                
                # Atualiza no sistema central (se disponível)
                if hasattr(self, 'central_ensemble_system') and self.central_ensemble_system:
                    try:
                        self.central_ensemble_system.update_model_performance(
                            model_name, prediction, actual_outcome, simulated_profit
                        )
                        log_trade_info(f"Performance do modelo {model_name} atualizada", level='INFO')
                    except Exception as e:
                        log_trade_info(f"Erro ao atualizar performance do modelo {model_name}: {e}", level='WARNING')
                else:
                    # Fallback: atualiza localmente
                    self._update_local_model_performance(model_name, prediction, actual_outcome, simulated_profit)
                
        except Exception as e:
            log_trade_info(f"⚠️ Erro atualizando performance dos modelos: {e}", level='WARNING')
    
    def _update_local_model_performance(self, model_name, prediction, actual_outcome, simulated_profit):
        """Atualiza performance local dos modelos (fallback)"""
        try:
            if not hasattr(self, 'local_model_performance'):
                self.local_model_performance = {}
            
            if model_name not in self.local_model_performance:
                self.local_model_performance[model_name] = {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'total_profit': 0.0,
                    'accuracy': 0.0
                }
            
            model_data = self.local_model_performance[model_name]
            
            # Atualiza contadores
            model_data['total_predictions'] += 1
            model_data['total_profit'] += simulated_profit
            
            # Verifica se previsão estava correta (dentro de uma margem)
            prediction_error = abs(prediction - actual_outcome)
            if prediction_error < 0.1:  # Margem de 10%
                model_data['correct_predictions'] += 1
            
            # Calcula nova accuracy
            model_data['accuracy'] = model_data['correct_predictions'] / model_data['total_predictions']
            
            log_trade_info(f"Performance local atualizada: {model_name} - Accuracy: {model_data['accuracy']:.2f}", level='INFO')
            
        except Exception as e:
            log_trade_info(f"Erro ao atualizar performance local do modelo {model_name}: {e}", level='WARNING')
    
    def get_future_prediction(self, market_data):
        """Obtém previsão futura baseada nos dados de mercado"""
        try:
            if market_data is None or market_data.empty:
                return 0.5
            
            # Análise de tendência simples
            if len(market_data) >= 5:
                recent_trend = market_data['close'].iloc[-5:].pct_change().mean()
                
                # Normaliza tendência para score entre 0 e 1
                if recent_trend > 0.01:  # Tendência forte de alta
                    return 0.8
                elif recent_trend > 0.005:  # Tendência moderada de alta
                    return 0.7
                elif recent_trend > 0:  # Tendência fraca de alta
                    return 0.6
                elif recent_trend > -0.005:  # Tendência fraca de baixa
                    return 0.4
                elif recent_trend > -0.01:  # Tendência moderada de baixa
                    return 0.3
                else:  # Tendência forte de baixa
                    return 0.2
            
            return 0.5  # Neutro se não há dados suficientes
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro na previsão futura: {e}"))
            return 0.5
    
    def get_signal_statistics(self):
        """Obtém estatísticas dos sinais"""
        try:
            if not self.signal_history:
                return {}
            
            signals = [entry['signal'] for entry in self.signal_history]
            confidences = [entry['confidence'] for entry in self.signal_history]
            
            return {
                'total_signals': len(signals),
                'signal_distribution': pd.Series(signals).value_counts().to_dict(),
                'avg_confidence': np.mean(confidences),
                'recent_signals': signals[-10:] if len(signals) >= 10 else signals
            }
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao obter estatísticas: {e}"))
            return {} 