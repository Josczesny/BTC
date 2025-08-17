#!/usr/bin/env python3
"""
DETECTOR AVANÇADO DE REGIME DE MERCADO V2.0
===========================================

Implementa detecção sofisticada de regimes de mercado:
- Análise de volatilidade dinâmica
- Detecção de tendências
- Análise de liquidez
- Análise de sentimento
- Classificação multi-dimensional
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Sistema avançado de detecção de regime de mercado
    """
    
    def __init__(self):
        self.regime_history = []
        self.volatility_thresholds = {
            'low': 0.02,      # 2% volatilidade
            'medium': 0.05,   # 5% volatilidade
            'high': 0.10      # 10% volatilidade
        }
        self.trend_thresholds = {
            'strong_bull': 0.7,
            'weak_bull': 0.3,
            'weak_bear': -0.3,
            'strong_bear': -0.7
        }
        
    def detect_market_regime(self, data: pd.DataFrame, 
                           sentiment_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detecta regime atual do mercado usando múltiplos indicadores
        
        Args:
            data: Dados OHLCV
            sentiment_data: Dados de sentimento (opcional)
            
        Returns:
            Dict com regime e métricas
        """
        try:
            if data is None or len(data) < 50:
                return self._get_default_regime()
            
            # ===== 1. ANÁLISE DE VOLATILIDADE =====
            volatility_metrics = self._analyze_volatility(data)
            
            # ===== 2. ANÁLISE DE TENDÊNCIA =====
            trend_metrics = self._analyze_trend(data)
            
            # ===== 3. ANÁLISE DE LIQUIDEZ =====
            liquidity_metrics = self._analyze_liquidity(data)
            
            # ===== 4. ANÁLISE DE SENTIMENTO =====
            sentiment_metrics = self._analyze_sentiment(sentiment_data)
            
            # ===== 5. CLASSIFICAÇÃO DO REGIME =====
            regime_classification = self._classify_regime(
                volatility_metrics, trend_metrics, liquidity_metrics, sentiment_metrics
            )
            
            # ===== 6. CÁLCULO DE CONFIANÇA =====
            confidence = self._calculate_regime_confidence(
                volatility_metrics, trend_metrics, liquidity_metrics, sentiment_metrics
            )
            
            # ===== 7. ATUALIZA HISTÓRICO =====
            self._update_regime_history(regime_classification, confidence)
            
            return {
                'regime': regime_classification['primary_regime'],
                'sub_regime': regime_classification['sub_regime'],
                'confidence': confidence,
                'volatility': volatility_metrics['current_volatility'],
                'trend_strength': trend_metrics['trend_strength'],
                'liquidity_score': liquidity_metrics['liquidity_score'],
                'sentiment_score': sentiment_metrics['sentiment_score'],
                'metrics': {
                    'volatility': volatility_metrics,
                    'trend': trend_metrics,
                    'liquidity': liquidity_metrics,
                    'sentiment': sentiment_metrics
                }
            }
            
        except Exception as e:
            print(f"❌ Erro na detecção de regime: {e}")
            return self._get_default_regime()
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analisa volatilidade do mercado"""
        
        # Retornos
        returns = data['close'].pct_change().dropna()
        
        # Volatilidade atual (20 períodos)
        current_volatility = returns.rolling(20).std().iloc[-1]
        
        # Volatilidade histórica (100 períodos)
        historical_volatility = returns.rolling(100).std().iloc[-1]
        
        # Volatilidade relativa
        relative_volatility = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # Volatilidade anualizada
        annualized_volatility = current_volatility * np.sqrt(252)
        
        # Classificação da volatilidade
        if current_volatility <= self.volatility_thresholds['low']:
            vol_class = 'low'
        elif current_volatility <= self.volatility_thresholds['medium']:
            vol_class = 'medium'
        else:
            vol_class = 'high'
        
        return {
            'current_volatility': current_volatility,
            'historical_volatility': historical_volatility,
            'relative_volatility': relative_volatility,
            'annualized_volatility': annualized_volatility,
            'volatility_class': vol_class
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analisa tendência do mercado"""
        
        # Médias móveis
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        sma_100 = data['close'].rolling(100).mean()
        
        current_price = data['close'].iloc[-1]
        
        # Força da tendência
        trend_strength = (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        # Direção da tendência
        trend_direction = np.where(current_price > sma_20.iloc[-1] > sma_50.iloc[-1], 1,
                                 np.where(current_price < sma_20.iloc[-1] < sma_50.iloc[-1], -1, 0))
        
        # Momentum
        momentum_5 = (current_price / data['close'].iloc[-6] - 1) * 100
        momentum_20 = (current_price / data['close'].iloc[-21] - 1) * 100
        
        # Classificação da tendência
        if trend_strength >= self.trend_thresholds['strong_bull']:
            trend_class = 'strong_bull'
        elif trend_strength >= self.trend_thresholds['weak_bull']:
            trend_class = 'weak_bull'
        elif trend_strength <= self.trend_thresholds['strong_bear']:
            trend_class = 'strong_bear'
        elif trend_strength <= self.trend_thresholds['weak_bear']:
            trend_class = 'weak_bear'
        else:
            trend_class = 'sideways'
        
        return {
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'momentum_5': momentum_5,
            'momentum_20': momentum_20,
            'trend_class': trend_class,
            'price_vs_sma20': (current_price / sma_20.iloc[-1] - 1) * 100,
            'price_vs_sma50': (current_price / sma_50.iloc[-1] - 1) * 100
        }
    
    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analisa liquidez do mercado"""
        
        # Volume médio
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Spread implícito (simulado)
        spread = (data['high'] - data['low']) / data['close']
        avg_spread = spread.rolling(20).mean().iloc[-1]
        
        # Liquidez score (inverso do spread)
        liquidity_score = max(0, 1.0 - (avg_spread * 10))
        
        # Classificação da liquidez
        if volume_ratio > 1.5 and liquidity_score > 0.7:
            liquidity_class = 'high'
        elif volume_ratio < 0.5 or liquidity_score < 0.3:
            liquidity_class = 'low'
        else:
            liquidity_class = 'medium'
        
        return {
            'volume_ratio': volume_ratio,
            'avg_spread': avg_spread,
            'liquidity_score': liquidity_score,
            'liquidity_class': liquidity_class
        }
    
    def _analyze_sentiment(self, sentiment_data: Optional[Dict]) -> Dict[str, float]:
        """Analisa sentimento do mercado"""
        
        if sentiment_data is None:
            return {
                'sentiment_score': 0.5,
                'sentiment_class': 'neutral',
                'sentiment_confidence': 0.1
            }
        
        # Extrai score de sentimento
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        
        # Converte de escala -1,1 para 0,1
        normalized_sentiment = (sentiment_score + 1) / 2.0
        
        # Classificação do sentimento
        if normalized_sentiment > 0.7:
            sentiment_class = 'bullish'
        elif normalized_sentiment < 0.3:
            sentiment_class = 'bearish'
        else:
            sentiment_class = 'neutral'
        
        # Confiança do sentimento
        sentiment_confidence = sentiment_data.get('confidence', 0.5)
        
        return {
            'sentiment_score': normalized_sentiment,
            'sentiment_class': sentiment_class,
            'sentiment_confidence': sentiment_confidence
        }
    
    def _classify_regime(self, volatility_metrics: Dict, trend_metrics: Dict,
                        liquidity_metrics: Dict, sentiment_metrics: Dict) -> Dict[str, str]:
        """Classifica o regime de mercado"""
        
        vol_class = volatility_metrics['volatility_class']
        trend_class = trend_metrics['trend_class']
        liquidity_class = liquidity_metrics['liquidity_class']
        sentiment_class = sentiment_metrics['sentiment_class']
        
        # ===== CLASSIFICAÇÃO PRIMÁRIA =====
        
        # Alta volatilidade + tendência forte
        if vol_class == 'high' and trend_class in ['strong_bull', 'strong_bear']:
            if trend_class == 'strong_bull':
                primary_regime = 'high_vol_bull'
            else:
                primary_regime = 'high_vol_bear'
        
        # Baixa volatilidade + tendência forte
        elif vol_class == 'low' and trend_class in ['strong_bull', 'strong_bear']:
            if trend_class == 'strong_bull':
                primary_regime = 'low_vol_bull'
            else:
                primary_regime = 'low_vol_bear'
        
        # Alta volatilidade + sem tendência clara
        elif vol_class == 'high' and trend_class == 'sideways':
            primary_regime = 'high_vol_sideways'
        
        # Baixa volatilidade + sem tendência clara
        elif vol_class == 'low' and trend_class == 'sideways':
            primary_regime = 'low_vol_sideways'
        
        # Casos intermediários
        else:
            if trend_class in ['weak_bull', 'strong_bull']:
                primary_regime = 'moderate_bull'
            elif trend_class in ['weak_bear', 'strong_bear']:
                primary_regime = 'moderate_bear'
            else:
                primary_regime = 'sideways'
        
        # ===== SUB-REGIME =====
        
        # Combina liquidez e sentimento
        if liquidity_class == 'high' and sentiment_class == 'bullish':
            sub_regime = 'high_liquidity_bullish'
        elif liquidity_class == 'high' and sentiment_class == 'bearish':
            sub_regime = 'high_liquidity_bearish'
        elif liquidity_class == 'low' and sentiment_class == 'bullish':
            sub_regime = 'low_liquidity_bullish'
        elif liquidity_class == 'low' and sentiment_class == 'bearish':
            sub_regime = 'low_liquidity_bearish'
        else:
            sub_regime = 'mixed_conditions'
        
        return {
            'primary_regime': primary_regime,
            'sub_regime': sub_regime
        }
    
    def _calculate_regime_confidence(self, volatility_metrics: Dict, trend_metrics: Dict,
                                   liquidity_metrics: Dict, sentiment_metrics: Dict) -> float:
        """Calcula confiança da classificação do regime"""
        
        # Confiança baseada na consistência dos indicadores
        confidence_factors = []
        
        # Volatilidade bem definida
        vol_confidence = 1.0 - abs(volatility_metrics['relative_volatility'] - 1.0)
        confidence_factors.append(vol_confidence)
        
        # Tendência bem definida
        trend_confidence = abs(trend_metrics['trend_strength'])
        confidence_factors.append(min(trend_confidence, 1.0))
        
        # Liquidez consistente
        liquidity_confidence = liquidity_metrics['liquidity_score']
        confidence_factors.append(liquidity_confidence)
        
        # Sentimento confiável
        sentiment_confidence = sentiment_metrics['sentiment_confidence']
        confidence_factors.append(sentiment_confidence)
        
        # Confiança média
        avg_confidence = np.mean(confidence_factors)
        
        # Ajusta baseado na consistência
        consistency = 1.0 - np.std(confidence_factors)
        final_confidence = avg_confidence * (0.7 + 0.3 * consistency)
        
        return max(0.1, min(0.95, final_confidence))
    
    def _update_regime_history(self, regime_classification: Dict, confidence: float):
        """Atualiza histórico de regimes"""
        timestamp = datetime.now()
        
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': regime_classification['primary_regime'],
            'sub_regime': regime_classification['sub_regime'],
            'confidence': confidence
        })
        
        # Mantém apenas últimos 100 registros
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def _get_default_regime(self) -> Dict[str, Any]:
        """Retorna regime padrão quando não há dados suficientes"""
        return {
            'regime': 'unknown',
            'sub_regime': 'unknown',
            'confidence': 0.1,
            'volatility': 0.5,
            'trend_strength': 0.0,
            'liquidity_score': 0.5,
            'sentiment_score': 0.5,
            'metrics': {
                'volatility': {'volatility_class': 'medium'},
                'trend': {'trend_class': 'sideways'},
                'liquidity': {'liquidity_class': 'medium'},
                'sentiment': {'sentiment_class': 'neutral'}
            }
        }
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Retorna resumo dos regimes detectados"""
        if not self.regime_history:
            return {'total_regimes': 0, 'current_regime': 'unknown'}
        
        # Regime atual
        current_regime = self.regime_history[-1]['regime']
        
        # Distribuição de regimes
        regime_counts = {}
        for record in self.regime_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Confiança média
        avg_confidence = np.mean([r['confidence'] for r in self.regime_history])
        
        return {
            'total_regimes': len(self.regime_history),
            'current_regime': current_regime,
            'regime_distribution': regime_counts,
            'avg_confidence': avg_confidence,
            'recent_regimes': [r['regime'] for r in self.regime_history[-10:]]
        } 