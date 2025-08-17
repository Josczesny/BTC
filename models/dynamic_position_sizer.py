#!/usr/bin/env python3
"""
SISTEMA DE POSITION SIZING DINÂMICO
===================================

Implementa Kelly Criterion modificado para position sizing ótimo:
- Calcula tamanho ótimo da posição baseado em performance histórica
- Ajusta baseado na confiança do sinal e volatilidade
- Risk management integrado
- Performance tracking para otimização contínua
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicPositionSizer:
    """
    Sistema de position sizing dinâmico usando Kelly Criterion modificado
    """
    
    def __init__(self):
        self.risk_per_trade = 0.02  # 2% risco por trade
        self.max_position_size = 0.1  # 10% máximo do capital
        self.min_position_size = 0.01  # 1% mínimo do capital
        
        # Histórico de performance
        self.trade_history = []
        self.win_rate = 0.5  # Taxa de acerto inicial
        self.avg_win = 0.02  # Ganho médio (2%)
        self.avg_loss = 0.015  # Perda média (1.5%)
        
        # Configurações Kelly
        self.kelly_fraction = 0.5  # Fração Kelly (conservador)
        self.volatility_penalty = 0.3  # Penalidade por volatilidade
        self.confidence_multiplier = 0.5  # Multiplicador por confiança
        
        print("🚀 Dynamic Position Sizer inicializado")
    
    def calculate_position_size(self, signal_strength: float, 
                              market_conditions: Dict[str, Any], 
                              balance: float) -> float:
        """
        Calcula tamanho ótimo da posição usando Kelly Criterion modificado
        
        Args:
            signal_strength: Força do sinal (0-1)
            market_conditions: Condições de mercado (volatilidade, regime, etc.)
            balance: Saldo disponível
            
        Returns:
            float: Tamanho da posição em USD
        """
        try:
            if balance <= 0:
                return 0.0
            
            # ===== 1. KELLY CRITERION BASE =====
            kelly_fraction = self._calculate_kelly_fraction()
            
            # ===== 2. AJUSTE POR CONFIANÇA DO SINAL =====
            confidence_adjustment = signal_strength * self.confidence_multiplier
            
            # ===== 3. AJUSTE POR VOLATILIDADE =====
            volatility = market_conditions.get('volatility', 0.5)
            volatility_penalty = volatility * self.volatility_penalty
            
            # ===== 4. AJUSTE POR REGIME DE MERCADO =====
            regime_adjustment = self._get_regime_adjustment(market_conditions)
            
            # ===== 5. CÁLCULO FINAL =====
            final_fraction = kelly_fraction * confidence_adjustment * (1 - volatility_penalty) * regime_adjustment
            
            # ===== 6. LIMITES DE SEGURANÇA =====
            final_fraction = max(self.min_position_size, min(self.max_position_size, final_fraction))
            
            # ===== 7. CALCULA POSITION SIZE =====
            position_size = balance * final_fraction
            
            # ===== 8. REGISTRA CÁLCULO =====
            self._register_calculation(signal_strength, market_conditions, final_fraction, position_size)
            
            return position_size
            
        except Exception as e:
            print(f"❌ Erro no cálculo de position size: {e}")
            # Retorna position size conservador
            return balance * self.min_position_size
    
    def _calculate_kelly_fraction(self) -> float:
        """Calcula fração Kelly baseada em performance histórica"""
        try:
            if self.avg_loss == 0:
                return self.min_position_size
            
            # Kelly Criterion: f = (bp - q) / b
            # onde: b = odds recebidas, p = probabilidade de ganho, q = probabilidade de perda
            b = self.avg_win / self.avg_loss  # Odds recebidas
            p = self.win_rate  # Probabilidade de ganho
            q = 1 - self.win_rate  # Probabilidade de perda
            
            kelly_fraction = (b * p - q) / b
            
            # Aplica fração Kelly conservadora
            kelly_fraction = kelly_fraction * self.kelly_fraction
            
            # Limites de segurança
            kelly_fraction = max(0.01, min(0.1, kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            print(f"❌ Erro no cálculo Kelly: {e}")
            return self.min_position_size
    
    def _get_regime_adjustment(self, market_conditions: Dict[str, Any]) -> float:
        """Ajusta position size baseado no regime de mercado"""
        regime = market_conditions.get('regime', 'normal')
        
        regime_adjustments = {
            'high_vol_bull': 0.8,    # Reduz em alta volatilidade bullish
            'high_vol_bear': 0.6,    # Reduz mais em alta volatilidade bearish
            'low_vol_bull': 1.2,     # Aumenta em baixa volatilidade bullish
            'low_vol_bear': 0.9,     # Reduz levemente em baixa volatilidade bearish
            'sideways': 0.7,         # Reduz em mercado lateral
            'normal': 1.0            # Sem ajuste
        }
        
        return regime_adjustments.get(regime, 1.0)
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """
        Atualiza performance baseada no resultado do trade
        
        Args:
            trade_result: Resultado do trade com 'profit', 'position_size', etc.
        """
        try:
            self.trade_history.append(trade_result)
            
            # Mantém apenas histórico recente (últimos 100 trades)
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            
            # Calcula novas estatísticas
            self._update_statistics()
            
        except Exception as e:
            print(f"❌ Erro ao atualizar performance: {e}")
    
    def _update_statistics(self):
        """Atualiza estatísticas de performance"""
        try:
            if len(self.trade_history) < 10:
                return
            
            # Calcula win rate
            wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
            self.win_rate = wins / len(self.trade_history)
            
            # Calcula ganho médio
            profits = [trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0]
            if profits:
                self.avg_win = np.mean(profits)
            
            # Calcula perda média
            losses = [abs(trade.get('profit', 0)) for trade in self.trade_history if trade.get('profit', 0) < 0]
            if losses:
                self.avg_loss = np.mean(losses)
            
        except Exception as e:
            print(f"❌ Erro ao atualizar estatísticas: {e}")
    
    def _register_calculation(self, signal_strength: float, market_conditions: Dict[str, Any], 
                            final_fraction: float, position_size: float):
        """Registra cálculo para análise"""
        calculation = {
            'timestamp': datetime.now(),
            'signal_strength': signal_strength,
            'market_conditions': market_conditions,
            'final_fraction': final_fraction,
            'position_size': position_size,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }
        
        # Aqui você pode salvar em log ou banco de dados
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do position sizer"""
        return {
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': len(self.trade_history),
            'kelly_fraction': self.kelly_fraction,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size
        }
    
    def reset_performance(self):
        """Reseta performance para valores iniciais"""
        self.trade_history = []
        self.win_rate = 0.5
        self.avg_win = 0.02
        self.avg_loss = 0.015
        print("🔄 Performance resetada")
    
    def optimize_parameters(self):
        """Otimiza parâmetros baseado em performance histórica"""
        try:
            if len(self.trade_history) < 20:
                return
            
            # Analisa performance recente
            recent_trades = self.trade_history[-20:]
            recent_wins = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
            recent_win_rate = recent_wins / len(recent_trades)
            
            # Ajusta Kelly fraction baseado em performance
            if recent_win_rate > 0.6:
                self.kelly_fraction = min(0.7, self.kelly_fraction * 1.1)
            elif recent_win_rate < 0.4:
                self.kelly_fraction = max(0.3, self.kelly_fraction * 0.9)
            
            print(f"🔄 Parâmetros otimizados - Win rate recente: {recent_win_rate:.2f}")
            
        except Exception as e:
            print(f"❌ Erro na otimização: {e}") 