#!/usr/bin/env python3
"""
SISTEMA DE RISK MANAGEMENT AVANÇADO
===================================

Implementa gestão de risco multi-layer:
- Limites de perda diária
- Controle de drawdown máximo
- Gestão de correlação entre posições
- Diversificação de portfólio
- Análise de risco em tempo real
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """
    Sistema de risk management avançado
    """
    
    def __init__(self):
        # Limites de risco
        self.max_daily_loss = 0.05  # 5% perda diária
        self.max_drawdown = 0.15    # 15% drawdown máximo
        self.max_correlation = 0.7   # Correlação máxima entre posições
        self.max_positions = 3      # Máximo de posições simultâneas
        self.max_portfolio_risk = 0.05  # 5% risco total do portfólio
        
        # Histórico de performance
        self.daily_pnl = []
        self.position_history = []
        self.risk_metrics = {}
        
        # Estado atual
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        self.active_positions = []
        
        print("🚀 Advanced Risk Manager inicializado")
    
    def can_open_position(self, new_signal: Dict[str, Any], 
                         existing_positions: List[Dict[str, Any]], 
                         daily_pnl: float) -> Tuple[bool, str]:
        """
        Verifica se pode abrir nova posição
        
        Args:
            new_signal: Sinal da nova posição
            existing_positions: Posições existentes
            daily_pnl: P&L diário atual
            
        Returns:
            Tuple[bool, str]: (pode abrir, razão)
        """
        try:
            # ===== 1. VERIFICA PERDA DIÁRIA =====
            if daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit exceeded: {daily_pnl:.2%}"
            
            # ===== 2. VERIFICA DRAWDOWN =====
            current_drawdown = self.calculate_drawdown()
            if current_drawdown > self.max_drawdown:
                return False, f"Maximum drawdown exceeded: {current_drawdown:.2%}"
            
            # ===== 3. VERIFICA NÚMERO DE POSIÇÕES =====
            if len(existing_positions) >= self.max_positions:
                return False, f"Maximum positions reached: {len(existing_positions)}"
            
            # ===== 4. VERIFICA CORRELAÇÃO =====
            if existing_positions:
                correlation = self.calculate_correlation(new_signal, existing_positions)
                if correlation > self.max_correlation:
                    return False, f"High correlation detected: {correlation:.2f}"
            
            # ===== 5. VERIFICA RISCO TOTAL DO PORTFÓLIO =====
            portfolio_risk = self.calculate_portfolio_risk(existing_positions + [new_signal])
            if portfolio_risk > self.max_portfolio_risk:
                return False, f"Portfolio risk too high: {portfolio_risk:.2%}"
            
            return True, "Trade allowed"
            
        except Exception as e:
            print(f"❌ Erro na verificação de risco: {e}")
            return False, f"Risk check error: {e}"
    
    def calculate_drawdown(self) -> float:
        """Calcula drawdown atual"""
        try:
            if not self.daily_pnl:
                return 0.0
            
            # Calcula equity curve
            equity_curve = np.cumsum(self.daily_pnl)
            
            # Encontra peak
            peak = np.maximum.accumulate(equity_curve)
            
            # Calcula drawdown
            drawdown = (equity_curve - peak) / peak
            
            # Retorna drawdown atual
            return abs(drawdown[-1]) if len(drawdown) > 0 else 0.0
            
        except Exception as e:
            print(f"❌ Erro no cálculo de drawdown: {e}")
            return 0.0
    
    def calculate_correlation(self, new_signal: Dict[str, Any], 
                            existing_positions: List[Dict[str, Any]]) -> float:
        """Calcula correlação entre nova posição e existentes"""
        try:
            if not existing_positions:
                return 0.0
            
            # Extrai características dos sinais
            new_features = self._extract_signal_features(new_signal)
            
            correlations = []
            for position in existing_positions:
                pos_features = self._extract_signal_features(position)
                
                # Calcula correlação
                if len(new_features) == len(pos_features):
                    correlation = np.corrcoef(new_features, pos_features)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            print(f"❌ Erro no cálculo de correlação: {e}")
            return 0.0
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calcula risco total do portfólio"""
        try:
            if not positions:
                return 0.0
            
            # Calcula risco individual de cada posição
            individual_risks = []
            for position in positions:
                risk = self._calculate_position_risk(position)
                individual_risks.append(risk)
            
            # Calcula risco total (simplificado)
            total_risk = np.sum(individual_risks)
            
            return min(total_risk, 1.0)  # Máximo 100%
            
        except Exception as e:
            print(f"❌ Erro no cálculo de risco do portfólio: {e}")
            return 0.0
    
    def _extract_signal_features(self, signal: Dict[str, Any]) -> List[float]:
        """Extrai features do sinal para cálculo de correlação"""
        try:
            features = []
            
            # Features básicas
            features.append(signal.get('confidence', 0.5))
            features.append(signal.get('signal_strength', 0.5))
            
            # Features de mercado
            market_conditions = signal.get('market_conditions', {})
            features.append(market_conditions.get('volatility', 0.5))
            features.append(market_conditions.get('trend_strength', 0.5))
            
            # Features de regime
            regime = market_conditions.get('regime', 'normal')
            regime_encoding = {
                'high_vol_bull': 0.8,
                'high_vol_bear': 0.2,
                'low_vol_bull': 0.9,
                'low_vol_bear': 0.1,
                'sideways': 0.5,
                'normal': 0.5
            }
            features.append(regime_encoding.get(regime, 0.5))
            
            return features
            
        except Exception as e:
            print(f"❌ Erro na extração de features: {e}")
            return [0.5, 0.5, 0.5, 0.5, 0.5]
    
    def _calculate_position_risk(self, position: Dict[str, Any]) -> float:
        """Calcula risco individual de uma posição"""
        try:
            # Risco baseado na volatilidade
            volatility = position.get('market_conditions', {}).get('volatility', 0.5)
            
            # Risco baseado na confiança (menor confiança = maior risco)
            confidence = position.get('confidence', 0.5)
            confidence_risk = 1.0 - confidence
            
            # Risco baseado no tamanho da posição
            position_size = position.get('position_size', 0.02)
            size_risk = position_size / 0.1  # Normalizado para 10%
            
            # Risco combinado
            total_risk = (volatility * 0.4 + confidence_risk * 0.3 + size_risk * 0.3)
            
            return min(total_risk, 1.0)
            
        except Exception as e:
            print(f"❌ Erro no cálculo de risco da posição: {e}")
            return 0.5
    
    def update_daily_pnl(self, pnl: float):
        """Atualiza P&L diário"""
        try:
            self.daily_pnl.append(pnl)
            
            # Mantém apenas histórico recente (últimos 30 dias)
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
            
            # Atualiza métricas
            self._update_risk_metrics()
            
        except Exception as e:
            print(f"❌ Erro ao atualizar P&L diário: {e}")
    
    def add_position(self, position: Dict[str, Any]):
        """Adiciona posição ao histórico"""
        try:
            position['timestamp'] = datetime.now()
            self.position_history.append(position)
            
            # Mantém apenas histórico recente
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]
            
        except Exception as e:
            print(f"❌ Erro ao adicionar posição: {e}")
    
    def _update_risk_metrics(self):
        """Atualiza métricas de risco"""
        try:
            if not self.daily_pnl:
                return
            
            # Calcula métricas
            self.risk_metrics = {
                'current_drawdown': self.calculate_drawdown(),
                'daily_loss': sum(pnl for pnl in self.daily_pnl if pnl < 0),
                'daily_gain': sum(pnl for pnl in self.daily_pnl if pnl > 0),
                'total_trades': len(self.position_history),
                'win_rate': self._calculate_win_rate(),
                'avg_win': self._calculate_avg_win(),
                'avg_loss': self._calculate_avg_loss(),
                'sharpe_ratio': self._calculate_sharpe_ratio()
            }
            
        except Exception as e:
            print(f"❌ Erro ao atualizar métricas: {e}")
    
    def _calculate_win_rate(self) -> float:
        """Calcula taxa de acerto"""
        try:
            if not self.position_history:
                return 0.5
            
            wins = sum(1 for pos in self.position_history if pos.get('profit', 0) > 0)
            return wins / len(self.position_history)
            
        except Exception as e:
            print(f"❌ Erro no cálculo de win rate: {e}")
            return 0.5
    
    def _calculate_avg_win(self) -> float:
        """Calcula ganho médio"""
        try:
            wins = [pos.get('profit', 0) for pos in self.position_history if pos.get('profit', 0) > 0]
            return np.mean(wins) if wins else 0.0
            
        except Exception as e:
            print(f"❌ Erro no cálculo de ganho médio: {e}")
            return 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calcula perda média"""
        try:
            losses = [abs(pos.get('profit', 0)) for pos in self.position_history if pos.get('profit', 0) < 0]
            return np.mean(losses) if losses else 0.0
            
        except Exception as e:
            print(f"❌ Erro no cálculo de perda média: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe ratio"""
        try:
            if not self.daily_pnl or len(self.daily_pnl) < 2:
                return 0.0
            
            returns = np.array(self.daily_pnl)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return mean_return / std_return
            
        except Exception as e:
            print(f"❌ Erro no cálculo de Sharpe ratio: {e}")
            return 0.0
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Retorna relatório completo de risco"""
        return {
            'limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_correlation': self.max_correlation,
                'max_positions': self.max_positions,
                'max_portfolio_risk': self.max_portfolio_risk
            },
            'current_status': {
                'current_drawdown': self.calculate_drawdown(),
                'daily_loss': sum(pnl for pnl in self.daily_pnl if pnl < 0),
                'active_positions': len(self.active_positions)
            },
            'performance_metrics': self.risk_metrics,
            'risk_level': self._get_risk_level()
        }
    
    def _get_risk_level(self) -> str:
        """Determina nível de risco atual"""
        try:
            drawdown = self.calculate_drawdown()
            
            if drawdown > 0.1:
                return "HIGH"
            elif drawdown > 0.05:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            print(f"❌ Erro na determinação do nível de risco: {e}")
            return "UNKNOWN"
    
    def reset(self):
        """Reseta o sistema"""
        self.daily_pnl = []
        self.position_history = []
        self.active_positions = []
        self.risk_metrics = {}
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        print("🔄 Risk Manager resetado") 