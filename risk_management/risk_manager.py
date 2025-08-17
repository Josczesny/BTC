#!/usr/bin/env python3
"""
GERENCIADOR DE RISCO
====================

Módulo responsável pelo gerenciamento de risco do sistema de trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors
from models.central_market_regime_system import CentralMarketRegimeSystem
from models.dynamic_position_sizer import DynamicPositionSizer

class RiskManager:
    """Gerenciador de risco do sistema"""
    
    def __init__(self):
        """Inicializa o gerenciador de risco"""
        self.max_daily_loss = 100.0  # USDT
        self.max_position_size = 100.0  # USDT
        self.max_drawdown = 0.15  # 15%
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Histórico de risco
        self.risk_history = []
        self.drawdown_history = []
    
    def check_risk_limits(self, balance, proposed_trade_amount, current_pnl=0):
        """Verifica limites de risco"""
        try:
            # Reseta contadores diários se necessário
            self._reset_daily_counters_if_needed()
            
            # Verifica perda diária
            if self.daily_pnl + current_pnl < -self.max_daily_loss:
                print(TerminalColors.error(f"❌ Limite de perda diária atingido: ${self.daily_pnl:.2f}"))
                return False, "daily_loss_limit"
            
            # Verifica tamanho da posição
            if proposed_trade_amount > self.max_position_size:
                print(TerminalColors.error(f"❌ Tamanho de posição muito alto: ${proposed_trade_amount:.2f}"))
                return False, "position_size_limit"
            
            # Verifica drawdown
            if balance > 0:
                drawdown = abs(current_pnl) / balance
                if drawdown > self.max_drawdown:
                    print(TerminalColors.error(f"❌ Drawdown muito alto: {drawdown:.2%}"))
                    return False, "drawdown_limit"
            
            # Verifica número de trades diários
            if self.daily_trades >= 20:
                print(TerminalColors.warning("⚠️ Limite de trades diários atingido"))
                return False, "daily_trades_limit"
            
            return True, "risk_ok"
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na verificação de risco: {e}"))
            return False, "error"
    
    def update_risk_metrics(self, trade_pnl, trade_amount):
        """Atualiza métricas de risco"""
        try:
            # Atualiza P&L diário
            self.daily_pnl += trade_pnl
            self.daily_trades += 1
            
            # Registra histórico
            self.risk_history.append({
                'timestamp': datetime.now(),
                'trade_pnl': trade_pnl,
                'trade_amount': trade_amount,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades
            })
            
            # Atualiza drawdown se necessário
            if trade_pnl < 0:
                self._update_drawdown(trade_pnl)
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao atualizar métricas de risco: {e}"))
    
    def _reset_daily_counters_if_needed(self):
        """Reseta contadores diários se necessário"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            print(TerminalColors.info("🔄 Contadores diários resetados"))
    
    def _update_drawdown(self, trade_pnl):
        """Atualiza histórico de drawdown"""
        try:
            self.drawdown_history.append({
                'timestamp': datetime.now(),
                'loss': abs(trade_pnl),
                'daily_pnl': self.daily_pnl
            })
            
            # Mantém apenas os últimos 100 registros
            if len(self.drawdown_history) > 100:
                self.drawdown_history = self.drawdown_history[-100:]
                
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao atualizar drawdown: {e}"))
    
    def calculate_position_size(self, balance, confidence, volatility):
        """Calcula tamanho da posição usando sistema centralizado"""
        try:
            # Usa sistema centralizado se disponível
            position_sizer = DynamicPositionSizer()
            
            signal_strength = confidence  # Usa confiança como signal strength
            market_conditions = {
                'volatility': volatility,
                'trend': 'neutral',
                'confidence': confidence
            }
            
            position_size = position_sizer.calculate_position_size(
                signal_strength, market_conditions, balance
            )
            return position_size
            
        except Exception as e:
            print(f"⚠️ Erro no sistema de position sizing centralizado: {e}")
            return 10.0  # Valor padrão
    
    def analyze_market_risk(self, market_data):
        """Analisa risco do mercado"""
        try:
            if market_data is None or market_data.empty:
                return {'risk_level': 'HIGH', 'volatility': 0.05, 'recommendation': 'wait'}
            
            # Calcula volatilidade
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calcula correlação serial
            autocorr = returns.autocorr()
            
            # Calcula skewness
            skewness = returns.skew()
            
            # Determina nível de risco
            risk_level = 'LOW'
            if volatility > 0.03:
                risk_level = 'HIGH'
            elif volatility > 0.02:
                risk_level = 'MEDIUM'
            
            # Recomendação baseada no risco
            recommendation = 'trade'
            if risk_level == 'HIGH' and abs(autocorr) > 0.1:
                recommendation = 'wait'
            elif risk_level == 'HIGH' and abs(skewness) > 1.0:
                recommendation = 'reduce_size'
            
            return {
                'risk_level': risk_level,
                'volatility': volatility,
                'autocorrelation': autocorr,
                'skewness': skewness,
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro na análise de risco de mercado: {e}"))
            return {'risk_level': 'HIGH', 'volatility': 0.05, 'recommendation': 'wait'}
    
    def get_risk_statistics(self):
        """Obtém estatísticas de risco"""
        try:
            if not self.risk_history:
                return {}
            
            # Estatísticas básicas
            total_trades = len(self.risk_history)
            profitable_trades = len([h for h in self.risk_history if h['trade_pnl'] > 0])
            
            # P&L total
            total_pnl = sum([h['trade_pnl'] for h in self.risk_history])
            
            # Maior perda
            max_loss = min([h['trade_pnl'] for h in self.risk_history])
            
            # Maior ganho
            max_gain = max([h['trade_pnl'] for h in self.risk_history])
            
            # Drawdown máximo
            max_drawdown = 0
            if self.drawdown_history:
                max_drawdown = max([d['loss'] for d in self.drawdown_history])
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'max_drawdown': max_drawdown,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades
            }
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao obter estatísticas de risco: {e}"))
            return {}
    
    def should_pause_trading(self):
        """Verifica se deve pausar o trading"""
        try:
            # Pausa se perda diária alta
            if self.daily_pnl < -self.max_daily_loss * 0.8:
                return True, "daily_loss_approaching"
            
            # Pausa se muitos trades perdidos seguidos
            recent_trades = self.risk_history[-10:] if len(self.risk_history) >= 10 else self.risk_history
            if len(recent_trades) >= 5:
                recent_losses = len([t for t in recent_trades if t['trade_pnl'] < 0])
                if recent_losses >= 4:  # 4 de 5 trades perdidos
                    return True, "consecutive_losses"
            
            # Pausa se drawdown muito alto
            if self.drawdown_history:
                recent_drawdown = sum([d['loss'] for d in self.drawdown_history[-5:]])
                if recent_drawdown > self.max_daily_loss * 0.5:
                    return True, "high_drawdown"
            
            return False, "ok"
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro na verificação de pausa: {e}"))
            return False, "error"
    
    def get_risk_recommendations(self):
        """Obtém recomendações de risco"""
        try:
            recommendations = []
            
            # Verifica perda diária
            if self.daily_pnl < -self.max_daily_loss * 0.5:
                recommendations.append("Reduzir tamanho das posições - perda diária alta")
            
            # Verifica número de trades
            if self.daily_trades > 15:
                recommendations.append("Pausar trading - muitos trades hoje")
            
            # Verifica drawdown
            if self.drawdown_history:
                recent_drawdown = sum([d['loss'] for d in self.drawdown_history[-3:]])
                if recent_drawdown > 50:
                    recommendations.append("Aumentar stop loss - drawdown recente alto")
            
            # Verifica volatilidade
            if len(self.risk_history) >= 5:
                recent_volatility = np.std([h['trade_pnl'] for h in self.risk_history[-5:]])
                if recent_volatility > 20:
                    recommendations.append("Reduzir exposição - alta volatilidade recente")
            
            return recommendations
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao obter recomendações: {e}"))
            return [] 