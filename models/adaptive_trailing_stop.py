#!/usr/bin/env python3
"""
SISTEMA DE TRAILING STOP ADAPTATIVO
===================================

Implementa trailing stop dinâmico baseado em ATR:
- Stop loss baseado em Average True Range
- Trailing stop que protege lucros
- Ajuste dinâmico baseado na volatilidade
- Integração com análise técnica
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdaptiveTrailingStop:
    """
    Sistema de trailing stop adaptativo baseado em ATR
    """
    
    def __init__(self):
        self.atr_period = 20
        self.trailing_multiplier = 2.0
        self.profit_lock_multiplier = 1.5  # Multiplicador quando em lucro
        self.min_stop_distance = 0.005  # 0.5% mínimo
        self.profit_threshold = 0.01  # 1% para ativar trailing
        self.max_profit_threshold = 0.03  # 3% para stop mais apertado
        
        # Estado atual
        self.current_stop_loss = None
        self.entry_price = None
        self.position_type = None  # 'long' ou 'short'
        
        print("🚀 Adaptive Trailing Stop inicializado")
    
    def set_position(self, entry_price: float, position_type: str = 'long'):
        """
        Define posição inicial
        
        Args:
            entry_price: Preço de entrada
            position_type: Tipo de posição ('long' ou 'short')
        """
        self.entry_price = entry_price
        self.position_type = position_type
        self.current_stop_loss = None
        
        print(f"📊 Posição definida: {position_type} @ ${entry_price:.2f}")
    
    def calculate_stop_loss(self, current_price: float, market_data: pd.DataFrame) -> Optional[float]:
        """
        Calcula stop loss adaptativo baseado em ATR
        
        Args:
            current_price: Preço atual
            market_data: Dados de mercado OHLCV
            
        Returns:
            float: Preço do stop loss ou None se não aplicável
        """
        try:
            if self.entry_price is None or market_data.empty:
                return None
            
            # Calcula ATR
            atr = self._calculate_atr(market_data, self.atr_period)
            
            # Calcula lucro atual
            if self.position_type == 'long':
                current_profit_pct = (current_price - self.entry_price) / self.entry_price
            else:  # short
                current_profit_pct = (self.entry_price - current_price) / self.entry_price
            
            # ===== 1. STOP LOSS INICIAL =====
            if self.current_stop_loss is None:
                if self.position_type == 'long':
                    # Stop inicial abaixo do preço de entrada
                    base_stop = self.entry_price - (atr * self.trailing_multiplier)
                else:
                    # Stop inicial acima do preço de entrada
                    base_stop = self.entry_price + (atr * self.trailing_multiplier)
                
                self.current_stop_loss = base_stop
                return base_stop
            
            # ===== 2. TRAILING STOP (apenas se em lucro) =====
            if current_profit_pct > self.profit_threshold:
                # Ajusta multiplicador baseado no lucro
                if current_profit_pct > self.max_profit_threshold:
                    # Lucro alto = stop mais apertado
                    multiplier = self.profit_lock_multiplier
                else:
                    # Lucro moderado = stop normal
                    multiplier = self.trailing_multiplier
                
                # Calcula novo stop
                if self.position_type == 'long':
                    new_stop = current_price - (atr * multiplier)
                    # Só move para cima (protege lucros)
                    if new_stop > self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        return new_stop
                else:  # short
                    new_stop = current_price + (atr * multiplier)
                    # Só move para baixo (protege lucros)
                    if new_stop < self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        return new_stop
            
            # ===== 3. STOP LOSS MÍNIMO =====
            # Garante distância mínima do preço atual
            if self.position_type == 'long':
                min_stop = current_price * (1 - self.min_stop_distance)
                if self.current_stop_loss < min_stop:
                    self.current_stop_loss = min_stop
            else:
                min_stop = current_price * (1 + self.min_stop_distance)
                if self.current_stop_loss > min_stop:
                    self.current_stop_loss = min_stop
            
            return self.current_stop_loss
            
        except Exception as e:
            print(f"❌ Erro no cálculo de stop loss: {e}")
            return None
    
    def should_close_position(self, current_price: float, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Verifica se deve fechar a posição
        
        Args:
            current_price: Preço atual
            market_data: Dados de mercado
            
        Returns:
            Tuple[bool, str]: (deve fechar, razão)
        """
        try:
            stop_loss = self.calculate_stop_loss(current_price, market_data)
            
            if stop_loss is None:
                return False, "no_stop_loss"
            
            # Verifica se preço atingiu stop loss
            if self.position_type == 'long':
                if current_price <= stop_loss:
                    return True, f"stop_loss_hit_long_{stop_loss:.2f}"
            else:  # short
                if current_price >= stop_loss:
                    return True, f"stop_loss_hit_short_{stop_loss:.2f}"
            
            return False, "position_active"
            
        except Exception as e:
            print(f"❌ Erro na verificação de fechamento: {e}")
            return False, "error"
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calcula Average True Range"""
        try:
            if len(data) < period:
                return 0.02  # Valor padrão se dados insuficientes
            
            # Calcula True Range
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calcula ATR (média móvel exponencial)
            atr = np.mean(true_range[-period:])
            
            return atr
            
        except Exception as e:
            print(f"❌ Erro no cálculo ATR: {e}")
            return 0.02
    
    def get_stop_info(self) -> Dict[str, Any]:
        """Retorna informações do stop loss atual"""
        return {
            'current_stop_loss': self.current_stop_loss,
            'entry_price': self.entry_price,
            'position_type': self.position_type,
            'atr_period': self.atr_period,
            'trailing_multiplier': self.trailing_multiplier,
            'profit_lock_multiplier': self.profit_lock_multiplier
        }
    
    def reset(self):
        """Reseta o sistema"""
        self.current_stop_loss = None
        self.entry_price = None
        self.position_type = None
        print("🔄 Trailing Stop resetado")
    
    def update_parameters(self, atr_period: Optional[int] = None, 
                         trailing_multiplier: Optional[float] = None,
                         profit_lock_multiplier: Optional[float] = None):
        """
        Atualiza parâmetros do sistema
        
        Args:
            atr_period: Período do ATR
            trailing_multiplier: Multiplicador do trailing
            profit_lock_multiplier: Multiplicador quando em lucro
        """
        if atr_period is not None:
            self.atr_period = atr_period
        
        if trailing_multiplier is not None:
            self.trailing_multiplier = trailing_multiplier
        
        if profit_lock_multiplier is not None:
            self.profit_lock_multiplier = profit_lock_multiplier
        
        print(f"🔄 Parâmetros atualizados: ATR={self.atr_period}, Trailing={self.trailing_multiplier}, Lock={self.profit_lock_multiplier}") 