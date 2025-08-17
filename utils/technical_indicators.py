#!/usr/bin/env python3
"""
SISTEMA CENTRALIZADO DE INDICADORES TÉCNICOS
============================================

Implementa todos os indicadores técnicos de forma centralizada para eliminar redundâncias.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False

class TechnicalIndicators:
    """
    Sistema centralizado de indicadores técnicos
    Elimina redundâncias em todo o projeto
    """
    
    @staticmethod
    def calculate_rsi(prices: Union[pd.Series, np.ndarray, List[float]], 
                     period: int = 14) -> np.ndarray:
        """
        Calcula RSI (Relative Strength Index) de forma otimizada
        
        Args:
            prices: Série de preços (pandas Series, numpy array ou lista)
            period: Período para cálculo (padrão: 14)
            
        Returns:
            Array numpy com valores de RSI
        """
        try:
            # Converte para numpy array se necessário
            if isinstance(prices, pd.Series):
                prices = prices.values
            elif isinstance(prices, list):
                prices = np.array(prices)
            
            if len(prices) < period + 1:
                return np.array([50.0] * len(prices))  # Valor neutro se dados insuficientes
            
            # Calcula diferenças
            deltas = np.diff(prices)
            
            # Separa ganhos e perdas
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calcula médias móveis exponenciais
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
            
            # Evita divisão por zero
            avg_losses = np.where(avg_losses == 0, 1e-10, avg_losses)
            
            # Calcula RS e RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Adiciona valores iniciais (período - 1 valores neutros)
            initial_values = np.full(period - 1, 50.0)
            rsi_complete = np.concatenate([initial_values, rsi])
            
            return rsi_complete
            
        except Exception as e:
            print(f"❌ Erro calculando RSI: {e}")
            return np.array([50.0] * len(prices))
    
    @staticmethod
    def calculate_macd(prices: Union[pd.Series, np.ndarray], 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula MACD (Moving Average Convergence Divergence)
        
        Returns:
            Tuple (MACD, Signal, Histogram)
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < slow_period:
                return np.array([]), np.array([]), np.array([])
            
            # Calcula EMAs
            ema_fast = TechnicalIndicators._calculate_ema(prices, fast_period)
            ema_slow = TechnicalIndicators._calculate_ema(prices, slow_period)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = TechnicalIndicators._calculate_ema(macd_line, signal_period)
            
            # Histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            print(f"❌ Erro calculando MACD: {e}")
            return np.array([]), np.array([]), np.array([])
    
    @staticmethod
    def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray], 
                                 period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula Bollinger Bands
        
        Returns:
            Tuple (Upper Band, Middle Band, Lower Band)
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.array([]), np.array([]), np.array([])
            
            # Média móvel simples
            middle_band = np.convolve(prices, np.ones(period)/period, mode='valid')
            
            # Desvio padrão
            std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
            
            # Bands
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            # Adiciona valores iniciais
            initial_upper = np.full(period-1, prices[0])
            initial_middle = np.full(period-1, prices[0])
            initial_lower = np.full(period-1, prices[0])
            
            upper_complete = np.concatenate([initial_upper, upper_band])
            middle_complete = np.concatenate([initial_middle, middle_band])
            lower_complete = np.concatenate([initial_lower, lower_band])
            
            return upper_complete, middle_complete, lower_complete
            
        except Exception as e:
            print(f"❌ Erro calculando Bollinger Bands: {e}")
            return np.array([]), np.array([]), np.array([])
    
    @staticmethod
    def calculate_sma(prices: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """Calcula Simple Moving Average"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.array([prices[0]] * len(prices))
            
            sma = np.convolve(prices, np.ones(period)/period, mode='valid')
            initial_values = np.full(period-1, prices[0])
            return np.concatenate([initial_values, sma])
            
        except Exception as e:
            print(f"❌ Erro calculando SMA: {e}")
            return np.array([prices[0]] * len(prices))
    
    @staticmethod
    def calculate_ema(prices: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """Calcula Exponential Moving Average"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.array([prices[0]] * len(prices))
            
            return TechnicalIndicators._calculate_ema(prices, period)
            
        except Exception as e:
            print(f"❌ Erro calculando EMA: {e}")
            return np.array([prices[0]] * len(prices))
    
    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Implementação interna de EMA"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_volatility(prices: Union[pd.Series, np.ndarray], 
                           period: int = 20) -> np.ndarray:
        """Calcula volatilidade (desvio padrão dos retornos)"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period + 1:
                return np.array([0.0] * len(prices))
            
            # Calcula retornos
            returns = np.diff(prices) / prices[:-1]
            
            # Calcula volatilidade móvel
            volatility = np.array([np.std(returns[i:i+period]) for i in range(len(returns)-period+1)])
            
            # Adiciona valores iniciais
            initial_values = np.full(period, 0.0)
            return np.concatenate([initial_values, volatility])
            
        except Exception as e:
            print(f"❌ Erro calculando volatilidade: {e}")
            return np.array([0.0] * len(prices))
    
    @staticmethod
    def calculate_momentum(prices: Union[pd.Series, np.ndarray], 
                          period: int = 10) -> np.ndarray:
        """Calcula momentum (diferença entre preço atual e preço n períodos atrás)"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.array([0.0] * len(prices))
            
            momentum = np.zeros_like(prices)
            momentum[period:] = prices[period:] - prices[:-period]
            
            return momentum
            
        except Exception as e:
            print(f"❌ Erro calculando momentum: {e}")
            return np.array([0.0] * len(prices))

def calculate_atr(df, period=14):
    if TA_LIB_AVAILABLE:
        return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

def calculate_adx(df, period=14):
    if TA_LIB_AVAILABLE:
        return talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        return pd.Series([None]*len(df))

def calculate_cci(df, period=20):
    if TA_LIB_AVAILABLE:
        return talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
        return (tp - ma) / (0.015 * md)

def calculate_obv(df):
    if TA_LIB_AVAILABLE:
        return talib.OBV(df['close'], df['volume'])
    else:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

def calculate_williams_r(df, period=14):
    if TA_LIB_AVAILABLE:
        return talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        return -100 * (highest_high - df['close']) / (highest_high - lowest_low)

def calculate_roc(df, period=10):
    if TA_LIB_AVAILABLE:
        return talib.ROC(df['close'], timeperiod=period)
    else:
        return df['close'].pct_change(periods=period) * 100

def calculate_mom(df, period=10):
    if TA_LIB_AVAILABLE:
        return talib.MOM(df['close'], timeperiod=period)
    else:
        return df['close'].diff(periods=period)

def calculate_trix(df, period=15):
    if TA_LIB_AVAILABLE:
        return talib.TRIX(df['close'], timeperiod=period)
    else:
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change() * 100

def calculate_ultosc(df, s1=7, s2=14, s3=28):
    if TA_LIB_AVAILABLE:
        return talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=s1, timeperiod2=s2, timeperiod3=s3)
    else:
        bp = df['close'] - df[['low', 'close']].min(axis=1)
        tr = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        avg7 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
        avg14 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
        avg28 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

# Instância global para uso em todo o projeto
technical_indicators = TechnicalIndicators() 