#!/usr/bin/env python3
"""
SISTEMA AVANÇADO DE FEATURE ENGINEERING V2.0
============================================

Implementa 20+ features sofisticadas para maximizar precisão da IA:
- Features de preços e retornos
- Indicadores técnicos avançados
- Análise de mercado e regime
- Features de machine learning
- Análise de volatilidade dinâmica
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineerV2:
    """
    Sistema avançado de feature engineering com 20+ features
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
        self.feature_importance = {}
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria conjunto completo de features avançadas (20+ features)
        """
        if data is None or len(data) < 50:
            return pd.DataFrame()
            
        features_df = data.copy()
        
        # ===== 1. FEATURES DE PREÇOS E RETORNOS =====
        features_df = self._add_price_features(features_df)
        
        # ===== 2. FEATURES DE VOLATILIDADE =====
        features_df = self._add_volatility_features(features_df)
        
        # ===== 3. FEATURES DE VOLUME =====
        features_df = self._add_volume_features(features_df)
        
        # ===== 4. INDICADORES TÉCNICOS AVANÇADOS =====
        features_df = self._add_technical_indicators(features_df)
        
        # ===== 5. FEATURES DE TENDÊNCIA =====
        features_df = self._add_trend_features(features_df)
        
        # ===== 6. FEATURES DE MOMENTUM =====
        features_df = self._add_momentum_features(features_df)
        
        # ===== 7. FEATURES DE REGIME DE MERCADO =====
        features_df = self._add_market_regime_features(features_df)
        
        # ===== 8. FEATURES DE MACHINE LEARNING =====
        features_df = self._add_ml_features(features_df)
        
        # Remove NaN e retorna
        features_df = features_df.dropna()
        
        # Registra features criadas
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de preços e retornos"""
        
        # Retornos em diferentes períodos
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Price acceleration
        df['price_acceleration'] = df['return_1'].diff()
        
        # High-Low ratio
        df['hl_ratio'] = df['high'] / df['low']
        
        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilidade dinâmica"""
        
        # Volatilidade rolling
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Realized volatility
        df['realized_volatility'] = df['log_return'].rolling(20).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
        )
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            (0.5 * (np.log(df['high'] / df['low']) ** 2) - 
             (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)).rolling(20).mean()
        )
        
        # Volatility ratio
        df['vol_ratio'] = df['volatility_10'] / df['volatility_50']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volume e liquidez"""
        
        # Volume momentum
        for period in [5, 10, 20]:
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume price trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Money flow index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # On-balance volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume weighted average price
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicadores técnicos avançados"""
        
        # RSI multi-timeframe
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de tendência"""
        
        # Médias móveis
        for period in [10, 20, 50, 100]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Trend strength
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Price vs moving averages
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        df['price_vs_ema20'] = df['close'] / df['ema_20'] - 1
        
        # Moving average crossovers
        df['sma_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['ema_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum"""
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        # MOM (Momentum)
        for period in [5, 10, 20]:
            df[f'mom_{period}'] = talib.MOM(df['close'], timeperiod=period)
        
        # CMO (Chande Momentum Oscillator)
        df['cmo'] = talib.CMO(df['close'], timeperiod=14)
        
        # TRIX
        df['trix'] = talib.TRIX(df['close'], timeperiod=30)
        
        # ULTOSC (Ultimate Oscillator)
        df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de regime de mercado"""
        
        # Market regime detection
        volatility = df['close'].pct_change().rolling(20).std()
        trend = df['close'].rolling(20).mean().pct_change(20)
        
        # Regime classification
        df['market_regime'] = np.where(
            (volatility > volatility.quantile(0.7)) & (trend > trend.quantile(0.7)),
            'high_vol_bull',
            np.where(
                (volatility > volatility.quantile(0.7)) & (trend < trend.quantile(0.3)),
                'high_vol_bear',
                np.where(
                    (volatility < volatility.quantile(0.3)) & (trend > trend.quantile(0.7)),
                    'low_vol_bull',
                    np.where(
                        (volatility < volatility.quantile(0.3)) & (trend < trend.quantile(0.3)),
                        'low_vol_bear',
                        'sideways'
                    )
                )
            )
        )
        
        # Regime encoding
        regime_map = {
            'high_vol_bull': 1.0,
            'high_vol_bear': -1.0,
            'low_vol_bull': 0.5,
            'low_vol_bear': -0.5,
            'sideways': 0.0
        }
        df['regime_encoded'] = df['market_regime'].map(regime_map)
        
        # Volatility regime
        df['vol_regime'] = np.where(volatility > volatility.quantile(0.7), 'high', 'low')
        
        return df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de machine learning"""
        
        # Price patterns (simplified)
        df['price_pattern'] = np.where(
            (df['close'] > df['open']) & (df['close'] > df['close'].shift(1)),
            'bullish',
            np.where(
                (df['close'] < df['open']) & (df['close'] < df['close'].shift(1)),
                'bearish',
                'neutral'
            )
        )
        
        # Pattern encoding
        pattern_map = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
        df['pattern_encoded'] = df['price_pattern'].map(pattern_map)
        
        # Support and resistance levels (simplified)
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Mean reversion signal
        df['mean_reversion'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Momentum divergence
        df['momentum_divergence'] = df['rsi_14'] - df['close'].pct_change(14) * 100
        
        return df
    
    def get_feature_summary(self) -> Dict:
        """Retorna resumo das features criadas"""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_categories': {
                'price_features': [f for f in self.feature_names if 'return' in f or 'momentum' in f or 'price' in f],
                'volatility_features': [f for f in self.feature_names if 'vol' in f],
                'volume_features': [f for f in self.feature_names if 'volume' in f or 'obv' in f or 'mfi' in f],
                'technical_features': [f for f in self.feature_names if 'rsi' in f or 'macd' in f or 'bb' in f or 'stoch' in f],
                'trend_features': [f for f in self.feature_names if 'sma' in f or 'ema' in f or 'trend' in f or 'adx' in f],
                'momentum_features': [f for f in self.feature_names if 'roc' in f or 'mom' in f or 'cmo' in f or 'trix' in f],
                'regime_features': [f for f in self.feature_names if 'regime' in f],
                'ml_features': [f for f in self.feature_names if 'pattern' in f or 'support' in f or 'resistance' in f or 'mean' in f]
            }
        } 