#!/usr/bin/env python3
"""
COLETOR DE DADOS
================

Módulo responsável pela coleta e validação de dados de mercado.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors

class DataCollector:
    """Coletor de dados de mercado"""
    
    def __init__(self, central_feature_engine=None, central_market_regime_system=None):
        """Inicializa o coletor de dados"""
        self.api_manager = None # Placeholder, will be initialized later if needed
        self.data_cache = {}
        self.last_update = None
        self.cache_duration = 60  # segundos
        self.central_feature_engine = central_feature_engine
        self.central_market_regime_system = central_market_regime_system
    
    def get_market_data(self, force_refresh=False):
        """Obtém dados de mercado com cache"""
        try:
            current_time = datetime.now()
            
            # Verifica se deve usar cache
            if (not force_refresh and 
                self.last_update and 
                (current_time - self.last_update).total_seconds() < self.cache_duration):
                return self.data_cache.get('market_data', pd.DataFrame())
            
            # Coleta dados da API
            data = self.api_manager.get_market_data()
            
            # Valida dados
            is_valid, validation_message = self.validate_market_data(data)
            
            if not is_valid:
                print(TerminalColors.warning(f"⚠️ Dados inválidos: {validation_message}"))
                # Retorna dados em cache se disponível
                return self.data_cache.get('market_data', pd.DataFrame())
            
            # Atualiza cache
            self.data_cache['market_data'] = data
            self.last_update = current_time
            
            print(TerminalColors.success(f"✅ Dados de mercado atualizados: {len(data)} registros"))
            
            return data
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter dados de mercado: {e}"))
            return self.data_cache.get('market_data', pd.DataFrame())
    
    def get_current_price(self):
        """Obtém preço atual"""
        try:
            return self.api_manager.get_current_price()
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter preço atual: {e}"))
            return 0.0
    
    def get_balance(self):
        """Obtém saldo da conta"""
        try:
            return self.api_manager.get_safe_balance()
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter saldo: {e}"))
            return 10000.0  # Saldo padrão para paper trading
    
    def validate_market_data(self, data):
        """Valida dados de mercado"""
        try:
            if data is None or data.empty:
                return False, "Dados vazios"
            
            # Verifica se tem colunas essenciais
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Colunas faltando: {missing_columns}"
            
            # Verifica se tem dados suficientes
            if len(data) < 20:
                return False, f"Dados insuficientes: {len(data)} registros (mínimo 20)"
            
            # Verifica valores negativos ou zero
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if (data[col] <= 0).any():
                    return False, f"Valores inválidos em {col}: zeros ou negativos"
            
            # Verifica se high >= low
            if (data['high'] < data['low']).any():
                return False, "Valores high < low detectados"
            
            # Verifica se close está entre high e low
            if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
                return False, "Valores close fora do range high-low"
            
            # Verifica se open está entre high e low
            if ((data['open'] > data['high']) | (data['open'] < data['low'])).any():
                return False, "Valores open fora do range high-low"
            
            # Verifica gaps muito grandes
            price_changes = data['close'].pct_change().abs()
            if (price_changes > 0.1).any():  # Mais de 10% de mudança
                print(TerminalColors.warning("⚠️ Gaps grandes detectados nos dados"))
            
            return True, "Dados válidos"
            
        except Exception as e:
            return False, f"Erro na validação: {e}"
    
    def prepare_advanced_features(self, data):
        """Prepara features avançadas usando sistema centralizado"""
        try:
            if self.central_feature_engine:
                return self.central_feature_engine.get_all_features(data, 'basic')
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
            rsi_values = self._calculate_rsi(df['close'].values)
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
    
    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Adiciona valores NaN no início para manter o tamanho
            rsi_full = np.full(len(prices), np.nan)
            rsi_full[period:] = rsi
            
            return rsi_full
        except:
            return np.full(len(prices), 50)
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bollinger Bands"""
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = ma + (std * std_dev)
            lower_band = ma - (std * std_dev)
            
            return upper_band, ma, lower_band
        except:
            return prices, prices, prices
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return macd, macd_signal, macd_histogram
        except:
            return prices * 0, prices * 0, prices * 0
    
    def _calculate_atr(self, data, period=14):
        """Calcula ATR (Average True Range)"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except:
            return data['close'] * 0.01
    
    def _calculate_adx(self, data, period=14):
        """Calcula ADX (Average Directional Index)"""
        try:
            # True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Directional Movement
            up_move = data['high'] - data['high'].shift()
            down_move = data['low'].shift() - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed values
            tr_smooth = true_range.rolling(window=period).mean()
            plus_di = (pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth) * 100
            minus_di = (pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth) * 100
            
            # ADX
            dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx
        except:
            return data['close'] * 0
    
    def get_market_summary(self, data):
        """Obtém resumo do mercado"""
        try:
            if data is None or data.empty:
                return {}
            
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
            
            # Mudança percentual
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Volatilidade
            volatility = data['close'].pct_change().std() * 100
            
            # Volume
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            
            # Tendência
            ma_short = data['close'].rolling(window=5).mean().iloc[-1]
            ma_long = data['close'].rolling(window=20).mean().iloc[-1]
            trend = "ALTA" if ma_short > ma_long else "BAIXA"
            
            return {
                'current_price': current_price,
                'price_change': price_change,
                'volatility': volatility,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'trend': trend,
                'data_points': len(data)
            }
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter resumo do mercado: {e}"))
            return {}
    
    def detect_market_regime(self, data):
        """Detecta regime de mercado usando sistema centralizado"""
        try:
            if self.central_market_regime_system:
                regime_info = self.central_market_regime_system.get_current_regime(data)
                return regime_info['regime']
            return self._detect_market_regime_fallback(data)
        except Exception as e:
            print(f"⚠️ Erro no sistema de market regime centralizado: {e}")
            return self._detect_market_regime_fallback(data)
    
    def _detect_market_regime_fallback(self, data):
        """Fallback para detecção de regime se sistema central não estiver disponível"""
        try:
            if data is None or len(data) < 20:
                return 'sideways'
            
            # Calcula volatilidade
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Calcula tendência
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Determina regime baseado em volatilidade e tendência
            if volatility > 0.03:  # Alta volatilidade
                return 'volatile'
            elif current_price > sma_20 > sma_50:  # Tendência de alta
                return 'trending_up'
            elif current_price < sma_20 < sma_50:  # Tendência de baixa
                return 'trending_down'
            else:
                return 'sideways'
            
        except Exception as e:
            print(f"⚠️ Erro no fallback de market regime: {e}")
            return 'sideways' 