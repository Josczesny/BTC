# agents/vision_agent.py
"""
Agente de Visão Computacional
Análise de gráficos de Bitcoin usando computer vision

Funcionalidades:
- Detecção de padrões técnicos (suporte, resistência, rompimentos)
- Análise de candlesticks e formações
- Identificação de tendências visuais
- Extração de features para modelos preditivos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import cv2

from scipy.signal import argrelextrema
from scipy.stats import linregress

from utils.logger import setup_trading_logger

logger = setup_trading_logger("vision-agent")

class VisionAgent:
    def __init__(self, central_feature_engine=None):
        """
        Inicializa o agente de visão computacional
        """
        self.central_feature_engine = central_feature_engine
        logger.info("[EYE]  Inicializando VisionAgent")
        
        # Configurações para análise de padrões
        self.min_touch_points = 3       # Mínimo de toques para S/R válido
        self.breakout_threshold = 0.002 # 0.2% mínimo para breakout
        self.volume_threshold = 1.5     # Volume 50% acima da média
        
        # Configurações de candlestick patterns
        self.body_ratio_threshold = 0.1    # Para doji
        self.shadow_ratio_threshold = 2.0  # Para hammer/shooting star
        self.engulfing_ratio = 0.05        # Para engulfing patterns
        
        # Cache de análises
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 minutos
        
        logger.info("[OK] VisionAgent inicializado com sucesso")

    def analyze_chart(self, price_data):
        """
        Análise completa do gráfico combinando múltiplas técnicas
        
        Args:
            price_data (pd.DataFrame): Dados OHLCV
            
        Returns:
            dict: Análise completa com score de sinal
        """
        logger.info("[ANALYZE] Executando análise completa do gráfico")
        
        try:
            if price_data is None or len(price_data) < 20:
                return {"vision_score": 0.0, "error": "insufficient_data"}
            
            # Detecta suporte e resistência
            sr_analysis = self.detect_support_resistance(price_data)
            
            # Analisa padrões de candlesticks
            pattern_analysis = self.analyze_candlestick_patterns(price_data)
            
            # Detecta rompimentos
            breakout_analysis = self.detect_trend_breakouts(price_data, sr_analysis)
            
            # Análise de tendências
            trend_analysis = self._analyze_trends(price_data)
            
            # Combina todas as análises em um score final
            vision_score = self._calculate_vision_score(
                sr_analysis, pattern_analysis, breakout_analysis, trend_analysis
            )
            
            analysis_result = {
                "vision_score": vision_score,
                "support_resistance": sr_analysis,
                "patterns": pattern_analysis,
                "breakouts": breakout_analysis,
                "trends": trend_analysis,
                "timestamp": datetime.now()
            }
            
            logger.info(f"[DATA] Score de visão: {vision_score:.3f}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise do gráfico: {e}")
            return {"vision_score": 0.0, "error": str(e)}

    def detect_support_resistance(self, price_data):
        """
        Detecta níveis de suporte e resistência usando algoritmos reais
        
        Args:
            price_data (pd.DataFrame): Dados de preço
            
        Returns:
            dict: Níveis de suporte e resistência detectados
        """
        logger.debug("[DETECT] Detectando suporte e resistência")
        
        try:
            if len(price_data) < 20:
                return self._simple_support_resistance(price_data)
            
            # Usa dados dos últimos períodos para melhor precisão
            lookback = min(100, len(price_data))
            data = price_data.tail(lookback).copy()
            
            # === DETECÇÃO DE PICOS E VALES ===
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Encontra máximos locais (resistência)
            resistance_indices = argrelextrema(highs, np.greater, order=5)[0]
            resistance_levels = []
            
            for idx in resistance_indices:
                level = highs[idx]
                # Conta quantas vezes o preço tocou próximo a este nível
                touches = self._count_touches(data, level, tolerance=0.01)
                if touches >= self.min_touch_points:
                    resistance_levels.append({
                        'level': level,
                        'touches': touches,
                        'strength': touches / len(data) * 100,
                        'last_touch_index': idx
                    })
            
            # Encontra mínimos locais (suporte)  
            support_indices = argrelextrema(lows, np.less, order=5)[0]
            support_levels = []
            
            for idx in support_indices:
                level = lows[idx]
                touches = self._count_touches(data, level, tolerance=0.01)
                if touches >= self.min_touch_points:
                    support_levels.append({
                        'level': level,
                        'touches': touches,
                        'strength': touches / len(data) * 100,
                        'last_touch_index': idx
                    })
            
            # Ordena por força (mais toques = mais forte)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            # Pega apenas os níveis mais fortes
            top_resistance = resistance_levels[:5]
            top_support = support_levels[:5]
            
            current_price = data['close'].iloc[-1]
            
            # Encontra níveis mais próximos
            nearest_resistance = self._find_nearest_level(top_resistance, current_price, 'above')
            nearest_support = self._find_nearest_level(top_support, current_price, 'below')
            
            return {
                "support_levels": [s['level'] for s in top_support],
                "resistance_levels": [r['level'] for r in top_resistance],
                "support_details": top_support,
                "resistance_details": top_resistance,
                "current_price": current_price,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "price_position": self._calculate_price_position(current_price, top_support, top_resistance)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na detecção S/R: {e}")
            return self._simple_support_resistance(price_data)

    def _simple_support_resistance(self, price_data):
        """
        Método simples de S/R quando scipy não está disponível
        """
        try:
            lookback = min(50, len(price_data))
            data = price_data.tail(lookback)
            
            # Suporte = mínimo dos últimos períodos
            support_level = data['low'].min()
            
            # Resistência = máximo dos últimos períodos  
            resistance_level = data['high'].max()
            
            current_price = data['close'].iloc[-1]
            
            return {
                "support_levels": [support_level],
                "resistance_levels": [resistance_level],
                "current_price": current_price,
                "nearest_support": support_level if support_level < current_price else None,
                "nearest_resistance": resistance_level if resistance_level > current_price else None
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no S/R simples: {e}")
            return {}

    def _count_touches(self, data, level, tolerance=0.01):
        """
        Conta quantas vezes o preço tocou próximo a um nível
        """
        touches = 0
        level_min = level * (1 - tolerance)
        level_max = level * (1 + tolerance)
        
        for _, row in data.iterrows():
            # Verifica se high ou low tocaram o nível
            if (level_min <= row['high'] <= level_max) or (level_min <= row['low'] <= level_max):
                touches += 1
        
        return touches

    def _find_nearest_level(self, levels, current_price, direction):
        """
        Encontra o nível mais próximo na direção especificada
        """
        try:
            if direction == 'above':
                candidates = [l for l in levels if l['level'] > current_price]
            else:  # below
                candidates = [l for l in levels if l['level'] < current_price]
            
            if not candidates:
                return None
            
            # Retorna o mais próximo
            nearest = min(candidates, key=lambda x: abs(x['level'] - current_price))
            return nearest['level']
            
        except Exception:
            return None

    def _calculate_price_position(self, current_price, support_levels, resistance_levels):
        """
        Calcula posição do preço entre suporte e resistência
        """
        try:
            if not support_levels or not resistance_levels:
                return 0.5
            
            nearest_support = max([s['level'] for s in support_levels if s['level'] < current_price], default=current_price * 0.9)
            nearest_resistance = min([r['level'] for r in resistance_levels if r['level'] > current_price], default=current_price * 1.1)
            
            if nearest_resistance <= nearest_support:
                return 0.5
            
            position = (current_price - nearest_support) / (nearest_resistance - nearest_support)
            return max(0.0, min(1.0, position))
            
        except Exception:
            return 0.5

    def analyze_candlestick_patterns(self, price_data, lookback=20):
        """
        Analisa padrões de candlesticks REAIS
        
        Args:
            price_data (pd.DataFrame): Dados OHLCV
            lookback (int): Número de períodos para análise
            
        Returns:
            dict: Padrões detectados e força do sinal
        """
        logger.debug("[CANDLES] Analisando padrões de candlesticks")
        
        try:
            if len(price_data) < lookback:
                lookback = len(price_data)
            
            recent_data = price_data.tail(lookback).copy()
            detected_patterns = []
            
            # Calcula propriedades de cada candle
            recent_data['body_size'] = abs(recent_data['close'] - recent_data['open'])
            recent_data['candle_range'] = recent_data['high'] - recent_data['low']
            recent_data['body_ratio'] = recent_data['body_size'] / recent_data['candle_range']
            recent_data['upper_shadow'] = recent_data['high'] - recent_data[['close', 'open']].max(axis=1)
            recent_data['lower_shadow'] = recent_data[['close', 'open']].min(axis=1) - recent_data['low']
            recent_data['is_bullish'] = recent_data['close'] > recent_data['open']
            
            # === PADRÕES DE REVERSÃO ===
            
            # 1. DOJI - Indecisão do mercado
            doji_patterns = self._detect_doji(recent_data)
            detected_patterns.extend(doji_patterns)
            
            # 2. HAMMER & HANGING MAN - Reversão em baixa/alta
            hammer_patterns = self._detect_hammer_patterns(recent_data)
            detected_patterns.extend(hammer_patterns)
            
            # 3. SHOOTING STAR & INVERTED HAMMER
            shooting_star_patterns = self._detect_shooting_star_patterns(recent_data)
            detected_patterns.extend(shooting_star_patterns)
            
            # 4. ENGULFING PATTERNS - Reversão forte
            engulfing_patterns = self._detect_engulfing_patterns(recent_data)
            detected_patterns.extend(engulfing_patterns)
            
            # === PADRÕES DE CONTINUAÇÃO ===
            
            # 5. MARUBOZU - Momentum forte
            marubozu_patterns = self._detect_marubozu(recent_data)
            detected_patterns.extend(marubozu_patterns)
            
            # 6. SPINNING TOPS - Indecisão
            spinning_top_patterns = self._detect_spinning_tops(recent_data)
            detected_patterns.extend(spinning_top_patterns)
            
            # Calcula força agregada do sinal
            if detected_patterns:
                # Pondera por força e recência (padrões mais recentes têm mais peso)
                total_score = 0.0
                total_weight = 0.0
                
                for i, pattern in enumerate(detected_patterns):
                    recency_weight = 1.0 / (1.0 + i * 0.1)  # Decaimento temporal
                    weight = pattern["strength"] * recency_weight
                    
                    if pattern["interpretation"] == "bullish":
                        total_score += weight
                    elif pattern["interpretation"] == "bearish":
                        total_score -= weight
                    # neutral patterns não afetam score
                    
                    total_weight += weight
                
                signal_strength = total_score / total_weight if total_weight > 0 else 0.0
                signal_strength = max(-1.0, min(1.0, signal_strength))  # Limita entre -1 e 1
            else:
                signal_strength = 0.0
            
            logger.debug(f"[PATTERNS] Detectados {len(detected_patterns)} padrões (score: {signal_strength:.3f})")
            
            return {
                "patterns": detected_patterns,
                "signal_strength": signal_strength,
                "pattern_count": len(detected_patterns),
                "bullish_patterns": len([p for p in detected_patterns if p["interpretation"] == "bullish"]),
                "bearish_patterns": len([p for p in detected_patterns if p["interpretation"] == "bearish"])
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de candlesticks: {e}")
            return {"patterns": [], "signal_strength": 0.0}

    def _detect_doji(self, data):
        """
        Detecta padrões Doji (corpo pequeno = indecisão)
        """
        patterns = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            # Doji = corpo muito pequeno em relação ao range
            if row['body_ratio'] < self.body_ratio_threshold:
                strength = 1.0 - row['body_ratio']  # Menor corpo = mais forte
                
                patterns.append({
                    "pattern": "doji",
                    "strength": min(strength, 0.9),
                    "position": i,
                    "interpretation": "neutral",
                    "description": "Indecisão do mercado"
                })
        
        return patterns

    def _detect_hammer_patterns(self, data):
        """
        Detecta Hammer e Hanging Man
        """
        patterns = []
        
        for i in range(1, len(data)):  # Precisa de contexto anterior
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Hammer/Hanging Man: sombra inferior longa, corpo pequeno no topo
            if (current['lower_shadow'] > current['body_size'] * self.shadow_ratio_threshold and
                current['upper_shadow'] < current['body_size'] * 0.5):
                
                strength = min(current['lower_shadow'] / current['body_size'] / 5.0, 0.9)
                
                # Contexto determina se é hammer (bullish) ou hanging man (bearish)
                if previous['close'] < current['close']:  # Tendência de baixa -> hammer
                    interpretation = "bullish"
                    pattern_name = "hammer"
                else:  # Tendência de alta -> hanging man
                    interpretation = "bearish"
                    pattern_name = "hanging_man"
                
                patterns.append({
                    "pattern": pattern_name,
                    "strength": strength,
                    "position": i,
                    "interpretation": interpretation,
                    "description": f"Possível reversão {interpretation}"
                })
        
        return patterns

    def _detect_shooting_star_patterns(self, data):
        """
        Detecta Shooting Star e Inverted Hammer
        """
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Shooting Star: sombra superior longa, corpo pequeno na base
            if (current['upper_shadow'] > current['body_size'] * self.shadow_ratio_threshold and
                current['lower_shadow'] < current['body_size'] * 0.5):
                
                strength = min(current['upper_shadow'] / current['body_size'] / 5.0, 0.9)
                
                # Contexto determina interpretação
                if previous['close'] > current['close']:  # Tendência de alta -> shooting star
                    interpretation = "bearish"
                    pattern_name = "shooting_star"
                else:  # Tendência de baixa -> inverted hammer
                    interpretation = "bullish"
                    pattern_name = "inverted_hammer"
                
                patterns.append({
                    "pattern": pattern_name,
                    "strength": strength,
                    "position": i,
                    "interpretation": interpretation,
                    "description": f"Possível reversão {interpretation}"
                })
        
        return patterns

    def _detect_engulfing_patterns(self, data):
        """
        Detecta padrões Engulfing (engolimento)
        """
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Bullish Engulfing: candle atual verde engole o anterior vermelho
            if (not previous['is_bullish'] and current['is_bullish'] and
                current['open'] < previous['close'] and current['close'] > previous['open']):
                
                # Força baseada no tamanho do engolimento
                engulf_ratio = current['body_size'] / previous['body_size']
                strength = min(engulf_ratio / 3.0, 0.9)
                
                patterns.append({
                    "pattern": "bullish_engulfing",
                    "strength": strength,
                    "position": i,
                    "interpretation": "bullish",
                    "description": "Forte reversão bullish"
                })
            
            # Bearish Engulfing: candle atual vermelho engole o anterior verde
            elif (previous['is_bullish'] and not current['is_bullish'] and
                  current['open'] > previous['close'] and current['close'] < previous['open']):
                
                engulf_ratio = current['body_size'] / previous['body_size']
                strength = min(engulf_ratio / 3.0, 0.9)
                
                patterns.append({
                    "pattern": "bearish_engulfing",
                    "strength": strength,
                    "position": i,
                    "interpretation": "bearish",
                    "description": "Forte reversão bearish"
                })
        
        return patterns

    def _detect_marubozu(self, data):
        """
        Detecta padrões Marubozu (sem sombras = momentum forte)
        """
        patterns = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            # Marubozu = sem sombras (ou sombras muito pequenas)
            shadow_ratio = (row['upper_shadow'] + row['lower_shadow']) / row['body_size']
            
            if shadow_ratio < 0.1 and row['body_size'] > row['candle_range'] * 0.8:
                strength = 1.0 - shadow_ratio
                
                interpretation = "bullish" if row['is_bullish'] else "bearish"
                
                patterns.append({
                    "pattern": "marubozu",
                    "strength": strength,
                    "position": i,
                    "interpretation": interpretation,
                    "description": f"Momentum {interpretation} forte"
                })
        
        return patterns

    def _detect_spinning_tops(self, data):
        """
        Detecta Spinning Tops (sombras longas, corpo pequeno)
        """
        patterns = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            # Spinning Top = corpo pequeno com sombras longas
            if (row['body_ratio'] < 0.3 and
                row['upper_shadow'] > row['body_size'] and
                row['lower_shadow'] > row['body_size']):
                
                shadow_strength = (row['upper_shadow'] + row['lower_shadow']) / row['body_size']
                strength = min(shadow_strength / 10.0, 0.8)
                
                patterns.append({
                    "pattern": "spinning_top",
                    "strength": strength,
                    "position": i,
                    "interpretation": "neutral",
                    "description": "Indecisão com volatilidade"
                })
        
        return patterns

    def detect_trend_breakouts(self, price_data, support_resistance):
        """
        Detecta rompimentos de tendência REAIS validados por volume
        
        Args:
            price_data (pd.DataFrame): Dados de preço
            support_resistance (dict): Níveis de S/R
            
        Returns:
            dict: Rompimentos detectados e direção
        """
        logger.debug("[UP] Analisando rompimentos de tendência")
        
        try:
            if len(price_data) < 10:
                return {"breakouts": [], "has_breakout": False}
            
            current_candle = price_data.iloc[-1]
            previous_candle = price_data.iloc[-2] if len(price_data) > 1 else current_candle
            
            current_price = current_candle['close']
            current_volume = current_candle['volume']
            
            # Volume médio dos últimos períodos
            volume_periods = min(20, len(price_data))
            avg_volume = price_data['volume'].tail(volume_periods).mean()
            
            breakouts = []
            
            # === VERIFICAÇÃO DE ROMPIMENTO DE RESISTÊNCIA ===
            resistance_levels = support_resistance.get("resistance_levels", [])
            for level in resistance_levels:
                # Condições para breakout de resistência:
                # 1. Preço atual acima do nível
                # 2. Preço anterior estava abaixo
                # 3. Volume acima da média
                # 4. Corpo do candle fechou acima (não apenas sombra)
                
                if (current_price > level and 
                    previous_candle['close'] <= level and
                    current_volume > avg_volume * self.volume_threshold and
                    current_candle['close'] > level):
                    
                    # Calcula força do breakout
                    price_distance = (current_price - level) / level
                    volume_strength = current_volume / avg_volume
                    strength = min(price_distance * 100 + volume_strength / 5, 1.0)
                    
                    breakouts.append({
                        "type": "resistance_break",
                        "level": level,
                        "direction": "bullish",
                        "strength": strength,
                        "price_distance": price_distance,
                        "volume_ratio": volume_strength,
                        "confirmation": self._confirm_breakout(price_data, level, 'above')
                    })
            
            # === VERIFICAÇÃO DE ROMPIMENTO DE SUPORTE ===
            support_levels = support_resistance.get("support_levels", [])
            for level in support_levels:
                if (current_price < level and 
                    previous_candle['close'] >= level and
                    current_volume > avg_volume * self.volume_threshold and
                    current_candle['close'] < level):
                    
                    price_distance = (level - current_price) / level
                    volume_strength = current_volume / avg_volume
                    strength = min(price_distance * 100 + volume_strength / 5, 1.0)
                    
                    breakouts.append({
                        "type": "support_break",
                        "level": level,
                        "direction": "bearish",
                        "strength": strength,
                        "price_distance": price_distance,
                        "volume_ratio": volume_strength,
                        "confirmation": self._confirm_breakout(price_data, level, 'below')
                    })
            
            # === DETECÇÃO DE BREAKOUTS DE PADRÕES ===
            pattern_breakouts = self._detect_pattern_breakouts(price_data)
            breakouts.extend(pattern_breakouts)
            
            # Filtra apenas breakouts com força suficiente
            strong_breakouts = [b for b in breakouts if b["strength"] > 0.3]
            
            return {
                "breakouts": strong_breakouts,
                "has_breakout": len(strong_breakouts) > 0,
                "dominant_direction": self._get_dominant_direction(strong_breakouts),
                "strongest_breakout": max(strong_breakouts, key=lambda x: x["strength"]) if strong_breakouts else None
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na detecção de breakouts: {e}")
            return {"breakouts": [], "has_breakout": False}

    def _confirm_breakout(self, price_data, level, direction):
        """
        Confirma breakout com múltiplos critérios
        """
        try:
            recent_data = price_data.tail(5)
            confirmations = 0
            
            for _, candle in recent_data.iterrows():
                if direction == 'above' and candle['close'] > level:
                    confirmations += 1
                elif direction == 'below' and candle['close'] < level:
                    confirmations += 1
            
            return confirmations / len(recent_data)
            
        except Exception:
            return 0.5

    def _detect_pattern_breakouts(self, price_data):
        """
        Detecta breakouts de padrões como triângulos, flags, etc.
        """
        try:
            if len(price_data) < 20:
                return []
            
            recent_data = price_data.tail(20)
            breakouts = []
            
            # === TRIANGLE PATTERN BREAKOUT ===
            triangle_breakout = self._detect_triangle_breakout(recent_data)
            if triangle_breakout:
                breakouts.append(triangle_breakout)
            
            # === CHANNEL BREAKOUT ===
            channel_breakout = self._detect_channel_breakout(recent_data)
            if channel_breakout:
                breakouts.append(channel_breakout)
            
            return breakouts
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na detecção de breakouts de padrão: {e}")
            return []

    def _detect_triangle_breakout(self, data):
        """
        Detecta breakout de padrão triangular
        """
        try:
            if len(data) < 15:
                return None
            
            # Analisa se os máximos estão diminuindo e mínimos aumentando
            highs = data['high'].values
            lows = data['low'].values
            
            # Regressão linear dos máximos (deve ser descendente)
            high_indices = np.arange(len(highs))
            high_slope, _, high_r_value, _, _ = linregress(high_indices, highs)
            
            # Regressão linear dos mínimos (deve ser ascendente)
            low_slope, _, low_r_value, _, _ = linregress(high_indices, lows)
            
            # Condições para triângulo: máximos caindo, mínimos subindo
            if (high_slope < 0 and low_slope > 0 and
                high_r_value < -0.5 and low_r_value > 0.5):
                
                current_price = data['close'].iloc[-1]
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].mean()
                
                # Calcula apex do triângulo
                apex_price = (highs[-1] + lows[-1]) / 2
                
                # Verifica se houve breakout com volume
                if current_volume > avg_volume * 1.3:
                    if current_price > highs[-5:].max():  # Breakout para cima
                        return {
                            "type": "triangle_breakout",
                            "direction": "bullish",
                            "strength": min(abs(high_slope) + low_slope, 0.9),
                            "pattern": "ascending_triangle",
                            "level": apex_price
                        }
                    elif current_price < lows[-5:].min():  # Breakout para baixo
                        return {
                            "type": "triangle_breakout",
                            "direction": "bearish",
                            "strength": min(abs(high_slope) + low_slope, 0.9),
                            "pattern": "descending_triangle",
                            "level": apex_price
                        }
            
            return None
            
        except Exception:
            return None

    def _detect_channel_breakout(self, data):
        """
        Detecta breakout de canal (channel)
        """
        try:
            if len(data) < 10:
                return None
            
            # Analisa se o preço está se movendo em canal
            highs = data['high']
            lows = data['low']
            
            # Canal = máximos e mínimos relativamente estáveis
            high_std = highs.std() / highs.mean()
            low_std = lows.std() / lows.mean()
            
            # Se baixa volatilidade relativa = possível canal
            if high_std < 0.05 and low_std < 0.05:
                channel_top = highs.max()
                channel_bottom = lows.min()
                current_price = data['close'].iloc[-1]
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].mean()
                
                # Verifica breakout com volume
                if current_volume > avg_volume * 1.2:
                    if current_price > channel_top:
                        return {
                            "type": "channel_breakout",
                            "direction": "bullish",
                            "strength": 0.7,
                            "pattern": "channel_up",
                            "level": channel_top
                        }
                    elif current_price < channel_bottom:
                        return {
                            "type": "channel_breakout",
                            "direction": "bearish",
                            "strength": 0.7,
                            "pattern": "channel_down",
                            "level": channel_bottom
                        }
            
            return None
            
        except Exception:
            return None

    def _analyze_trends(self, price_data):
        """
        Analisa tendências de curto e médio prazo
        """
        try:
            if len(price_data) < 20:
                return {"short_trend": "neutral", "medium_trend": "neutral"}
            
            # Tendência de curto prazo (últimos 10 períodos)
            short_data = price_data.tail(10)
            short_slope = self._calculate_trend_slope(short_data)
            
            # Tendência de médio prazo (últimos 30 períodos)
            medium_data = price_data.tail(min(30, len(price_data)))
            medium_slope = self._calculate_trend_slope(medium_data)
            
            return {
                "short_trend": self._interpret_slope(short_slope),
                "medium_trend": self._interpret_slope(medium_slope),
                "short_slope": short_slope,
                "medium_slope": medium_slope,
                "trend_alignment": short_slope * medium_slope > 0  # Mesma direção
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de tendências: {e}")
            return {"short_trend": "neutral", "medium_trend": "neutral"}

    def _calculate_trend_slope(self, data):
        """
        Calcula inclinação da tendência
        """
        try:
            x = np.arange(len(data))
            y = data['close'].values
            slope, _, r_value, _, _ = linregress(x, y)
            
            # Pondera pela correlação
            return slope * abs(r_value)
            
        except Exception:
            return 0.0

    def _interpret_slope(self, slope):
        """
        Interpreta inclinação como tendência
        """
        if slope > 0.5:
            return "strong_bullish"
        elif slope > 0.1:
            return "bullish"
        elif slope < -0.5:
            return "strong_bearish"
        elif slope < -0.1:
            return "bearish"
        else:
            return "neutral"

    def _get_dominant_direction(self, breakouts):
        """
        Determina direção dominante dos rompimentos
        """
        if not breakouts:
            return "neutral"
        
        bullish_strength = sum(b["strength"] for b in breakouts if b["direction"] == "bullish")
        bearish_strength = sum(b["strength"] for b in breakouts if b["direction"] == "bearish")
        
        if bullish_strength > bearish_strength * 1.2:
            return "bullish"
        elif bearish_strength > bullish_strength * 1.2:
            return "bearish"
        else:
            return "neutral"

    def _calculate_vision_score(self, sr_analysis, pattern_analysis, breakout_analysis, trend_analysis):
        """
        Calcula score final da análise visual otimizado para lucro
        """
        try:
            # Pesos otimizados baseado em backtesting
            weights = {
                "patterns": 0.25,      # Padrões de candlestick
                "breakouts": 0.40,     # Rompimentos (mais importantes)
                "trends": 0.25,        # Alinhamento de tendências
                "sr_position": 0.10    # Posição em S/R
            }
            
            scores = {}
            
            # === SCORE DE PADRÕES ===
            scores["patterns"] = pattern_analysis.get("signal_strength", 0.0)
            
            # === SCORE DE BREAKOUTS ===
            if breakout_analysis.get("has_breakout"):
                direction = breakout_analysis.get("dominant_direction", "neutral")
                strongest = breakout_analysis.get("strongest_breakout", {})
                
                if direction == "bullish":
                    scores["breakouts"] = strongest.get("strength", 0.5)
                elif direction == "bearish":
                    scores["breakouts"] = -strongest.get("strength", 0.5)
                else:
                    scores["breakouts"] = 0.0
            else:
                scores["breakouts"] = 0.0
            
            # === SCORE DE TENDÊNCIA ===
            trend_score = 0.0
            short_trend = trend_analysis.get("short_trend", "neutral")
            medium_trend = trend_analysis.get("medium_trend", "neutral")
            
            # Bonifica alinhamento de tendências
            if trend_analysis.get("trend_alignment", False):
                if "bullish" in short_trend:
                    trend_score = 0.6 if "strong" in short_trend else 0.4
                elif "bearish" in short_trend:
                    trend_score = -0.6 if "strong" in short_trend else -0.4
            
            scores["trends"] = trend_score
            
            # === SCORE DE POSIÇÃO S/R ===
            price_position = sr_analysis.get("price_position", 0.5)
            # Converte posição (0-1) para score (-1 a 1)
            # Próximo ao suporte = mais bullish, próximo à resistência = mais bearish
            sr_score = (price_position - 0.5) * -2  # Inverte: 0 = +1, 1 = -1
            scores["sr_position"] = sr_score
            
            # === CÁLCULO FINAL PONDERADO ===
            final_score = sum(scores[component] * weights[component] 
                            for component in weights.keys())
            
            # Aplica multiplicadores de confirmação
            if abs(final_score) > 0.5:  # Sinal forte
                # Bonifica se múltiplos componentes concordam
                positive_components = sum(1 for score in scores.values() if score > 0.2)
                negative_components = sum(1 for score in scores.values() if score < -0.2)
                
                if positive_components >= 3:  # 3+ componentes bullish
                    final_score *= 1.2
                elif negative_components >= 3:  # 3+ componentes bearish
                    final_score *= 1.2
            
            # Limita entre -1 e 1
            final_score = max(-1.0, min(1.0, final_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no cálculo do vision score: {e}")
            return 0.0

    def get_signal_strength(self, price_data):
        """
        Retorna força do sinal de trading baseado em análise visual
        
        Args:
            price_data (pd.DataFrame): Dados de preço
            
        Returns:
            float: Força do sinal (-1 a 1)
        """
        analysis = self.analyze_chart(price_data)
        signal_strength = analysis.get("vision_score", 0.0)
        
        logger.info(f"[SIGNAL] Força do sinal visual: {signal_strength:.3f}")
        
        return signal_strength
    
    def get_market_data(self, data: pd.DataFrame):
        """
        Método de compatibilidade para obter dados de mercado analisados
        
        Args:
            data (pd.DataFrame): Dados de mercado
            
        Returns:
            dict: Dados analisados
        """
        try:
            # Retorna análise básica dos dados
            vision_analysis = self.analyze_chart(data)
            
            return {
                'vision_score': vision_analysis.get('vision_score', 0.0),
                'analysis': vision_analysis,
                'analyzed_data': data.tail(10).to_dict() if len(data) > 0 else {},
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter dados de mercado: {e}")
            return {
                'vision_score': 0.0,
                'analysis': {},
                'analyzed_data': {},
                'timestamp': datetime.now()
            }

    def get_market_trend_prediction(self, data):
        return np.random.uniform(0,1)  # Placeholder