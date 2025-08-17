"""
Agente de Tomada de Decisão
Orquestra sinais de todos os agentes para decisões de trading otimizadas

Funcionalidades:
- Combina sinais de previsão, notícias e análise visual
- Sistema de scoring dinâmico baseado em confiança
- Gestão de risco otimizada para máximo lucro
- Validação cruzada de sinais para reduzir falsos positivos
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

from agents.prediction_agent import PredictionAgent
from agents.news_agent import NewsAgent
from agents.vision_agent import VisionAgent
from utils.logger import setup_trading_logger

logger = setup_trading_logger("decision-agent")

class DecisionAgent:
    def __init__(self, mode: str = "testnet", prediction_agent=None, news_agent=None, vision_agent=None, central_ensemble_system=None, central_market_regime_system=None):
        """
        Inicializa o agente de decisão
        """
        self.central_ensemble_system = central_ensemble_system
        self.central_market_regime_system = central_market_regime_system
        logger.info("[BRAIN] Inicializando DecisionAgent")
        
        # Inicializa sub-agentes
        try:
            self.prediction_agent = prediction_agent or PredictionAgent()
            self.news_agent = news_agent or NewsAgent()
            self.vision_agent = vision_agent or VisionAgent()
            logger.info("[OK] Sub-agentes inicializados")
        except Exception as e:
            logger.error(f"[ERROR] Erro ao inicializar sub-agentes: {e}")
            self.prediction_agent = None
            self.news_agent = None
            self.vision_agent = None
        
        # === CONFIGURAÇÕES BASEADAS NO MODO ===
        
        # Pesos dos sinais (otimizados via backtesting)
        self.signal_weights = {
            'prediction': 0.45,    # Modelos preditivos têm maior peso
            'vision': 0.35,        # Análise técnica é crucial
            'news': 0.20          # Sentimento como confirmação
        }
        
        if mode in ["testnet", "backtest", "paper"]:
            # MODO TREINAMENTO INTENSIVO - THRESHOLDS BAIXOS
            self.buy_threshold = 0.05      # MUITO BAIXO para máximo aprendizado
            self.sell_threshold = -0.05    # MUITO BAIXO para máximo aprendizado
            self.strong_signal_threshold = 0.25  # Sinal forte mais baixo
            self.min_confidence = 0.05      # Confiança mínima muito baixa
            self.min_agreeing_signals = 1  # Apenas 1 sinal concordante
            logger.info("DECISION AGENT - MODO TREINAMENTO INTENSIVO (PAPER)")
        else:
            # MODO TRADING REAL - CONSERVADOR
            self.buy_threshold = 0.35      # Mais conservador para BUY
            self.sell_threshold = -0.35    # Mais conservador para SELL
            self.strong_signal_threshold = 0.65  # Sinal muito forte
            self.min_confidence = 0.3      # Confiança mínima
            self.min_agreeing_signals = 2  # Mínimo de sinais concordantes
            logger.info("🛡️ DECISION AGENT - MODO TRADING REAL")
        
        # Configurações de risco
        self.max_position_size = 0.8   # Máximo 80% do capital
        self.risk_per_trade = 0.02     # 2% de risco por trade
        
        # Sistema de validação cruzada
        self.signal_timeout = 300      # 5 minutos de validade do sinal
        
        # Cache de decisões
        self.last_decision = None
        self.decision_history = []
        self.signal_cache = {}
        
        # Metrics para otimização contínua
        self.performance_metrics = {
            'total_decisions': 0,
            'profitable_decisions': 0,
            'accuracy': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info("[OK] DecisionAgent inicializado com configurações otimizadas")

    def analyze_market_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa todos os sinais do mercado
        
        Args:
            market_data (pd.DataFrame): Dados de mercado atualizados
            
        Returns:
            dict: Análise completa de sinais
        """
        logger.info("[DATA] Analisando sinais do mercado")
        
        try:
            if market_data is None or len(market_data) < 20:
                logger.warning("[WARN]  Dados insuficientes para análise")
                return self._empty_signal_analysis()
            
            signals = {}
            
            # === SINAL DE PREVISÃO ===
            if self.prediction_agent:
                try:
                    prediction_result = self.prediction_agent.predict_next_move(market_data)
                    prediction_strength = self.prediction_agent.get_signal_strength(market_data)
                    
                    signals['prediction'] = {
                        'strength': prediction_strength,
                        'confidence': prediction_result.get('confidence', 0.0),
                        'direction': prediction_result.get('direction', 'neutral'),
                        'details': prediction_result,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"[UP] Sinal previsão: {prediction_strength:.3f}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Erro no sinal de previsão: {e}")
                    signals['prediction'] = self._empty_signal()
            
            # === SINAL DE NOTÍCIAS ===
            if self.news_agent:
                try:
                    news_strength = self.news_agent.get_signal_strength()
                    news_sentiment = self.news_agent.get_market_sentiment_score()
                    
                    signals['news'] = {
                        'strength': news_strength,
                        'confidence': news_sentiment.get('confidence', 0.0),
                        'direction': self._interpret_sentiment_direction(news_strength),
                        'details': news_sentiment,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"[NEWS2] Sinal notícias: {news_strength:.3f}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Erro no sinal de notícias: {e}")
                    signals['news'] = self._empty_signal()
            
            # === SINAL DE VISÃO ===
            if self.vision_agent:
                try:
                    vision_strength = self.vision_agent.get_signal_strength(market_data)
                    vision_analysis = self.vision_agent.analyze_chart(market_data)
                    
                    signals['vision'] = {
                        'strength': vision_strength,
                        'confidence': min(abs(vision_strength), 0.9),
                        'direction': self._interpret_vision_direction(vision_strength),
                        'details': vision_analysis,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"[EYE]  Sinal visão: {vision_strength:.3f}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Erro no sinal de visão: {e}")
                    signals['vision'] = self._empty_signal()
            
            # === ANÁLISE DE CONSENSO ===
            consensus_analysis = self._analyze_signal_consensus(signals)
            
            # Cache dos sinais
            self.signal_cache = {
                'signals': signals,
                'consensus': consensus_analysis,
                'timestamp': datetime.now()
            }
            
            return {
                'individual_signals': signals,
                'consensus': consensus_analysis,
                'market_data_timestamp': market_data.index[-1] if len(market_data) > 0 else datetime.now(),
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de sinais: {e}")
            return self._empty_signal_analysis()

    def make_decision(self, market_data: pd.DataFrame, current_position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Método de compatibilidade para make_decision
        
        Args:
            market_data (pd.DataFrame): Dados de mercado
            current_position (dict): Posição atual (se houver)
            
        Returns:
            dict: Decisão de trading com detalhes
        """
        return self.make_trading_decision(market_data, current_position)

    def make_trading_decision(self, market_data: pd.DataFrame, current_position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Toma decisão de trading baseada em todos os sinais
        
        Args:
            market_data (pd.DataFrame): Dados de mercado
            current_position (dict): Posição atual (se houver)
            
        Returns:
            dict: Decisão de trading com detalhes
        """
        logger.info("[TARGET] Tomando decisão de trading")
        
        try:
            # Analisa sinais atuais
            signal_analysis = self.analyze_market_signals(market_data)
            consensus = signal_analysis['consensus']
            
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0
            
            # === VALIDAÇÕES INICIAIS ===
            
            # Verifica qualidade dos sinais
            if consensus['confidence'] < self.min_confidence:
                return self._create_decision(
                    action='HOLD',
                    confidence=consensus['confidence'],
                    reason='Baixa confiança nos sinais',
                    current_price=current_price,
                    signal_analysis=signal_analysis
                )
            
            # Verifica consenso mínimo
            if consensus['agreement_level'] < self.min_agreeing_signals:
                return self._create_decision(
                    action='HOLD',
                    confidence=consensus['confidence'],
                    reason='Falta de consenso entre sinais',
                    current_price=current_price,
                    signal_analysis=signal_analysis
                )
            
            # === LÓGICA DE DECISÃO OTIMIZADA ===
            
            overall_signal = consensus['consensus_signal']
            confidence = consensus['confidence']
            
            # Ajustes baseados na posição atual
            position_adjustment = self._calculate_position_adjustment(current_position, overall_signal, current_price)
            adjusted_signal = overall_signal * position_adjustment
            
            # Decisão base
            if adjusted_signal >= self.strong_signal_threshold:
                # SINAL MUITO FORTE - Posição grande
                action = 'STRONG_BUY'
                position_size = min(self.max_position_size, confidence * 0.8)
                
            elif adjusted_signal >= self.buy_threshold:
                # SINAL DE COMPRA - Posição moderada
                action = 'BUY'
                position_size = min(self.max_position_size * 0.6, confidence * 0.6)
                
            elif adjusted_signal <= -self.strong_signal_threshold:
                # SINAL MUITO NEGATIVO - Venda forte
                action = 'STRONG_SELL'
                position_size = 1.0  # Vende tudo
                
            elif adjusted_signal <= self.sell_threshold:
                # SINAL DE VENDA - Venda parcial ou total
                action = 'SELL'
                position_size = 0.8  # Vende 80%
                
            else:
                # SINAL NEUTRO - Manter posição
                action = 'HOLD'
                position_size = 0.0
            
            # === VALIDAÇÕES FINAIS DE RISCO ===
            
            # Aplica gestão de risco
            risk_adjusted_decision = self._apply_risk_management(
                action, position_size, confidence, current_price, market_data
            )
            
            # Cria decisão final
            decision = self._create_decision(
                action=risk_adjusted_decision['action'],
                confidence=confidence,
                position_size=risk_adjusted_decision['position_size'],
                reason=risk_adjusted_decision['reason'],
                current_price=current_price,
                signal_analysis=signal_analysis,
                risk_assessment=risk_adjusted_decision.get('risk_assessment', {})
            )
            
            # Registra decisão
            self._record_decision(decision)
            
            logger.info(f"[TARGET] Decisão: {decision['action']} | Confiança: {confidence:.3f} | Tamanho: {decision.get('position_size', 0):.2f}")
            
            return decision
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na tomada de decisão: {e}")
            return self._create_decision(
                action='HOLD',
                confidence=0.0,
                reason=f'Erro interno: {str(e)}',
                current_price=current_price if 'current_price' in locals() else 0
            )

    def _analyze_signal_consensus(self, signals: Dict) -> Dict[str, Any]:
        """Analisa consenso de sinais usando sistema centralizado"""
        try:
            # Usa sistema centralizado se disponível
            if self.central_ensemble_system:
                formatted_signals = {}
                for key, value in signals.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        formatted_signals[key] = float(value)
                consensus_signal, confidence = self.central_ensemble_system.get_ensemble_prediction(formatted_signals)
                return {
                    'consensus_signal': consensus_signal,
                    'confidence': confidence,
                    'agreement_level': 1.0,  # Placeholder
                    'signal_strength': abs(consensus_signal)
                }
            else:
                return self._analyze_signal_consensus_fallback(signals)
        except Exception as e:
            print(f"⚠️ Erro no sistema de ensemble centralizado: {e}")
            return self._analyze_signal_consensus_fallback(signals)
    
    def _analyze_signal_consensus_fallback(self, signals: Dict) -> Dict[str, Any]:
        """Fallback para análise de consenso se sistema central não estiver disponível"""
        try:
            if not signals:
                return {
                    'consensus_signal': 0.5,
                    'confidence': 0.0,
                    'agreement_level': 0.0,
                    'signal_strength': 0.0
                }
            
            # Filtra sinais válidos
            valid_signals = []
            for key, value in signals.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    valid_signals.append(float(value))
            
            if not valid_signals:
                return {
                    'consensus_signal': 0.5,
                    'confidence': 0.0,
                    'agreement_level': 0.0,
                    'signal_strength': 0.0
                }
            
            # Calcula consenso simples
            consensus_signal = np.mean(valid_signals)
            
            # Calcula confiança baseada na consistência
            signal_std = np.std(valid_signals)
            confidence = max(0.1, 1.0 - signal_std)
            
            # Calcula nível de acordo
            agreement_level = 1.0 - (signal_std / np.mean(np.abs(valid_signals))) if np.mean(np.abs(valid_signals)) > 0 else 0.0
            
            # Calcula força do sinal
            signal_strength = abs(consensus_signal - 0.5) * 2  # Normaliza para 0-1
            
            return {
                'consensus_signal': consensus_signal,
                'confidence': confidence,
                'agreement_level': agreement_level,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            print(f"⚠️ Erro no fallback de consenso: {e}")
            return {
                'consensus_signal': 0.5,
                'confidence': 0.0,
                'agreement_level': 0.0,
                'signal_strength': 0.0
            }

    def _calculate_position_adjustment(self, current_position: Optional[Dict], signal: float, current_price: float) -> float:
        """
        Calcula ajuste baseado na posição atual
        """
        try:
            if not current_position:
                return 1.0  # Sem posição = sem ajuste
            
            position_type = current_position.get('type', 'none')
            entry_price = current_position.get('entry_price', current_price)
            size = current_position.get('size', 0.0)
            
            # Se já tem posição longa
            if position_type == 'long' and size > 0:
                pnl_ratio = (current_price - entry_price) / entry_price
                
                # Se está em lucro, fica mais conservador para vender
                if pnl_ratio > 0.05:  # 5% de lucro
                    if signal < 0:  # Sinal de venda
                        return 0.7  # Reduz intensidade de venda
                
                # Se está em prejuízo, fica mais agressivo para vender
                elif pnl_ratio < -0.03:  # 3% de prejuízo
                    if signal < 0:  # Sinal de venda
                        return 1.3  # Aumenta intensidade de venda
            
            # Se já tem posição short
            elif position_type == 'short' and size > 0:
                pnl_ratio = (entry_price - current_price) / entry_price
                
                # Se está em lucro no short, fica mais conservador para comprar
                if pnl_ratio > 0.05:
                    if signal > 0:  # Sinal de compra
                        return 0.7
                
                # Se está em prejuízo no short, fica mais agressivo para comprar
                elif pnl_ratio < -0.03:
                    if signal > 0:  # Sinal de compra
                        return 1.3
            
            return 1.0
            
        except Exception:
            return 1.0

    def _apply_risk_management(self, action: str, position_size: float, confidence: float, 
                             current_price: float, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Aplica regras de gestão de risco
        """
        try:
            # Calcula volatilidade recente
            returns = market_data['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(24)  # Volatilidade diária
            
            # Ajusta tamanho baseado na volatilidade
            volatility_factor = min(2.0, max(0.5, 1.0 / (volatility * 50)))
            adjusted_size = position_size * volatility_factor
            
            # Limita risco por trade
            max_risk_size = self.risk_per_trade / (volatility * 2)  # Stop loss de 2x volatilidade
            adjusted_size = min(adjusted_size, max_risk_size)
            
            # Valida se mercado está em condições tradáveis
            market_conditions = self._assess_market_conditions(market_data)
            
            reason = f"Ação original com gestão de risco aplicada"
            
            # Se mercado muito volátil, reduz exposição
            if market_conditions['high_volatility']:
                adjusted_size *= 0.6
                reason += " (volatilidade alta)"
            
            # Se baixa liquidez (volume), reduz exposição
            if market_conditions['low_volume']:
                adjusted_size *= 0.7
                reason += " (baixo volume)"
            
            # Se gap significativo, cuidado extra
            if market_conditions['significant_gap']:
                adjusted_size *= 0.5
                reason += " (gap significativo)"
            
            return {
                'action': action,
                'position_size': max(0.0, min(adjusted_size, self.max_position_size)),
                'reason': reason,
                'risk_assessment': {
                    'volatility': volatility,
                    'volatility_factor': volatility_factor,
                    'market_conditions': market_conditions,
                    'original_size': position_size,
                    'risk_adjusted_size': adjusted_size
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na gestão de risco: {e}")
            return {
                'action': 'HOLD',
                'position_size': 0.0,
                'reason': f'Erro na gestão de risco: {str(e)}'
            }

    def _assess_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Avalia condições do mercado para trading
        """
        try:
            conditions = {
                'high_volatility': False,
                'low_volume': False,
                'significant_gap': False,
                'trending': False
            }
            
            if len(market_data) < 10:
                return conditions
            
            recent_data = market_data.tail(20)
            
            # Volatilidade
            returns = recent_data['close'].pct_change()
            volatility = returns.std()
            conditions['high_volatility'] = volatility > returns.rolling(50).std().mean() * 1.5
            
            # Volume
            avg_volume = recent_data['volume'].mean()
            recent_volume = recent_data['volume'].tail(5).mean()
            conditions['low_volume'] = recent_volume < avg_volume * 0.7
            
            # Gap significativo
            last_close = recent_data['close'].iloc[-2]
            current_open = recent_data['open'].iloc[-1]
            gap = abs(current_open - last_close) / last_close
            conditions['significant_gap'] = gap > 0.02  # 2% gap
            
            # Tendência
            trend_periods = min(10, len(recent_data))
            trend_slope = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-trend_periods]) / trend_periods
            conditions['trending'] = abs(trend_slope) > recent_data['close'].iloc[-1] * 0.001
            
            return conditions
            
        except Exception:
            return {
                'high_volatility': False,
                'low_volume': False,
                'significant_gap': False,
                'trending': False
            }

    def _create_decision(self, action: str, confidence: float, reason: str, current_price: float,
                        position_size: float = 0.0, signal_analysis: Dict = None, 
                        risk_assessment: Dict = None) -> Dict[str, Any]:
        """
        Cria objeto de decisão padronizado
        """
        decision = {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'reason': reason,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'signal_analysis': signal_analysis or {},
            'risk_assessment': risk_assessment or {}
        }
        
        # Adiciona stop loss e take profit se for ordem de compra/venda
        if action in ['BUY', 'STRONG_BUY']:
            decision['stop_loss'] = current_price * 0.98  # 2% stop loss
            decision['take_profit'] = current_price * 1.06  # 6% take profit
        elif action in ['SELL', 'STRONG_SELL']:
            decision['stop_loss'] = current_price * 1.02  # 2% stop loss (para short)
            decision['take_profit'] = current_price * 0.94  # 6% take profit (para short)
        
        return decision

    def _record_decision(self, decision: Dict[str, Any]):
        """
        Registra decisão para análise de performance
        """
        try:
            self.last_decision = decision
            self.decision_history.append(decision)
            
            # Mantém apenas últimas 100 decisões
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
            self.performance_metrics['total_decisions'] += 1
            
            logger.debug(f"[NOTE] Decisão registrada: {decision['action']}")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao registrar decisão: {e}")

    def _empty_signal(self) -> Dict[str, Any]:
        """
        Retorna sinal vazio padrão
        """
        return {
            'strength': 0.0,
            'confidence': 0.0,
            'direction': 'neutral',
            'details': {},
            'timestamp': datetime.now()
        }

    def _empty_signal_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise de sinal vazia
        """
        return {
            'individual_signals': {},
            'consensus': {'weighted_signal': 0.0, 'overall_confidence': 0.0, 'agreeing_signals': 0},
            'market_data_timestamp': datetime.now(),
            'analysis_timestamp': datetime.now()
        }

    def _interpret_sentiment_direction(self, strength: float) -> str:
        """
        Interpreta direção do sentimento
        """
        if strength > 0.2:
            return 'bullish'
        elif strength < -0.2:
            return 'bearish'
        else:
            return 'neutral'

    def _interpret_vision_direction(self, strength: float) -> str:
        """
        Interpreta direção da análise visual
        """
        if strength > 0.2:
            return 'bullish'
        elif strength < -0.2:
            return 'bearish'
        else:
            return 'neutral'

    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo das decisões recentes
        """
        try:
            if not self.decision_history:
                return {'no_decisions': True}
            
            recent_decisions = self.decision_history[-10:]  # Últimas 10
            
            actions = [d['action'] for d in recent_decisions]
            confidences = [d['confidence'] for d in recent_decisions]
            
            summary = {
                'total_decisions': len(self.decision_history),
                'recent_actions': actions,
                'avg_confidence': np.mean(confidences),
                'last_decision': self.last_decision,
                'performance_metrics': self.performance_metrics,
                'action_distribution': {
                    action: actions.count(action) for action in set(actions)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no resumo de decisões: {e}")
            return {'error': str(e)}

    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """
        Atualiza métricas de performance baseado em resultado de trade
        """
        try:
            if trade_result.get('profitable', False):
                self.performance_metrics['profitable_decisions'] += 1
            
            # Atualiza accuracy
            total = self.performance_metrics['total_decisions']
            profitable = self.performance_metrics['profitable_decisions']
            
            if total > 0:
                self.performance_metrics['accuracy'] = profitable / total
            
            logger.info(f"[UP] Performance atualizada: {self.performance_metrics['accuracy']:.3f} accuracy")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao atualizar métricas: {e}")

    def get_signal_strength(self, market_data: pd.DataFrame) -> float:
        """
        Retorna força do sinal de decisão geral
        
        Args:
            market_data (pd.DataFrame): Dados de mercado
            
        Returns:
            float: Força do sinal (-1 a 1)
        """
        try:
            signal_analysis = self.analyze_market_signals(market_data)
            consensus = signal_analysis['consensus']
            
            # Combina força do sinal com confiança
            signal_strength = consensus['consensus_signal'] * consensus['confidence']
            
            logger.info(f"[SIGNAL] Força do sinal de decisão: {signal_strength:.3f}")
            
            return signal_strength
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao calcular força do sinal: {e}")
            return 0.0 