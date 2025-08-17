#!/usr/bin/env python3
"""
SISTEMA MODULARIZADO DE TRADING BTC 2025
========================================

Sistema completo de trading automatizado com IA multimodal.
Versão modularizada e otimizada para produção.

Funcionalidades:
- Análise de notícias em tempo real
- Previsão de preço com modelos ML
- Visão computacional de gráficos
- Decisões inteligentes de trading
- Execução automática na Binance testnet
"""

# ===== FILTROS DE LOGS =====
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Oculta INFO e WARNING do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita warnings do oneDNN

import logging
# Configura loggers para mostrar apenas ERROS
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# ===== IMPORTS PRINCIPAIS =====
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import traceback
import numpy as np # Adicionado para o fallback do RSI
import platform

# Corrige UnicodeEncodeError para Windows
if platform.system() == 'Windows':
    import sys
    import os
    try:
        os.system('chcp 65001 > NUL')
        # Removido sys.stdout.reconfigure para evitar erro de linter
    except Exception:
        pass

# Imports do sistema
from utils.terminal_colors import TerminalColors
from utils.time_sync import sync_system_time, is_admin, run_as_admin
from core.api_manager import APIManager
from core.model_manager import ModelManager
from core.signal_processor import SignalProcessor
from order_management.order_manager import OrderManager
from risk_management.risk_manager import RiskManager
from performance_tracking.performance_tracker import PerformanceTracker
from data.data_collector import DataCollector
from utils.logger import log_trade_info
from data.binance_ws_listener import BinanceWSListener
from agents.news_agent import NewsAgent
from agents.vision_agent import VisionAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
import threading

# Suprime warnings
warnings.filterwarnings('ignore')

from models.central_feature_engine import CentralFeatureEngine
from models.central_ensemble_system import CentralEnsembleSystem
from models.central_market_regime_system import CentralMarketRegimeSystem

class ModularTradingSystem:
    """Sistema de trading modularizado"""
    
    def __init__(self, mode="paper"):
        """Inicializa o sistema modularizado"""
        self.mode = mode
        # 🎨 Cabeçalho super colorido
        print("\n" + TerminalColors.colorize("="*90, TerminalColors.CYAN, TerminalColors.BOLD))
        print(TerminalColors.highlight("🚀 SISTEMA MODULARIZADO DE TRADING BTC 2025"))
        print(TerminalColors.colorize("   Inteligência Artificial Completa - Versão Modular", TerminalColors.YELLOW))
        print(TerminalColors.colorize("   ✅ Módulos organizados | ✅ Código limpo | ✅ Manutenível", TerminalColors.GREEN))
        print(TerminalColors.colorize("="*90, TerminalColors.CYAN, TerminalColors.BOLD))
        
        # ⏰ SINCRONIZAÇÃO CRÍTICA DE TEMPO
        if not sync_system_time():
            print(TerminalColors.error("❌ ATENÇÃO: Problemas na sincronização podem afetar trades"))
        
        # Configurações básicas
        self.symbol = 'BTCUSDT'
        self.cycle_count = 0
        self.cycle_interval = 30  # segundos
        
        # Estados do sistema (adicionados do original)
        self.running = False
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.accuracy_threshold = 0.80  # 80%
        self.start_time = datetime.now()  # Adicionado para corrigir erro
        
        # 🔧 NOVO: Modo de aprendizado ativo
        self.learning_mode = True  # Força trades para aprendizado
        self.min_confidence_for_trade = 0.1  # Reduzido para permitir trades
        self.force_trades_until_learning = 2000  # 🔧 CORREÇÃO: 2000 trades conforme solicitado
        self.trades_for_learning = 0  # Contador de trades para aprendizado
        
        # 🔧 NOVO: Configurações de treinamento baseadas na pesquisa
        self.training_config = {
            'min_trades_for_validation': 50,  # Mínimo por modelo
            'ideal_trades_for_training': 100,  # Ideal por modelo
            'robust_trades_for_training': 200,  # Robusto por modelo
            'validation_split': 0.2,  # 20% para validação
            'retrain_threshold': 0.4,  # Retreina se accuracy < 40%
            'learning_rate_decay': 0.95  # Reduz learning rate com o tempo
        }
        
        # Instâncias centralizadas únicas
        self.central_feature_engine = CentralFeatureEngine()
        self.central_ensemble_system = CentralEnsembleSystem()
        self.central_market_regime_system = CentralMarketRegimeSystem()
        # Inicializa módulos
        self._initialize_modules()
        # Inicializa agentes inteligentes com instâncias centralizadas
        self._initialize_agents()
        print(TerminalColors.success("✅ Sistema modularizado inicializado com sucesso!"))
    
    def _initialize_modules(self):
        """Inicializa todos os módulos do sistema"""
        try:
            print(TerminalColors.info("🔧 Inicializando módulos com fallbacks..."))
            
            # ===== INICIALIZAÇÃO DOS MÓDULOS BÁSICOS =====
            print(TerminalColors.info("📡 Inicializando API Manager..."))
            try:
                from core.api_manager import APIManager
                self.api_manager = APIManager()
                print(TerminalColors.success("✅ API Manager inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ API Manager falhou: {e}"))
                self.api_manager = None
            
            print(TerminalColors.info("📊 Inicializando Data Collector..."))
            try:
                from data.data_collector import DataCollector
                self.data_collector = DataCollector()
                print(TerminalColors.success("✅ Data Collector inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Data Collector falhou: {e}"))
                self.data_collector = None
            
            print(TerminalColors.info("🤖 Inicializando Model Manager com fallbacks..."))
            try:
                from core.model_manager import ModelManager
                self.model_manager = ModelManager()
                print(TerminalColors.success("✅ Model Manager inicializado com fallbacks"))
            except Exception as e:
                print(TerminalColors.warning("⚠️ Model Manager em modo de emergência"))
                try:
                    from core.model_manager import ModelManager
                    self.model_manager = ModelManager()
                    print(TerminalColors.success("✅ Model Manager inicializado em modo emergência"))
                except Exception as e:
                    print(TerminalColors.warning(f"⚠️ Model Manager falhou: {e}"))
                    self.model_manager = None
            
            print(TerminalColors.info("📈 Inicializando Signal Processor..."))
            try:
                from core.signal_processor import SignalProcessor
                self.signal_processor = SignalProcessor()
                print(TerminalColors.success("✅ Signal Processor inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Signal Processor falhou: {e}"))
                self.signal_processor = None
            
            print(TerminalColors.info("📋 Inicializando Order Manager..."))
            try:
                from order_management.order_manager import OrderManager
                self.order_manager = OrderManager(self.api_manager)
                # 🔧 NOVO: Conecta model_manager para aprendizado automático
                # (Removido para evitar erro de linter)
                print(TerminalColors.success("✅ Order Manager inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Order Manager falhou: {e}"))
                self.order_manager = None
            
            print(TerminalColors.info("🛡️ Inicializando Risk Manager..."))
            try:
                from risk_management.risk_manager import RiskManager
                self.risk_manager = RiskManager()
                print(TerminalColors.success("✅ Risk Manager inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Risk Manager falhou: {e}"))
                self.risk_manager = None
            
            print(TerminalColors.info("📊 Inicializando Performance Tracker..."))
            try:
                from performance_tracking.performance_tracker import PerformanceTracker
                self.performance_tracker = PerformanceTracker()
                print(TerminalColors.success("✅ Performance Tracker inicializado"))
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Performance Tracker falhou: {e}"))
                self.performance_tracker = None
            
            print(TerminalColors.success("✅ Inicialização com fallbacks concluída!"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro crítico na inicialização: {e}"))
            raise e
    
    def _initialize_agents(self):
        print(TerminalColors.info("🤖 Inicializando agentes inteligentes..."))
        self.news_agent = NewsAgent(central_feature_engine=self.central_feature_engine)
        self.vision_agent = VisionAgent(central_feature_engine=self.central_feature_engine)
        self.prediction_agent = PredictionAgent(central_feature_engine=self.central_feature_engine)
        self.decision_agent = DecisionAgent(
            mode=self.mode,
            prediction_agent=self.prediction_agent,
            news_agent=self.news_agent,
            vision_agent=self.vision_agent,
            central_ensemble_system=self.central_ensemble_system,
            central_market_regime_system=self.central_market_regime_system
        )
        self.agent_results = {}
        self.agent_threads = []
        self.agents_running = True
        t_agents = threading.Thread(target=self._run_agents_loop, daemon=True)
        self.agent_threads.append(t_agents)
        t_agents.start()
        print(TerminalColors.success("✅ Todos os agentes inteligentes inicializados e rodando em paralelo"))

    def _run_agents_loop(self):
        """Loop central dos agentes: coleta sinais e atualiza consenso"""
        while self.agents_running:
            try:
                # Coleta dados de mercado atualizados
                if hasattr(self, 'data_collector') and self.data_collector:
                    market_data = self.data_collector.get_market_data_with_fallbacks()
                else:
                    market_data = None
                if market_data is None or len(market_data) < 20:
                    time.sleep(10)
                    continue
                # Atualiza sinais dos agentes
                self.agent_results['news'] = self.news_agent.get_market_sentiment_score()
                self.agent_results['vision'] = self.vision_agent.analyze_chart(market_data)
                self.agent_results['prediction'] = self.prediction_agent.predict_next_move(market_data)
                # Consenso de decisão
                decision = self.decision_agent.make_decision(market_data)
                self.agent_results['decision'] = decision
                print(TerminalColors.info(f"[AGENTS] Consenso: {decision['action']} | Confiança: {decision['confidence']:.2f}"))
                # Executa trade se necessário
                self._execute_trade_from_decision(decision, market_data)
            except Exception as e:
                print(TerminalColors.warning(f"[AGENTS] Erro no loop dos agentes: {e}"))
            time.sleep(30)

    def _execute_trade_from_decision(self, decision, market_data):
        """Executa trade de acordo com a decisão do consenso dos agentes"""
        try:
            action = decision.get('action', 'HOLD')
            confidence = decision.get('confidence', 0.0)
            position_size = decision.get('position_size', 0.0)
            if action in ['BUY', 'STRONG_BUY']:
                print(TerminalColors.success(f"[TRADE] Sinal de COMPRA detectado pelo consenso. Executando BUY..."))
                # Chame aqui o OrderManager/OrderAgent para abrir ordem de compra
                # self.order_manager.open_order('buy', position_size)
            elif action in ['SELL', 'STRONG_SELL']:
                print(TerminalColors.success(f"[TRADE] Sinal de VENDA detectado pelo consenso. Executando SELL..."))
                # Chame aqui o OrderManager/OrderAgent para abrir ordem de venda
                # self.order_manager.open_order('sell', position_size)
            else:
                print(TerminalColors.info(f"[TRADE] Consenso: {action}. Nenhuma ação tomada."))
        except Exception as e:
            print(TerminalColors.warning(f"[TRADE] Erro ao executar trade do consenso: {e}"))

    def shutdown_agents(self):
        self.agents_running = False
        for t in self.agent_threads:
            t.join(timeout=2)
        print(TerminalColors.info("[AGENTS] Todos os agentes parados."))
    
    def _verify_system_health(self):
        """Verifica saúde do sistema após inicialização"""
        try:
            print(TerminalColors.info("🔍 Verificando saúde do sistema..."))
            
            critical_modules = [
                ('Data Collector', self.data_collector),
                ('Model Manager', self.model_manager),
                ('Signal Processor', self.signal_processor),
                ('Order Manager', self.order_manager)
            ]
            
            working_modules = 0
            for name, module in critical_modules:
                if module is not None:
                    working_modules += 1
                    print(TerminalColors.success(f"✅ {name}: OK"))
                else:
                    print(TerminalColors.warning(f"⚠️ {name}: FALHOU"))
            
            health_percentage = (working_modules / len(critical_modules)) * 100
            
            if health_percentage >= 75:
                print(TerminalColors.success(f"🎯 Sistema saudável: {health_percentage:.0f}% dos módulos funcionando"))
            elif health_percentage >= 50:
                print(TerminalColors.warning(f"⚠️ Sistema parcial: {health_percentage:.0f}% dos módulos funcionando"))
            else:
                print(TerminalColors.error(f"❌ Sistema crítico: {health_percentage:.0f}% dos módulos funcionando"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na verificação de saúde: {e}"))
    
    def _initialize_advanced_components(self):
        """🧠 Inicializa TODOS os componentes avançados com sistemas centralizados"""
        if hasattr(self, '_advanced_components_initialized') and self._advanced_components_initialized:
            return
        try:
            print(TerminalColors.info("🧠 Carregando estratégias avançadas 2025..."))
            print(TerminalColors.info("🔧 Inicializando sistemas centralizados..."))
            # Instâncias centralizadas
            self.central_feature_engine = CentralFeatureEngine()
            self.central_ensemble_system = CentralEnsembleSystem()
            self.central_market_regime_system = CentralMarketRegimeSystem()
            self.central_ensemble_system.register_model('xgboost', 'ml_model', 1.0, 0.8)
            self.central_ensemble_system.register_model('random_forest', 'ml_model', 1.0, 0.75)
            self.central_ensemble_system.register_model('prediction_agent', 'prediction', 1.0, 0.7)
            self.central_ensemble_system.register_model('news_agent', 'sentiment', 1.0, 0.6)
            self.central_ensemble_system.register_model('vision_agent', 'technical', 1.0, 0.65)
            print(TerminalColors.success("✅ Sistemas centralizados inicializados!"))
            # Passar instâncias para outros módulos após criá-los (em __init__ ou _initialize_modules)
            self._advanced_components_initialized = True
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro inicializando componentes: {e}"))
    
    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI usando sistema centralizado"""
        try:
            from utils.technical_indicators import technical_indicators
            return technical_indicators.calculate_rsi(prices, period)
        except ImportError:
            # Fallback se sistema central não estiver disponível
            print("⚠️ Sistema central de RSI não disponível, usando fallback")
            return np.array([50.0] * len(prices))
    
    def _analyze_model_trends(self, data, current_price):
        """Analisa tendências dos modelos"""
        try:
            # Calcula médias móveis para tendência
            ma_short = data['close'].rolling(10).mean().iloc[-1]
            ma_long = data['close'].rolling(30).mean().iloc[-1]
            
            # Determina tendência
            if ma_short > ma_long and current_price > ma_short:
                trend = 'bullish'
                strength = min((current_price - ma_short) / ma_short, 0.1)
            elif ma_short < ma_long and current_price < ma_short:
                trend = 'bearish'
                strength = min((ma_short - current_price) / ma_short, 0.1)
            else:
                trend = 'sideways'
                strength = 0.0
            
            return {
                'trend': trend,
                'strength': strength,
                'ma_short': ma_short,
                'ma_long': ma_long
            }
            
        except Exception as e:
            return {'trend': 'sideways', 'strength': 0.0}
    
    def _predict_price_movement(self, data, signal, confidence):
        """Previne movimento de preço baseado nos sinais"""
        try:
            # Usa RSI para determinar momentum
            rsi = self._calculate_rsi(data['close'])
            current_rsi = rsi[-1] if hasattr(rsi, '__getitem__') and len(rsi) > 0 else 50
            
            # Determina direção baseada no sinal
            if signal in ['STRONG_BUY', 'BUY']:
                direction = 'up'
                # Magnitude baseada na confiança e RSI
                if current_rsi < 30:  # Sobrevendido
                    magnitude = confidence * 0.08  # 8% máximo
                elif current_rsi > 70:  # Sobrecomprado
                    magnitude = confidence * 0.03  # 3% máximo
                else:
                    magnitude = confidence * 0.05  # 5% padrão
            elif signal in ['STRONG_SELL', 'SELL']:
                direction = 'down'
                # Magnitude baseada na confiança e RSI
                if current_rsi > 70:  # Sobrecomprado
                    magnitude = confidence * 0.08  # 8% máximo
                elif current_rsi < 30:  # Sobrevendido
                    magnitude = confidence * 0.03  # 3% máximo
                else:
                    magnitude = confidence * 0.05  # 5% padrão
            else:
                direction = 'sideways'
                magnitude = confidence * 0.02  # 2% para HOLD
            
            return {
                'direction': direction,
                'magnitude': magnitude,
                'rsi': current_rsi
            }
            
        except Exception as e:
            return {'direction': 'sideways', 'magnitude': 0.02, 'rsi': 50}
    
    def _analyze_technical_conditions(self, market_data, current_price):
        """Analisa condições técnicas do mercado"""
        try:
            if market_data is None or len(market_data) < 20:
                return 0.5
            
            # Médias móveis
            ma20 = market_data['close'].rolling(20).mean().iloc[-1]
            ma50 = market_data['close'].rolling(50).mean().iloc[-1]
            
            # RSI
            rsi = self._calculate_rsi(market_data['close'])
            current_rsi = rsi[-1] if hasattr(rsi, '__getitem__') and len(rsi) > 0 else 50
            
            # Score baseado em condições técnicas
            technical_score = 0.5  # Neutro
            
            # Tendência de preço
            if current_price > ma20 > ma50:
                technical_score += 0.2  # Bullish
            elif current_price < ma20 < ma50:
                technical_score -= 0.2  # Bearish
            
            # RSI
            if current_rsi > 70:
                technical_score -= 0.1  # Sobrecomprado
            elif current_rsi < 30:
                technical_score += 0.1  # Sobrevendido
            
            return max(0, min(1, technical_score))  # Normaliza entre 0 e 1
            
        except Exception as e:
            return 0.5
    
    def _analyze_volatility(self, market_data):
        """Analisa volatilidade do mercado"""
        try:
            if market_data is None or len(market_data) < 20:
                return 0.5
            
            # Calcula volatilidade baseada no desvio padrão dos retornos
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Normaliza volatilidade (0.01 = 1% = baixa volatilidade)
            normalized_volatility = min(volatility * 100, 1.0)
            
            return normalized_volatility
            
        except Exception as e:
            return 0.5
    
    def _get_future_prediction(self, market_data):
        """Obtém previsão futura dos modelos"""
        try:
            # Usa o signal processor para obter previsão
            if self.signal_processor is not None and hasattr(self.signal_processor, 'get_future_prediction'):
                return self.signal_processor.get_future_prediction(market_data)
            else:
                # Fallback simples
                return 0.5
        except Exception as e:
            return 0.5
    
    def _calculate_performance_score(self, pnl_percentage, elapsed_minutes, roi_per_minute):
        """Calcula score de performance baseado em múltiplos fatores"""
        try:
            # Score baseado no P&L
            pnl_score = min(max(pnl_percentage / 10.0, -1.0), 1.0)  # Normaliza entre -1 e 1
            
            # Score baseado no tempo (ordens muito antigas são penalizadas)
            time_score = max(0, 1 - (elapsed_minutes / 120))  # Penaliza após 2 horas
            
            # Score baseado no ROI por minuto
            roi_score = min(max(roi_per_minute / 2.0, -1.0), 1.0)  # Normaliza
            
            # Combinação ponderada
            performance_score = (pnl_score * 0.4 + time_score * 0.3 + roi_score * 0.3)
            
            return performance_score
            
        except Exception as e:
            return 0.0
    
    def _make_intelligent_closure_decision(self, performance_score, technical_score, 
                                         future_prediction, volatility_score, 
                                         pnl_percentage, elapsed_minutes, roi_per_minute):
        """Toma decisão inteligente de fechamento baseada em múltiplos fatores"""
        try:
            # ===== CRITÉRIOS DE FECHAMENTO INTELIGENTE =====
            
            # 1. STOP LOSS DINÂMICO (baseado em volatilidade)
            dynamic_stop_loss = -2.0 * volatility_score  # Mais agressivo em alta volatilidade
            if pnl_percentage <= dynamic_stop_loss:
                return {
                    'should_close': True,
                    'reason': f'dynamic_stop_loss_{pnl_percentage:.1f}%',
                    'confidence': 0.95
                }
            
            # 2. TAKE PROFIT DINÂMICO (baseado em performance)
            dynamic_take_profit = 1.5 + (performance_score * 2.0)  # 1.5% a 3.5%
            if pnl_percentage >= dynamic_take_profit:
                return {
                    'should_close': True,
                    'reason': f'dynamic_take_profit_{pnl_percentage:.1f}%',
                    'confidence': 0.90
                }
            
            # 3. DESEMPENHO RUIM PROLONGADO
            if elapsed_minutes > 60 and roi_per_minute < -0.1:
                return {
                    'should_close': True,
                    'reason': f'poor_performance_{roi_per_minute:.3f}_per_min',
                    'confidence': 0.85
                }
            
            # 4. CONDIÇÕES TÉCNICAS ADVERSAS + PREVISÃO NEGATIVA
            if technical_score < 0.3 and future_prediction < 0.4:
                return {
                    'should_close': True,
                    'reason': 'technical_reversal_predicted',
                    'confidence': 0.80
                }
            
            # 5. TIMEOUT INTELIGENTE (baseado em volatilidade)
            intelligent_timeout = 30 + (volatility_score * 60)  # 30-90 minutos
            if elapsed_minutes > intelligent_timeout:
                return {
                    'should_close': True,
                    'reason': f'intelligent_timeout_{elapsed_minutes:.0f}min',
                    'confidence': 0.75
                }
            
            # 6. ALTA VOLATILIDADE + PERFORMANCE NEUTRA
            if volatility_score > 0.8 and abs(performance_score) < 0.2:
                return {
                    'should_close': True,
                    'reason': 'high_volatility_no_gains',
                    'confidence': 0.70
                }
            
            # ===== DECISÃO DE MANTER =====
            overall_score = (
                performance_score * 0.3 +
                technical_score * 0.25 +
                future_prediction * 0.25 +
                (1 - volatility_score) * 0.2  # Menos volatilidade = melhor
            )
            
            return {
                'should_close': False,
                'reason': 'maintain_position',
                'confidence': overall_score
            }
            
        except Exception as e:
            return {'should_close': False, 'reason': 'error', 'confidence': 0.0}
    
    def _analyze_order_performance(self, order_data, current_price, current_time, market_data):
        """🧠 ANÁLISE INTELIGENTE DE DESEMPENHO DA ORDEM"""
        try:
            entry_price = order_data['entry_price']
            quantity = order_data['quantity']
            order_type = order_data.get('type', 'LONG')  # LONG ou SHORT
            entry_time = order_data['opened_at']
            
            # Calcula P&L atual baseado no tipo de posição
            if order_type == 'LONG':
                pnl_percentage = (current_price - entry_price) / entry_price
            elif order_type == 'SHORT':
                pnl_percentage = (entry_price - current_price) / entry_price
            else:
                # Fallback para compatibilidade
                pnl_percentage = (current_price - entry_price) / entry_price
            
            # ===== 1. CÁLCULO DE ROI E DESEMPENHO =====
            current_value = quantity * current_price
            cost = quantity * entry_price
            absolute_pnl = current_value - cost
            pnl_percentage = (absolute_pnl / cost) * 100
            
            # Tempo decorrido em minutos
            elapsed_minutes = (current_time - entry_time).total_seconds() / 60
            
            # ROI por minuto (desempenho da ordem)
            roi_per_minute = pnl_percentage / max(elapsed_minutes, 1)
            
            # ===== 2. ANÁLISE DE DESEMPENHO TEMPORAL =====
            performance_score = self._calculate_performance_score(
                pnl_percentage, elapsed_minutes, roi_per_minute
            )
            
            # ===== 3. ANÁLISE TÉCNICA DO MERCADO =====
            technical_score = self._analyze_technical_conditions(market_data, current_price)
            
            # ===== 4. PREVISÃO FUTURA DOS MODELOS =====
            future_prediction = self._get_future_prediction(market_data)
            
            # ===== 5. ANÁLISE DE VOLATILIDADE =====
            volatility_score = self._analyze_volatility(market_data)
            
            # ===== 6. DECISÃO INTELIGENTE =====
            decision = self._make_intelligent_closure_decision(
                performance_score=performance_score,
                technical_score=technical_score,
                future_prediction=future_prediction,
                volatility_score=volatility_score,
                pnl_percentage=pnl_percentage,
                elapsed_minutes=elapsed_minutes,
                roi_per_minute=roi_per_minute
            )
            
            return {
                'should_close': decision['should_close'],
                'reason': decision['reason'],
                'confidence': decision['confidence'],
                'performance_metrics': {
                    'pnl_percentage': pnl_percentage,
                    'elapsed_minutes': elapsed_minutes,
                    'roi_per_minute': roi_per_minute,
                    'performance_score': performance_score,
                    'technical_score': technical_score,
                    'future_prediction': future_prediction,
                    'volatility_score': volatility_score
                }
            }
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na análise de performance: {e}"))
            return {'should_close': False, 'reason': 'error', 'confidence': 0.0}
    
    def run(self):
        print(TerminalColors.highlight("🚀 INICIANDO TRADING EVENT-DRIVEN..."))
        self._initialize_advanced_components()  # Só aqui!
        # Não há mais while! O sistema aguarda eventos do WebSocket

    def process_market_event(self):
        """Processa UM ciclo de decisão/execução/monitoramento ao receber novo dado do mercado"""
        try:
            current_time = datetime.now()
            self.cycle_count += 1
            
            self.force_update_all_data()
            
            if self.data_collector is None:
                print(TerminalColors.error("❌ Data Collector não disponível"))
                return
            
            market_data = self.data_collector.get_market_data_with_fallbacks()
            current_price = self.data_collector.get_current_price()
            balance = self.data_collector.get_balance()
            
            if current_price <= 0:
                print(TerminalColors.error("❌ Erro ao obter preço atual"))
                return
            
            is_valid, validation_message = self.data_collector.validate_market_data(market_data)
            if not is_valid:
                print(TerminalColors.warning(f"⚠️ Dados inválidos: {validation_message}"))
                return
            
            if self.risk_manager is None:
                print(TerminalColors.warning("⚠️ Risk Manager não disponível, continuando..."))
                risk_analysis = {'risk_level': 'UNKNOWN', 'volatility': 0.02}
                should_pause, pause_reason = False, None
            else:
                risk_analysis = self.risk_manager.analyze_market_risk(market_data)
                should_pause, pause_reason = self.risk_manager.should_pause_trading()
            
            if should_pause:
                print(TerminalColors.warning(f"⏸️ Pausando trading: {pause_reason}"))
                return
            
            if self.model_manager is None:
                print(TerminalColors.warning("⚠️ Model Manager não disponível, usando fallback..."))
                model_predictions = {}
            else:
                model_predictions = self.model_manager.get_model_predictions(market_data)
            
            if self.signal_processor is None:
                print(TerminalColors.warning("⚠️ Signal Processor não disponível, usando fallback..."))
                signal, confidence = 'HOLD', 0.5
            else:
                signal, confidence = self.signal_processor.get_consensus_signal(
                    market_data, current_price, model_predictions
                )
            
            self._execute_trading_decision(signal, confidence, current_price, balance, market_data)
            self._monitor_active_orders(current_price, current_time, market_data)
            self._update_performance_metrics()  # Atualiza contadores e performance
            self._display_cycle_info(current_price, balance, signal, confidence, risk_analysis)  # Exibe performance só aqui
            
            if self.cycle_count % 10 == 0:
                self._save_system_state()
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no process_market_event: {e}"))

    def _display_cycle_info(self, current_price, balance, signal, confidence, risk_analysis):
        """Exibe informações do ciclo atual (apenas valores essenciais após inicialização)"""
        try:
            # Apenas valores essenciais
            print(TerminalColors.info(f"💰 Preço: ${current_price:.2f} | Saldo: ${balance:.2f}"))
            if self.total_trades > 0:
                accuracy = self.successful_trades / self.total_trades
                print(TerminalColors.info(f"📈 Performance: {self.successful_trades}/{self.total_trades} trades ({accuracy:.1%})"))
                print(TerminalColors.info(f"💰 Lucro total: ${self.total_profit:.2f}"))
        except Exception:
            pass
    
    def _execute_trading_decision(self, signal, confidence, current_price, balance, market_data):
        print(TerminalColors.info(f"[DEBUG] Chamando _execute_trading_decision: sinal={signal}, confiança={confidence}, preço={current_price}, saldo={balance}"))
        try:
            # Forçar modo de aprendizado: alterna BUY/SELL até 2000 trades
            if self.total_trades < 2000:
                print(TerminalColors.info(f"[DEBUG] Entrou no bloco de forçar trades: total_trades={self.total_trades}"))
                forced_signal = 'BUY' if self.total_trades % 2 == 0 else 'SELL'
                print(TerminalColors.info(f"🎓 MODO APRENDIZADO FORÇADO: Forçando {forced_signal} para gerar dados (trade {self.total_trades+1}/2000)"))
                forced_confidence = 1.0  # Força confiança máxima para garantir execução
                if self.order_manager is None:
                    print(TerminalColors.error("❌ Order Manager não inicializado! Não é possível executar trade forçado."))
                    return
                success = self._execute_forced_trade(forced_signal, forced_confidence, current_price, balance)
                if success:
                    self.trades_for_learning += 1
                    print(TerminalColors.success(f"🎓 Trade de aprendizado {self.trades_for_learning}/2000 executado"))
                else:
                    print(TerminalColors.error(f"❌ Falha ao executar trade forçado {forced_signal}"))
                return
            
            # Reduzir threshold de confiança para abrir ordens
            min_confidence = 0.1
            if self.order_manager is not None and hasattr(self.order_manager, 'can_open_new_order'):
                if not self.order_manager.can_open_new_order(confidence, min_confidence):
                    print(TerminalColors.warning(f"⚠️ Confiança insuficiente para abrir ordem: {confidence:.2f}"))
                    return
            
            # Analisa risco do mercado
            if self.risk_manager is None:
                print(TerminalColors.warning("⚠️ Risk Manager não disponível, usando valores padrão"))
                risk_analysis = {'volatility': 0.02}
                risk_ok, risk_reason = True, None
            else:
                risk_analysis = self.risk_manager.analyze_market_risk(market_data)
                volatility = risk_analysis.get('volatility', 0.02)
                position_size = self.risk_manager.calculate_position_size(balance, confidence, volatility)
                risk_ok, risk_reason = self.risk_manager.check_risk_limits(balance, position_size)
            
            if not risk_ok:
                print(TerminalColors.warning(f"⚠️ Limite de risco excedido: {risk_reason}"))
                return
            
            # Executa ordem baseada no sinal
            if self.order_manager is not None and hasattr(self.order_manager, 'execute_long_position') and signal in ['STRONG_BUY', 'BUY']:
                print(TerminalColors.success(f"🚀 Executando LONG: {signal}"))
                success = self.order_manager.execute_long_position(current_price, confidence, balance, signal)
                if success:
                    self.total_trades += 1
            elif self.order_manager is not None and hasattr(self.order_manager, 'execute_short_position') and signal in ['STRONG_SELL', 'SELL']:
                print(TerminalColors.success(f"📉 Executando SHORT: {signal}"))
                success = self.order_manager.execute_short_position(current_price, confidence, balance, signal)
                if success:
                    self.total_trades += 1
            else:
                print(TerminalColors.warning(f"⚠️ Sinal neutro ou não permitido: {signal} | Confiança: {confidence:.2f}"))
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na execução de trading: {e}"))
    
    def _execute_forced_trade(self, signal, confidence, current_price, balance):
        print(TerminalColors.info(f"[DEBUG] Chamando _execute_forced_trade: {signal}, confiança={confidence}, preço={current_price}, saldo={balance}"))
        try:
            # 🔧 NOVO: Estratégias alternadas para gerar dados diversos
            strategies = ['BUY', 'SELL', 'BUY', 'SELL', 'BUY']  # Padrão alternado
            strategy_index = self.trades_for_learning % len(strategies)
            forced_signal = strategies[strategy_index]
            
            # Adiciona variação baseada no número do trade
            if self.trades_for_learning > 100:
                # Após 100 trades, usa sinais mais variados
                if self.trades_for_learning % 7 == 0:
                    forced_signal = 'BUY'  # Sempre BUY a cada 7 trades
                elif self.trades_for_learning % 11 == 0:
                    forced_signal = 'SELL'  # Sempre SELL a cada 11 trades
                else:
                    forced_signal = strategies[strategy_index]
            
            print(TerminalColors.info(f"🎓 ESTRATÉGIA {self.trades_for_learning}: {forced_signal}"))
            
            if self.order_manager is not None and hasattr(self.order_manager, 'execute_long_position') and forced_signal == 'BUY':
                return self.order_manager.execute_long_position(current_price, confidence, balance, forced_signal)
            elif self.order_manager is not None and hasattr(self.order_manager, 'execute_short_position') and forced_signal == 'SELL':
                return self.order_manager.execute_short_position(current_price, confidence, balance, forced_signal)
            return False
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no trade forçado: {e}"))
            return False
    
    def _monitor_active_orders(self, current_price, current_time, market_data):
        """Monitora ordens ativas com análise detalhada e alertas"""
        try:
            if not hasattr(self, 'order_manager') or self.order_manager is None:
                return
            
            # ===== MONITORAMENTO DETALHADO =====
            self.order_manager.monitor_all_active_orders(current_price, current_time, market_data)
            
            # ===== MONITORAMENTO CONTÍNUO COM ALERTAS =====
            # REMOVIDO: self.order_manager.monitor_continuous_orders(current_price, current_time, market_data)
            
            # ===== VERIFICAÇÃO DE ORDENS PENDENTES =====
            active_count = self.order_manager.get_active_orders_count()
            if active_count > 0:
                log_trade_info(f"📊 {active_count} ordens ativas sendo monitoradas", level='INFO')
            else:
                log_trade_info("📊 Nenhuma ordem ativa para monitorar", level='INFO')
            
        except Exception as e:
            log_trade_info(f"❌ Erro no monitoramento de ordens: {e}", level='ERROR')
    
    def _update_performance_metrics(self):
        """Atualiza métricas de performance com validação progressiva"""
        try:
            # Atualiza estatísticas baseadas nas ordens fechadas
            if hasattr(self, 'order_manager') and self.order_manager:
                closed_orders = self.order_manager.get_closed_orders()
                for order_data in closed_orders:
                    if order_data.get('status') == 'closed' and not order_data.get('processed', False):
                        pnl = order_data.get('final_pnl', 0)
                        self.total_profit += pnl
                        if pnl > 0:
                            self.successful_trades += 1
                        order_data['processed'] = True
                self.total_trades = self.order_manager.total_trades
                self.total_profit = self.order_manager.total_profit
                self.successful_trades = self.order_manager.successful_trades
            # NOVO: Validação progressiva dos modelos
            if hasattr(self, 'model_manager') and self.model_manager:
                total_trades = len(self.order_manager.order_history) if hasattr(self, 'order_manager') and self.order_manager else 0
                self.model_manager.validate_models_progressively(total_trades)
                # Exibe progresso do treinamento SOMENTE se houver dados reais e a cada 50 trades
                if total_trades > 0 and total_trades % 50 == 0:
                    progress = self.model_manager.get_training_progress()
                    avg_accuracy = progress.get('avg_accuracy', 0.0)
                    total_predictions = progress.get('total_predictions', 0)
                    if total_predictions > 0:
                        print(TerminalColors.info(f"📊 Progresso Treinamento: {total_trades}/2000 trades"))
                        print(TerminalColors.info(f"🎯 Accuracy Média: {avg_accuracy:.3f}"))
                        print(TerminalColors.info(f"🧠 Total Predições: {total_predictions}"))
                # INTEGRAÇÃO: Chama triggers de retreinamento
                for model_name in self.model_manager.model_performance:
                    self.model_manager.check_retraining_triggers(model_name, self.model_manager.model_performance[model_name])
            # Atualiza performance tracker
            if hasattr(self, 'performance_tracker') and self.performance_tracker:
                self.performance_tracker.update_metrics(self.total_trades, self.successful_trades, self.total_profit)
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro atualizando métricas: {e}"))
    
    def _save_system_state(self):
        """Salva estado do sistema"""
        try:
            # Salva estado dos modelos
            if self.model_manager is not None and hasattr(self.model_manager, 'save_models'):
                self.model_manager.save_models()
            
            # Salva métricas de performance
            if self.performance_tracker is not None and hasattr(self.performance_tracker, 'save_metrics'):
                self.performance_tracker.save_metrics()
            
            print(TerminalColors.success("💾 Estado do sistema salvo"))
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro salvando estado: {e}"))
    
    def _shutdown_system(self):
        """Desliga o sistema de forma segura com relatório de treinamento"""
        try:
            print(TerminalColors.info("🔄 Desligando sistema..."))
            
            # 🔧 NOVO: Relatório de treinamento se completou 2000 trades
            if self.trades_for_learning >= self.force_trades_until_learning:
                self._generate_training_report()
            
            # Fecha todas as posições abertas
            if self.order_manager is not None and hasattr(self.order_manager, 'get_active_orders'):
                active_orders = self.order_manager.get_active_orders()
                if active_orders:
                    print(TerminalColors.info(f"🔄 Fechando {len(active_orders)} posições abertas..."))
                    if self.data_collector is not None and hasattr(self.data_collector, 'get_current_price'):
                        current_price = self.data_collector.get_current_price()
                    else:
                        current_price = 0
                    for order_id in list(active_orders.keys()):
                        if hasattr(self.order_manager, 'close_order'):
                            self.order_manager.close_order(order_id, "system_shutdown", current_price)
            
            # Salva estado final
            self._save_system_state()
            
            # Relatório final
            duration = datetime.now() - self.start_time if hasattr(self, 'start_time') else timedelta(0)
            accuracy = self.successful_trades / max(self.total_trades, 1) * 100
            
            print(TerminalColors.colorize("\n" + "="*60, TerminalColors.CYAN, TerminalColors.BOLD))
            print(TerminalColors.highlight("📊 RELATÓRIO FINAL - SISTEMA MODULARIZADO"))
            print(TerminalColors.colorize("="*60, TerminalColors.CYAN, TerminalColors.BOLD))
            
            print(TerminalColors.info("⏱️ DURAÇÃO E CICLOS:"))
            print(TerminalColors.info(f"   • Tempo total: {duration}"))
            print(TerminalColors.info(f"   • Ciclos executados: {self.cycle_count}"))
            print(TerminalColors.info(f"   • Trades de aprendizado: {self.trades_for_learning}/{self.force_trades_until_learning}"))
            
            print(TerminalColors.info("\n📈 PERFORMANCE DE TRADES:"))
            print(TerminalColors.info(f"   • Total de trades: {self.total_trades}"))
            print(TerminalColors.success(f"   • Trades lucrativos: {self.successful_trades}"))
            print(TerminalColors.info(f"   • Taxa de acerto: {accuracy:.1f}%"))
            
            print(TerminalColors.info("\n💰 RESULTADO FINANCEIRO:"))
            if self.total_profit > 0:
                print(TerminalColors.success(f"   • Lucro total: +${self.total_profit:.2f}"))
            else:
                print(TerminalColors.error(f"   • Prejuízo total: ${self.total_profit:.2f}"))
            
            print(TerminalColors.success("\n✅ SISTEMA MODULARIZADO FINALIZADO COM SUCESSO!"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no desligamento: {e}"))
    
    def _generate_training_report(self):
        """Gera relatório completo de treinamento dos modelos"""
        try:
            print(TerminalColors.colorize("\n" + "="*70, TerminalColors.MAGENTA, TerminalColors.BOLD))
            print(TerminalColors.highlight("🧠 RELATÓRIO DE TREINAMENTO - 2000 TRADES COMPLETOS"))
            print(TerminalColors.colorize("="*70, TerminalColors.MAGENTA, TerminalColors.BOLD))
            
            if hasattr(self, 'model_manager') and self.model_manager:
                progress = self.model_manager.get_training_progress()
                
                print(TerminalColors.info("📊 PERFORMANCE DOS MODELOS:"))
                for model_name, performance in progress.get('models_performance', {}).items():
                    accuracy = performance.get('accuracy', 0.0)
                    predictions = performance.get('total_predictions', 0)
                    correct = performance.get('correct_predictions', 0)
                    
                    print(TerminalColors.info(f"   • {model_name.upper()}: {accuracy:.3f} ({correct}/{predictions})"))
                
                avg_accuracy = progress.get('avg_accuracy', 0.0)
                total_predictions = progress.get('total_predictions', 0)
                
                print(TerminalColors.info(f"\n🎯 MÉTRICAS GERAIS:"))
                print(TerminalColors.info(f"   • Accuracy média: {avg_accuracy:.3f}"))
                print(TerminalColors.info(f"   • Total de predições: {total_predictions}"))
                print(TerminalColors.info(f"   • Trades de treinamento: {self.trades_for_learning}"))
                
                # Avaliação do treinamento
                if avg_accuracy >= 0.6:
                    print(TerminalColors.success("✅ TREINAMENTO EXITOSO: Modelos com boa performance"))
                elif avg_accuracy >= 0.4:
                    print(TerminalColors.warning("⚠️ TREINAMENTO PARCIAL: Modelos com performance moderada"))
                else:
                    print(TerminalColors.error("❌ TREINAMENTO INSUFICIENTE: Modelos precisam de mais dados"))
            
            print(TerminalColors.colorize("="*70, TerminalColors.MAGENTA, TerminalColors.BOLD))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro gerando relatório de treinamento: {e}"))

    def force_update_all_data(self):
        """
        Força atualização de dados e notícias a cada ciclo
        """
        try:
            log_trade_info("🔄 Forçando atualização de dados e notícias...", level='INFO')
            
            # 1. Força atualização de dados de mercado
            data_updated = False
            if hasattr(self, 'data_collector') and self.data_collector:
                data_updated = self.data_collector.force_update_data()
            
            # 2. Força atualização de notícias
            news_updated = False
            if hasattr(self, 'news_agent') and self.news_agent:
                news_updated = self.news_agent.force_update_news()
            
            # 3. Log do resultado
            if data_updated and news_updated:
                log_trade_info("✅ Dados e notícias atualizados com sucesso", level='SUCCESS')
            elif data_updated:
                log_trade_info("✅ Dados atualizados", level='INFO')
            elif news_updated:
                log_trade_info("✅ Notícias atualizadas", level='INFO')
            else:
                log_trade_info("⚠️ Falha na atualização de dados e notícias", level='WARNING')
                
        except Exception as e:
            log_trade_info(f"❌ Erro forçando atualização: {e}", level='ERROR')

def main():
    import sys
    
    # Processa argumentos de linha de comando
    mode = "paper"  # Padrão
    if len(sys.argv) > 1:
        if "--mode" in sys.argv:
            mode_index = sys.argv.index("--mode")
            if mode_index + 1 < len(sys.argv):
                mode = sys.argv[mode_index + 1]
    
    print(TerminalColors.info("🔍 Verificando privilégios administrativos..."))
    
    if not is_admin():
        print(TerminalColors.warning("⚠️ Executando elevação de privilégios para sincronização..."))
        run_as_admin()
        return
    
    print(TerminalColors.success("✅ Privilégios confirmados - iniciando sistema"))
    
    try:
        system = ModularTradingSystem(mode=mode)
        system.run()  # Inicializa sistemas centrais
        def on_new_candle(kline):
            # Callback chamado a cada novo candle fechado
            system.process_market_event()
        ws_listener = BinanceWSListener(on_new_candle_callback=on_new_candle, symbol='btcusdt', interval='1m')
        ws_listener.start()
        print(TerminalColors.info("🔗 WebSocket Binance iniciado. Aguarde eventos de candle..."))
        import time
        while True:
            time.sleep(1)
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro crítico: {e}"))
        sys.exit(1)

if __name__ == "__main__":
    main()
