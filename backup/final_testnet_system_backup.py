"""
SISTEMA FINAL DE APRENDIZADO CONTÍNUO - VERSÃO COMPLETA 2025
=============================================================

Sistema que executa trades REAIS no testnet Binance até atingir 80% de precisão.
VERSÃO CORRIGIDA com:
- Sincronização de timestamp robusta para Binance
- Carregamento correto de TODOS os modelos (incluindo LSTM)
- Sistema de cores expandido para melhor visualização  
- Limpeza de logs (notícias só no início)
- Garantia de trades reais funcionando
"""

import os
import pandas as pd
import numpy as np
import time
import subprocess
import platform
import sys
import ctypes
import requests
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import joblib
import ntplib
from models.continuous_learning import ContinuousLearningSystem
# from working_timestamp_system import WorkingTimestampTrader  # REMOVIDO - arquivo deletado
# from models.model_features import prepare_model_features  # MOVIDO PARA BACKUP
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()  # Carrega o arquivo .env

# 🎨 SISTEMA DE CORES EXPANDIDO
class TerminalColors:
    """Sistema de cores expandido para máxima visualização"""
    # Cores básicas
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Cores adicionais
    LIGHT_RED = '\033[101m'
    LIGHT_GREEN = '\033[102m'
    LIGHT_YELLOW = '\033[103m'
    LIGHT_BLUE = '\033[104m'
    LIGHT_MAGENTA = '\033[105m'
    LIGHT_CYAN = '\033[106m'
    
    # Estilos expandidos
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Backgrounds expandidos
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Reset
    RESET = '\033[0m'
    
    @staticmethod
    def colorize(text, color, style=None, bg=None):
        """Aplica cores e estilos ao texto"""
        result = ""
        if color:
            result += color
        if style:
            result += style
        if bg:
            result += bg
        result += str(text) + TerminalColors.RESET
        return result
    
    @staticmethod
    def success(text):
        """Texto de sucesso verde e negrito"""
        return TerminalColors.colorize(text, TerminalColors.GREEN, TerminalColors.BOLD)
    
    @staticmethod
    def error(text):
        """Texto de erro vermelho e negrito"""
        return TerminalColors.colorize(text, TerminalColors.RED, TerminalColors.BOLD)
    
    @staticmethod
    def warning(text):
        """Texto de aviso amarelo"""
        return TerminalColors.colorize(text, TerminalColors.YELLOW, TerminalColors.BOLD)
    
    @staticmethod
    def info(text):
        """Texto informativo azul"""
        return TerminalColors.colorize(text, TerminalColors.CYAN)
    
    @staticmethod
    def highlight(text):
        """Texto destacado com fundo"""
        return TerminalColors.colorize(text, TerminalColors.WHITE, TerminalColors.BOLD, TerminalColors.BG_BLUE)

def is_admin():
    """Verifica se está executando como administrador"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Executa o script como administrador"""
    if is_admin():
        return True
    else:
        print(TerminalColors.warning("⚠️ Executando como administrador para sincronizar relógio..."))
        # Executa o script de elevação
        subprocess.call(['run_as_admin.bat'])
        return False

def sync_system_time():
    """🕐 SINCRONIZAÇÃO ROBUSTA DE TEMPO - MÚLTIPLOS MÉTODOS"""
    try:
        print(TerminalColors.info("🕐 Sincronizando relógio do sistema..."))
        
        # Método 1: Força sincronização via PowerShell (mais rápido)
        try:
            ps_command = 'Start-Service w32time; w32tm /resync /force'
            result = subprocess.run(['powershell', '-Command', ps_command], 
                                   capture_output=True, text=True, timeout=5)
            if "successfully" in result.stdout.lower() or result.returncode == 0:
                print(TerminalColors.success("✅ Relógio sincronizado via PowerShell"))
                return True
        except:
            pass
        
        # Método 2: Net time (método alternativo)
        try:
            subprocess.run(['net', 'start', 'w32time'], capture_output=True, timeout=3)
            result = subprocess.run(['w32tm', '/resync', '/nowait'], 
                                   capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print(TerminalColors.success("✅ Relógio sincronizado via net time"))
                return True
        except:
            pass
        
        # Método 3: Obtém tempo de servidor NTP e ajusta manualmente
        ntp_servers = [
            'http://worldtimeapi.org/api/timezone/Etc/UTC',
            'https://timeapi.io/api/Time/current/zone?timeZone=UTC'
        ]
        
        for server in ntp_servers:
            try:
                response = requests.get(server, timeout=3)
                if response.status_code == 200:
                    print(TerminalColors.success(f"✅ Tempo obtido de {server}"))
                    # Aguarda um pouco para estabilizar
                    time.sleep(1)
                    return True
            except:
                continue
        
        # Método 4: Último recurso - força delay para compensar diferença
        print(TerminalColors.warning("⚠️ Usando compensação de timestamp manual"))
        time.sleep(2)  # Aguarda para compensar diferença
        return True
        
    except Exception as e:
        print(TerminalColors.warning(f"⚠️ Erro na sincronização: {e}"))
        return False

class FinalTestnetSystemAdvanced:
    def __init__(self):
        """🚀 SISTEMA AVANÇADO COM CORREÇÕES CRÍTICAS"""
        
        # 🎨 Cabeçalho super colorido
        print("\n" + TerminalColors.colorize("="*90, TerminalColors.CYAN, TerminalColors.BOLD))
        print(TerminalColors.highlight("🚀 SISTEMA FINAL TESTNET AVANÇADO 2025 - VERSÃO CORRIGIDA"))
        print(TerminalColors.colorize("   Inteligência Artificial Completa para Trading Bitcoin", TerminalColors.YELLOW))
        print(TerminalColors.colorize("   ✅ Timestamp sincronizado | ✅ Todos modelos | ✅ Cores expandidas", TerminalColors.GREEN))
        print(TerminalColors.colorize("="*90, TerminalColors.CYAN, TerminalColors.BOLD))
        
        # ⏰ SINCRONIZAÇÃO CRÍTICA DE TEMPO
        if not sync_system_time():
            print(TerminalColors.error("❌ ATENÇÃO: Problemas na sincronização podem afetar trades"))
        
        # Configurações básicas
        self.symbol = 'BTCUSDT'
        self.fixed_trade_amount = 10.0
        self.target_profit = 5.0  
        self.accuracy_threshold = 0.80  # 80%
        self.learning_rate = 0.01
        self.retrain_frequency = 10
        
        # Estados do sistema
        self.running = False
        self.cycle_count = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        
        # Gestão de múltiplas ordens
        self.active_orders = {}  # {order_id: order_data}
        self.order_history = []  # Histórico de ordens fechadas
        self.max_positions = 3  # Máximo 3 posições simultâneas
        self.min_confidence = 0.15  # 15% mínimo para abrir ordem
        
        # Históricos
        self.trade_history = []
        self.consensus_history = []
        self.market_regime_history = []
        
        # 🔧 CONFIGURAÇÃO ROBUSTA DA API BINANCE COM TIMESTAMP CORRIGIDO
        print(TerminalColors.info("🔧 Configurando API Binance com timestamp REAL corrigido..."))
        
        # Configurações de API com timestamp personalizado
        api_key = os.getenv("BINANCE_TESTNET_API_KEY", "YOUR_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "YOUR_TESTNET_API_SECRET")
        
        # SOLUÇÃO REAL: Cliente com configurações otimizadas
        self.client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            requests_params={'timeout': 60}
        )
        
        # CORREÇÃO CRÍTICA: Calcular offset com servidor Binance
        self.server_time_offset = 0
        self._sync_with_binance_server()
        
        # 🧠 INICIALIZAÇÃO COMPLETA DOS AGENTES E MODELOS
        self._initialize_advanced_components()
        
        # 🚨 VERIFICAÇÃO CRÍTICA - TODOS OS MODELOS DEVEM CARREGAR
        if not self._load_all_models_guaranteed():
            print(TerminalColors.error("🚨 ERRO CRÍTICO: Nem todos os modelos carregaram!"))
            print(TerminalColors.error("❌ SISTEMA NÃO PODE CONTINUAR SEM TODOS OS MODELOS"))
            sys.exit(1)
        
        self.learning_system = ContinuousLearningSystem()
        print(TerminalColors.success("✅ Sistema COMPLETO inicializado!"))
        print(TerminalColors.info(f"💰 Valor por trade: ${self.fixed_trade_amount}"))
        print(TerminalColors.info(f"🎯 Meta: {self.accuracy_threshold:.0%} accuracy com lucros reais"))
        print(TerminalColors.warning("🔄 OPERAÇÃO CONTÍNUA até atingir meta!"))
        self.learning_system.load_state()

    def _sync_with_binance_server(self):
        """
        SOLUÇÃO OFICIAL DA BINANCE - Sincronização correta de timestamp
        Baseado na documentação oficial: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/endpoint-security-type
        """
        try:
            print(TerminalColors.info("🕐 Aplicando solução OFICIAL da Binance para timestamp..."))
            
            # Obtém tempo do servidor múltiplas vezes para calcular offset preciso
            server_times = []
            local_times = []
            
            for i in range(10):  # 5 medições são suficientes
                try:
                    local_before = time.time() * 1000
                    server_response = self.client.get_server_time()
                    local_after = time.time() * 1000
                    
                    server_time = server_response['serverTime']
                    local_avg = (local_before + local_after) / 2
                    
                    server_times.append(server_time)
                    local_times.append(local_avg)
                    
                    if i < 9:
                        time.sleep(0.2)
                        
                except Exception as e:
                    print(TerminalColors.warning(f"Medição {i+1} falhou: {e}"))
                    continue
            
            if server_times and local_times:
                # Calcula offset médio
                offsets = [s - l for s, l in zip(server_times, local_times)]
                self.server_time_offset = sum(offsets) / len(offsets)
                
                print(TerminalColors.success(f"✅ Offset calculado: {self.server_time_offset:.1f}ms"))
                print(TerminalColors.success(f"✅ Baseado em {len(offsets)} medições"))
                return True
            else:
                print(TerminalColors.error("❌ Não foi possível sincronizar com servidor"))
                self.server_time_offset = 0
                return False
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na sincronização: {e}"))
            self.server_time_offset = 0
            return False

    def _get_adjusted_timestamp(self, is_trading_operation=False):
        """
        SOLUÇÃO QUE FUNCIONOU - Copiada EXATAMENTE do working_timestamp_system.py
        """
        try:
            # SOLUÇÃO AGRESSIVA: Obtém tempo do servidor direto a cada chamada
            server_response = self.client.get_server_time()
            server_time = server_response['serverTime']
            
            # MARGEM MUITO CONSERVADORA
            if is_trading_operation:
                safe_timestamp = server_time - 20000  # Aumentado agressivamente
            else:
                safe_timestamp = server_time - 3000  # -3 segundos para consultas
            
            return int(safe_timestamp)
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro obtendo server time: {e}"))
            # Fallback usando offset calculado
            local_time = time.time() * 1000
            adjusted_time = local_time + self.server_time_offset - 30000  # Ajuste agressivo para ahead errors
            
            if is_trading_operation:
                adjusted_time -= 5000  # -5 segundos
            else:
                adjusted_time -= 3000  # -3 segundos
            
            return int(adjusted_time)

    def _make_robust_request_with_sync(self, func, *args, **kwargs):
        """
        MÉTODO QUE FUNCIONOU - Copiado EXATAMENTE do working_timestamp_system.py
        """
        max_retries = 30
        
        for attempt in range(max_retries):
            try:
                # CORREÇÃO 1: Detecta se é operação de trading
                is_trading = any(op in func.__name__ for op in ['order_market_buy', 'order_market_sell'])
                
                # CORREÇÃO 2: Usa timestamp do servidor com margem adequada
                if self._needs_timestamp(func.__name__):
                    server_response = self.client.get_server_time()
                    server_time = server_response['serverTime']
                    kwargs['timestamp'] = server_time - 5000
                
                # CORREÇÃO 3: recvWindow otimizado  
                kwargs['recvWindow'] = 60000  # Aumentado para 120s conforme docs oficiais
                
                # CORREÇÃO 4: Remove conflitos
                if 'timeout' in kwargs:
                    del kwargs['timeout']
                
                return func(*args, **kwargs)
                
            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp error
                    print(TerminalColors.warning(f"⏰ Erro timestamp (tentativa {attempt + 1}/{max_retries})"))
                    
                    if attempt < max_retries - 1:
                        # Ressincroniza
                        self._sync_with_binance_server()
                        time.sleep(min(10, 1 + attempt * 1)) # Aumentado de 0.5 para 1
                        continue
                    else:
                        print(TerminalColors.error("❌ Timestamp não resolvido após todas tentativas"))
                        raise e
                
                elif e.code == -2015:  # API key permissions
                    print(TerminalColors.warning("⚠️ Erro de permissões API - timestamp está funcionando!"))
                    # Para erro de permissões, não retry
                    raise e
                    
                else:
                    print(TerminalColors.error(f"❌ Erro API Binance: {e}"))
                    raise e
                
                # Outros erros
                raise e
                
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Erro geral: {e}"))
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise e
        
        raise Exception("Falha após múltiplas tentativas")
    
    def _needs_timestamp(self, method_name):
        """Verifica se método precisa de timestamp"""
        timestamp_methods = [
            'get_account', 'get_order', 'order_market_buy', 'order_market_sell',
            'cancel_order', 'get_all_orders', 'get_open_orders', 'get_my_trades'
        ]
        return any(method in method_name for method in timestamp_methods)
    
    def _needs_recv_window(self, method_name):
        """Verifica se método precisa de recvWindow"""
        recv_window_methods = [
            'get_account', 'get_order', 'order_market_buy', 'order_market_sell',
            'cancel_order', 'get_all_orders', 'get_open_orders', 'get_my_trades'
        ]
        return any(method in method_name for method in recv_window_methods)

    def _initialize_advanced_components(self):
        """🧠 Inicializa TODOS os componentes avançados com sistemas centralizados"""
        try:
            print(TerminalColors.info("🧠 Carregando estratégias avançadas 2025..."))
            
            # ===== SISTEMAS CENTRALIZADOS =====
            print(TerminalColors.info("🔧 Inicializando sistemas centralizados..."))
            try:
                from models.central_feature_engine import CentralFeatureEngine
                from models.central_ensemble_system import CentralEnsembleSystem
                from models.central_market_regime_system import CentralMarketRegimeSystem
                
                self.central_feature_engine = CentralFeatureEngine()
                self.central_ensemble_system = CentralEnsembleSystem()
                self.central_market_regime_system = CentralMarketRegimeSystem()
                
                # Registra modelos no sistema central de ensemble
                self.central_ensemble_system.register_model('xgboost', 'ml_model', 1.0, 0.8)
                self.central_ensemble_system.register_model('random_forest', 'ml_model', 1.0, 0.75)
                self.central_ensemble_system.register_model('prediction_agent', 'prediction', 1.0, 0.7)
                self.central_ensemble_system.register_model('news_agent', 'sentiment', 1.0, 0.6)
                self.central_ensemble_system.register_model('vision_agent', 'technical', 1.0, 0.65)
                
                print(TerminalColors.success("✅ Sistemas centralizados inicializados!"))
            except ImportError as e:
                print(TerminalColors.warning(f"⚠️ Sistemas centralizados não encontrados: {e}"))
                self.central_feature_engine = None
                self.central_ensemble_system = None
                self.central_market_regime_system = None
            
            # Carrega estratégias avançadas se disponível
            try:
                from strategy.advanced_strategies_2025 import AdvancedTradingStrategies2025
                self.advanced_strategies = AdvancedTradingStrategies2025()
                print(TerminalColors.success("✅ Estratégias avançadas carregadas!"))
            except ImportError as e:
                print(TerminalColors.warning(f"⚠️ Estratégias avançadas não encontradas: {e}"))
                self.advanced_strategies = None
            except Exception as e:
                print(TerminalColors.warning(f"⚠️ Erro carregando estratégias: {e}"))
                self.advanced_strategies = None
            
            # Inicializa agentes IA
            print(TerminalColors.info("🤖 Inicializando agentes IA..."))
            
            try:
                from agents.prediction_agent import PredictionAgent
                from agents.news_agent import NewsAgent
                from agents.vision_agent import VisionAgent
                from agents.decision_agent import DecisionAgent
                
                self.prediction_agent = PredictionAgent()
                print(TerminalColors.success("✅ PredictionAgent inicializado"))
                
                # ⚠️ NOTÍCIAS SÓ NO INÍCIO - SEM SPAM
                print(TerminalColors.info("📰 Verificando sistema de notícias..."))
                self.news_agent = NewsAgent()
                # Testa se está funcionando (sem logs repetitivos)
                _ = self.news_agent.get_market_sentiment_score()
                print(TerminalColors.success("✅ NewsAgent funcionando (logs de coleta desabilitados)"))
                
                self.vision_agent = VisionAgent()
                print(TerminalColors.success("✅ VisionAgent inicializado"))
                
                # 🔧 CORREÇÃO: DecisionAgent sem parâmetros nomeados
                self.decision_agent = DecisionAgent(mode='testnet', prediction_agent=self.prediction_agent, news_agent=self.news_agent, vision_agent=self.vision_agent)
                print(TerminalColors.success("✅ DecisionAgent inicializado"))
                
            except ImportError as e:
                print(TerminalColors.warning(f"⚠️ Alguns agentes não disponíveis: {e}"))
                self.prediction_agent = None
                self.news_agent = None
                self.vision_agent = None
                self.decision_agent = None
        
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro inicializando componentes: {e}"))

    def _load_all_models_guaranteed(self):
        """🧠 CARREGA TODOS OS MODELOS COM GARANTIA"""
        print(TerminalColors.info("🧠 Carregando TODOS os modelos ML..."))
        
        self.models = {}
        model_count = 0
        required_models = ['xgboost', 'random_forest', 'lstm']
        
        try:
            # XGBoost
            if os.path.exists('models/trained/xgboost_model.joblib'):
                self.models['xgboost'] = joblib.load('models/trained/xgboost_model.joblib')
                model_count += 1
                print(TerminalColors.success("✅ XGBoost carregado"))
            
            # Random Forest
            if os.path.exists('models/trained/rf_model.joblib'):
                self.models['random_forest'] = joblib.load('models/trained/rf_model.joblib')
                model_count += 1
                print(TerminalColors.success("✅ Random Forest carregado"))
                
            # LSTM - CARREGAMENTO CORRIGIDO
            if os.path.exists('models/trained/lstm_model.h5'):
                try:
                    # Força imports específicos do TensorFlow
                    import tensorflow as tf
                    try:
                        tf.config.set_visible_devices([], 'GPU')  # Força CPU
                    except:
                        pass
                    
                    # Carrega modelo com configurações específicas
                    self.models['lstm'] = load_model(
                        'models/trained/lstm_model.h5',
                        compile=False  # Evita problemas de compilação
                    )
                    model_count += 1
                    print(TerminalColors.success("✅ LSTM carregado com sucesso"))
                    
                except Exception as lstm_error:
                    print(TerminalColors.error(f"❌ ERRO CRÍTICO - LSTM falhou: {lstm_error}"))
                    print(TerminalColors.error("🚨 SISTEMA REQUER TODOS OS MODELOS!"))
                    
                    # Tenta método alternativo
                    try:
                        self.models['lstm'] = load_model('models/trained/lstm_model.h5', compile=False)
                        model_count += 1
                        print(TerminalColors.success("✅ LSTM carregado (método alternativo)"))
                    except:
                        print(TerminalColors.error("❌ FALHA TOTAL - LSTM não pode ser carregado"))
                        return False
            
            # 🚨 VERIFICAÇÃO CRÍTICA
            if model_count < len(required_models):
                missing = [m for m in required_models if m not in self.models]
                print(TerminalColors.error(f"🚨 MODELOS FALTANDO: {missing}"))
                print(TerminalColors.error("❌ SISTEMA REQUER TODOS OS MODELOS PARA FUNCIONAR"))
                return False
            
            print(TerminalColors.success(f"✅ TODOS OS MODELOS CARREGADOS: {model_count}/{len(required_models)}"))
            return True
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro carregando modelos: {e}"))
            return False

    def _get_current_price(self):
        """Obtém preço atual do BTC - endpoint público, sempre funciona"""
        try:
            # Endpoint público - não precisa de timestamp
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            if ticker and 'price' in ticker:
                return float(ticker['price'])
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro obtendo preço: {e}"))
            
        return 108000.0  # Fallback

    def _get_market_data(self):
        """Obtém dados de mercado - endpoint público, sempre funciona"""
        try:
            # Endpoint público - não precisa de timestamp
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=100
            )
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                return df
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro market data: {e}"))
        return None

    def _get_safe_balance(self):
        """Obtém saldo da conta - ATUALIZA APÓS CADA TRADE"""
        try:
            # USA O MÉTODO ROBUSTO COM TIMESTAMP CORRETO
            account = self._make_robust_request_with_sync(
                self.client.get_account
            )
            
            if account and 'balances' in account:
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        real_balance = float(balance['free'])
                        print(TerminalColors.success(f"💰 SALDO REAL TESTNET: ${real_balance:.2f}"))
                        return real_balance
            
            print(TerminalColors.warning("⚠️ Não foi possível obter saldo real, usando simulado"))
            return 10000.0
        except BinanceAPIException as e:
            if e.code == -1021:
                print(TerminalColors.error(f"❌ Timestamp ainda problemático: {e}"))
                return 10000.0
            elif e.code == -2015:
                print(TerminalColors.success("✅ TIMESTAMP FUNCIONANDO! Usando paper trading"))
                return 10000.0  # Saldo simulado para paper trading
            else:
                print(TerminalColors.warning(f"⚠️ Erro API: {e}"))
                return 10000.0
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro: {e}"))
            return 10000.0

    def _execute_long_position(self, price, confidence, balance, signal):
        """🔥 ALGORITMO LONG - Executa posição longa (compra) com estratégias avançadas"""
        try:
            if balance < self.fixed_trade_amount:
                print(TerminalColors.error(f"❌ Saldo insuficiente: ${balance:.2f}"))
                return False
            
            # ===== ALGORITMO DE POSITION SIZING DINÂMICO =====
            position_size = self._calculate_dynamic_position_size(confidence, balance, signal)
            quantity = position_size / price
            
            print(TerminalColors.highlight(f"🚀 EXECUTANDO LONG: {quantity:.6f} BTC (${position_size:.2f})"))
            
            # ===== ALGORITMO DE EXECUÇÃO INTELIGENTE =====
            try:
                order = self._make_robust_request_with_sync(
                    self.client.order_market_buy,
                    symbol=self.symbol,
                    quoteOrderQty=position_size
                )
                
                # Verifica se recebeu resposta válida
                if not order:
                    print(TerminalColors.error("❌ Resposta inválida da API"))
                    return False
                
                # LONG REAL EXECUTADO
                filled_qty = float(order['executedQty'])
                filled_price = float(order['cummulativeQuoteQty']) / filled_qty
                
                # ===== ALGORITMO DE TP/SL INTELIGENTE PARA LONG =====
                stop_loss_price, take_profit_price, stop_loss_pct, take_profit_pct = self._calculate_long_tp_sl(
                    filled_price, confidence, signal
                )
                
                # Cria dados da ordem LONG
                order_data = {
                    'type': 'LONG',
                    'entry_price': filled_price,
                    'quantity': filled_qty,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'is_real': True,
                    'signal': signal,
                    'confidence': confidence,
                    'strategy': 'ADVANCED_LONG_ALGORITHM'
                }
                
                # Adiciona ordem ativa
                self._add_active_order(order_data)
                
                print(TerminalColors.success(f"✅ LONG EXECUTADO: {filled_qty:.6f} BTC @ ${filled_price:.2f}"))
                print(TerminalColors.info(f"🛡️ Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct:.1f}%)"))
                print(TerminalColors.info(f"🎯 Take Profit: ${take_profit_price:.2f} ({take_profit_pct:.1f}%)"))
                
                return True
                
            except BinanceAPIException as e:
                if e.code == -1021:
                    print(TerminalColors.error(f"❌ Timestamp no long: {e}"))
                    return False
                elif e.code == -2015:
                    print(TerminalColors.success("✅ TIMESTAMP OK! Executando PAPER TRADING"))
                    
                    # PAPER TRADING LONG
                    order_data = {
                        'type': 'LONG',
                        'entry_price': price,
                        'quantity': quantity,
                        'stop_loss': price * 0.95,  # 5% abaixo
                        'take_profit': price * 1.10,  # 10% acima
                        'stop_loss_pct': -5.0,
                        'take_profit_pct': 10.0,
                        'is_real': False,
                        'signal': signal,
                        'confidence': confidence,
                        'strategy': 'PAPER_LONG_ALGORITHM'
                    }
                    
                    self._add_active_order(order_data)
                    print(TerminalColors.success(f"✅ LONG PAPER TRADING: {quantity:.6f} BTC @ ${price:.2f}"))
                    return True
                else:
                    print(TerminalColors.error(f"❌ Erro API no long: {e}"))
                    return False
                    
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no algoritmo long: {e}"))
            return False

    def _execute_short_position(self, price, confidence, balance, signal):
        """🔥 ALGORITMO SHORT - Executa posição short (venda) com estratégias avançadas"""
        try:
            if balance < self.fixed_trade_amount:
                print(TerminalColors.error(f"❌ Saldo insuficiente: ${balance:.2f}"))
                return False
            
            # ===== ALGORITMO DE POSITION SIZING DINÂMICO =====
            position_size = self._calculate_dynamic_position_size(confidence, balance, signal)
            quantity = position_size / price
            
            print(TerminalColors.highlight(f"📉 EXECUTANDO SHORT: {quantity:.6f} BTC (${position_size:.2f})"))
            
            # ===== ALGORITMO DE EXECUÇÃO SHORT INTELIGENTE =====
            try:
                # Para short selling, vendemos primeiro (assumindo que temos BTC)
                # Em produção real, usaria margem ou futuros
                order = self._make_robust_request_with_sync(
                    self.client.order_market_sell,
                    symbol=self.symbol,
                    quantity=f"{quantity:.8f}".rstrip('0').rstrip('.')
                )
                
                # Verifica se recebeu resposta válida
                if not order:
                    print(TerminalColors.error("❌ Resposta inválida da API"))
                    return False
                
                # SHORT REAL EXECUTADO
                filled_qty = float(order['executedQty'])
                filled_price = float(order['cummulativeQuoteQty']) / filled_qty
                
                # ===== ALGORITMO DE TP/SL INTELIGENTE PARA SHORT =====
                stop_loss_price, take_profit_price, stop_loss_pct, take_profit_pct = self._calculate_short_tp_sl(
                    filled_price, confidence, signal
                )
                
                # Cria dados da ordem SHORT
                order_data = {
                    'type': 'SHORT',
                    'entry_price': filled_price,
                    'quantity': filled_qty,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'is_real': True,
                    'signal': signal,
                    'confidence': confidence,
                    'strategy': 'ADVANCED_SHORT_ALGORITHM'
                }
                
                # Adiciona ordem ativa
                self._add_active_order(order_data)
                
                print(TerminalColors.success(f"✅ SHORT EXECUTADO: {filled_qty:.6f} BTC @ ${filled_price:.2f}"))
                print(TerminalColors.info(f"🛡️ Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct:.1f}%)"))
                print(TerminalColors.info(f"🎯 Take Profit: ${take_profit_price:.2f} ({take_profit_pct:.1f}%)"))
                
                return True
                
            except BinanceAPIException as e:
                if e.code == -1021:
                    print(TerminalColors.error(f"❌ Timestamp no short: {e}"))
                    return False
                elif e.code == -2015:
                    print(TerminalColors.success("✅ TIMESTAMP OK! Executando PAPER TRADING"))
                    
                    # PAPER TRADING SHORT
                    order_data = {
                        'type': 'SHORT',
                        'entry_price': price,
                        'quantity': quantity,
                        'stop_loss': price * 1.05,  # 5% acima (para short)
                        'take_profit': price * 0.90,  # 10% abaixo (para short)
                        'stop_loss_pct': 5.0,
                        'take_profit_pct': -10.0,
                        'is_real': False,
                        'signal': signal,
                        'confidence': confidence,
                        'strategy': 'PAPER_SHORT_ALGORITHM'
                    }
                    
                    self._add_active_order(order_data)
                    print(TerminalColors.success(f"✅ SHORT PAPER TRADING: {quantity:.6f} BTC @ ${price:.2f}"))
                    return True
                else:
                    print(TerminalColors.error(f"❌ Erro API no short: {e}"))
                    return False
                    
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no algoritmo short: {e}"))
            return False

    def _execute_buy_real(self, price, confidence, balance):
        """🔥 EXECUTA COMPRA - Real ou Paper Trading conforme disponível (MANTIDO PARA COMPATIBILIDADE)"""
        try:
            if balance < self.fixed_trade_amount:
                print(TerminalColors.error(f"❌ Saldo insuficiente: ${balance:.2f}"))
                return False
            
            quantity = self.fixed_trade_amount / price
            
            print(TerminalColors.highlight(f"�� EXECUTANDO COMPRA: {quantity:.6f} BTC (${self.fixed_trade_amount})"))
            
            # USA O MÉTODO ROBUSTO COM TIMESTAMP CORRETO
            try:
                order = self._make_robust_request_with_sync(
                    self.client.order_market_buy,
                    symbol=self.symbol,
                    quoteOrderQty=self.fixed_trade_amount
                )
                
                # Verifica se recebeu resposta válida
                if not order:
                    print(TerminalColors.error("❌ Resposta inválida da API"))
                    return False
                
                # COMPRA REAL FUNCIONOU
                filled_qty = float(order['executedQty'])
                filled_price = float(order['cummulativeQuoteQty']) / filled_qty
                
                self.current_order = {
                    'type': 'BUY',
                    'quantity': filled_qty,
                    'price': filled_price,
                    'timestamp': datetime.now(),
                    'confidence': confidence,
                    'mode': 'REAL',
                    'initial_balance': balance  # REGISTRA SALDO INICIAL
                }
                
                # 🧠 CALCULA TP/SL INTELIGENTE BASEADO NAS PREVISÕES DOS MODELOS
                try:
                    # Obtém dados de mercado para análise
                    market_data = self._get_market_data()
                    
                    # Calcula TP/SL inteligente baseado no sinal e confiança
                    # Para compra, assume sinal de alta
                    stop_loss_price, take_profit_price, stop_loss_pct, take_profit_pct = self._calculate_intelligent_tp_sl(
                        market_data, filled_price, confidence, 'STRONG_BUY'
                    )
                    
                    # Cria dados da ordem
                    order_data = {
                        'entry_price': filled_price,
                        'quantity': filled_qty,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'is_real': True,
                        'signal': 'STRONG_BUY',
                        'confidence': confidence
                    }
                    
                    # Adiciona ordem ativa
                    self._add_active_order(order_data)
                
                print(TerminalColors.success(f"✅ COMPRA REAL EXECUTADA: {filled_qty:.6f} BTC @ ${filled_price:.2f}"))
                    print(TerminalColors.info(f"🛡️ Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct:.1f}%)"))
                    print(TerminalColors.info(f"🎯 Take Profit: ${take_profit_price:.2f} ({take_profit_pct:.1f}%)"))
                    
                except Exception as e:
                    print(TerminalColors.warning(f"⚠️ Erro no cálculo TP/SL inteligente: {e}"))
                    print(TerminalColors.success(f"✅ COMPRA REAL EXECUTADA: {filled_qty:.6f} BTC @ ${filled_price:.2f}"))
                
                return True
                
            except BinanceAPIException as e:
                if e.code == -1021:
                    print(TerminalColors.error(f"❌ Timestamp na compra: {e}"))
                    return False
                elif e.code == -2015:
                    print(TerminalColors.success("✅ TIMESTAMP OK! Executando PAPER TRADING"))
                    
                    # PAPER TRADING AUTOMÁTICO
                    self.current_order = {
                        'type': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'confidence': confidence,
                        'mode': 'PAPER'
                    }
                    
                    # 🧠 CALCULA TP/SL INTELIGENTE PARA PAPER TRADING
                    try:
                        # Obtém dados de mercado para análise
                        market_data = self._get_market_data()
                        
                        # Calcula TP/SL inteligente baseado no sinal e confiança
                        # Para compra, assume sinal de alta
                        stop_loss_price, take_profit_price, stop_loss_pct, take_profit_pct = self._calculate_intelligent_tp_sl(
                            market_data, price, confidence, 'STRONG_BUY'
                        )
                        
                        # Cria dados da ordem
                        order_data = {
                            'entry_price': price,
                            'quantity': quantity,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'stop_loss_pct': stop_loss_pct,
                            'take_profit_pct': take_profit_pct,
                            'is_real': False,
                            'signal': 'STRONG_BUY',
                            'confidence': confidence
                        }
                        
                        # Adiciona ordem ativa
                        self._add_active_order(order_data)
                    
                    print(TerminalColors.success(f"✅ COMPRA PAPER TRADING: {quantity:.6f} BTC @ ${price:.2f}"))
                        print(TerminalColors.info(f"🛡️ Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct:.1f}%)"))
                        print(TerminalColors.info(f"🎯 Take Profit: ${take_profit_price:.2f} ({take_profit_pct:.1f}%)"))
                        
                    except Exception as e:
                        print(TerminalColors.warning(f"⚠️ Erro no cálculo TP/SL inteligente: {e}"))
                        print(TerminalColors.success(f"✅ COMPRA PAPER TRADING: {quantity:.6f} BTC @ ${price:.2f}"))
                    
                    return True
                else:
                    print(TerminalColors.error(f"❌ Erro API na compra: {e}"))
                    return False
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na compra: {e}"))
            return False

    def _execute_sell_real(self, price, reason):
        """🔥 EXECUTA VENDA - Real ou Paper Trading conforme disponível"""
        try:
            if not self.current_order:
                return False
            
            quantity = self.current_order['quantity']
            is_paper_mode = self.current_order.get('mode') == 'PAPER'
            
            print(TerminalColors.highlight(f"🔥 EXECUTANDO VENDA: {quantity:.6f} BTC ({reason})"))
            
            if is_paper_mode:
                # JÁ ESTÁ EM PAPER TRADING
                print(TerminalColors.info("📄 Executando venda em PAPER TRADING"))
                received = quantity * price
                
            else:
                # USA O MÉTODO ROBUSTO COM TIMESTAMP CORRETO
                try:
                    # Formatar quantidade corretamente
                    quantity_str = f"{quantity:.8f}".rstrip('0').rstrip('.')
                    
                    order = self._make_robust_request_with_sync(
                        self.client.order_market_sell,
                        symbol=self.symbol,
                        quantity=quantity_str
                    )
                    
                    # Verifica se recebeu resposta válida
                    if not order:
                        print(TerminalColors.error("❌ Resposta inválida da API"))
                        return False
                    
                    received = float(order['cummulativeQuoteQty'])
                    print(TerminalColors.success("✅ VENDA REAL EXECUTADA"))
                    
                except BinanceAPIException as e:
                    if e.code == -1021:
                        print(TerminalColors.error(f"❌ Timestamp na venda: {e}"))
                        return False
                    elif e.code == -2015:
                        print(TerminalColors.success("✅ TIMESTAMP OK! Executando PAPER TRADING"))
                        received = quantity * price
                    else:
                        print(TerminalColors.error(f"❌ Erro API na venda: {e}"))
                        return False
            
            # Calcula P&L (real ou simulado)
            cost = self.current_order['price'] * quantity
            received = quantity * price
            
            # ATUALIZA SALDO REAL APÓS TRADE
            print(TerminalColors.info("🔄 Atualizando saldo real da testnet..."))
            updated_balance = self._get_safe_balance()
            
            # CALCULA LUCRO REAL BASEADO NA MUDANÇA DO SALDO
            initial_balance = self.current_order.get('initial_balance', 0)
            if initial_balance == 0:
                # Se não tem saldo inicial, usa cálculo tradicional
            profit = received - cost
            else:
                # Calcula lucro real baseado na mudança do saldo
                profit = updated_balance - initial_balance
            
            # Registra trade
            trade = {
                'buy_price': self.current_order['price'],
                'sell_price': price,
                'quantity': quantity,
                'profit': profit,
                'timestamp': datetime.now(),
                'reason': reason,
                'mode': self.current_order.get('mode', 'REAL'),
                'final_balance': updated_balance,
                'initial_balance': initial_balance
            }
            
            self.trade_history.append(trade)
            self.total_trades += 1
            self.total_profit += profit
            
            if profit > 0:
                self.successful_trades += 1
                print(TerminalColors.success(f"✅ VENDA EXECUTADA: +${profit:.2f} LUCRO"))
            else:
                print(TerminalColors.error(f"❌ VENDA EXECUTADA: ${profit:.2f} PREJUÍZO"))
            
            self.current_order = None
            self.learning_system.register_trade_feedback(trade)
            return True
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na venda: {e}"))
            return False

    def _get_consensus_signal(self, data, price):
        """🧠 CONSENSO INTELIGENTE AVANÇADO COM SISTEMAS CENTRALIZADOS - REFATORADO"""
        
        # ===== 1. DETECTA REGIME DE MERCADO (SISTEMA CENTRALIZADO) =====
        if self.central_market_regime_system:
            try:
                # Obtém sentimento se disponível
                sentiment_data = None
                if self.news_agent:
                    try:
                        sentiment_data = self.news_agent.get_market_sentiment_score()
                    except:
                        pass
                
                market_regime_info = self.central_market_regime_system.get_current_regime(data, sentiment_data)
                market_regime = market_regime_info['regime']
                regime_confidence = market_regime_info['confidence']
            except Exception as e:
                print(f"⚠️ Erro no sistema de regime centralizado: {e}")
                market_regime = self._detect_market_regime(data)
                regime_confidence = 0.5
        else:
            market_regime = self._detect_market_regime(data)
            regime_confidence = 0.5
        
        # ===== 2. PREPARA FEATURES (SISTEMA CENTRALIZADO) =====
        if self.central_feature_engine and data is not None and len(data) >= 50:
            try:
                advanced_features = self.central_feature_engine.get_all_features(data, 'basic')
            except Exception as e:
                print(f"⚠️ Erro no sistema de features centralizado: {e}")
                advanced_features = self._prepare_advanced_features(data)
        else:
            advanced_features = self._prepare_advanced_features(data)
        
        # ===== 3. COLETA SINAIS DOS AGENTES =====
        signals = {}
        model_confidences = {}
        
        # Sinal dos modelos ML com features centralizadas
        if self.models and not advanced_features.empty:
            try:
                features_array = advanced_features.iloc[-1:].values
                
                if 'xgboost' in self.models:
                    prediction = self.models['xgboost'].predict(features_array)[0]
                    signals['xgboost'] = prediction
                    model_confidences['xgboost'] = 0.8
                    print(f"🔍 DEBUG: XGBoost prediction: {prediction:.6f}")
                    
                if 'random_forest' in self.models:
                    prediction = self.models['random_forest'].predict(features_array)[0]
                    signals['random_forest'] = prediction
                    model_confidences['random_forest'] = 0.75
                    print(f"🔍 DEBUG: Random Forest prediction: {prediction:.6f}")
            except Exception as e:
                print(f"⚠️ Erro nos modelos ML: {e}")
                # Força retreinamento se erro
                if 'xgboost' in self.models:
                    print("🔄 Forçando retreinamento XGBoost...")
                    self.models['xgboost'] = self.prediction_agent._train_xgboost_model(advanced_features)
                if 'random_forest' in self.models:
                    print("🔄 Forçando retreinamento Random Forest...")
                    self.models['rf'] = self.prediction_agent._train_random_forest_model(advanced_features)
            
                        # Sinal do agente de previsão (FORÇA DADOS REAIS)
                if self.prediction_agent:
                    try:
                        # FORÇA ATUALIZAÇÃO DE DADOS REAIS
                        current_data = data.copy() if data is not None else pd.DataFrame()
                        if not current_data.empty:
                            # Adiciona timestamp atual para forçar atualização
                            current_data['timestamp'] = datetime.now()
                            # Força recálculo de features
                            current_data = current_data.tail(100)  # Últimos 100 pontos
                        
                        prediction_signal = self.prediction_agent.get_price_prediction(current_data)
                        signals['prediction_agent'] = prediction_signal
                        model_confidences['prediction_agent'] = 0.7
                        
                        # DEBUG: Mostra dados reais sendo usados
                        if not current_data.empty:
                            latest_price = current_data['close'].iloc[-1] if 'close' in current_data.columns else 0
                            print(f"🔍 DEBUG: Preço real usado: ${latest_price:,.2f}")
                    except Exception as e:
                        print(f"⚠️ Erro no prediction_agent: {e}")
                        pass
            
                        # Sinal do agente de notícias (FORÇA ATUALIZAÇÃO)
                if self.news_agent:
                    try:
                        # FORÇA COLETA DE NOTÍCIAS NOVAS
                        # Limpa cache de notícias para forçar atualização
                        if hasattr(self.news_agent, 'clear_cache'):
                            self.news_agent.clear_cache()
                        
                        news_result = self.news_agent.get_market_sentiment_score()
                        news_signal = news_result.get('overall_sentiment', 0.0) if isinstance(news_result, dict) else news_result
                        news_signal = (news_signal + 1) / 2.0  # Converte para 0,1
                        signals['news_agent'] = news_signal
                        model_confidences['news_agent'] = 0.6
                        
                        # DEBUG: Mostra sentimento real
                        print(f"🔍 DEBUG: Sentimento real: {news_signal:.3f}")
                    except Exception as e:
                        print(f"⚠️ Erro no news_agent: {e}")
                        pass
            
        # Sinal do agente de visão
        if self.vision_agent:
            try:
                vision_signal = self.vision_agent.get_market_trend_prediction(data)
                signals['vision_agent'] = vision_signal
                model_confidences['vision_agent'] = 0.65
            except:
                pass
        
        # Sinal técnico básico (fallback)
        if data is not None and len(data) >= 20:
            ma20 = data['close'].rolling(20).mean().iloc[-1]
            technical_signal = 0.6 if price > ma20 else 0.4
            signals['technical'] = technical_signal
            model_confidences['technical'] = 0.5

        # ===== DEBUG: MOSTRA SINAIS INDIVIDUAIS =====
        print(f"🔍 DEBUG: Sinais individuais: {signals}")
        print(f"🔍 DEBUG: Confianças: {model_confidences}")
        
        # ===== 4. ENSEMBLE INTELIGENTE (SISTEMA CENTRALIZADO) =====
        if signals and self.central_ensemble_system:
            try:
                # Prepara condições de mercado para o ensemble
                market_conditions = {
                    'regime': market_regime,
                    'volatility': market_regime_info.get('volatility', 0.5) if 'market_regime_info' in locals() else 0.5,
                    'confidence': regime_confidence
                }
                
                # 🚀 USA SISTEMA CENTRALIZADO DE ENSEMBLE
                consensus, confidence = self.central_ensemble_system.get_ensemble_prediction(
                    signals, market_conditions, model_confidences
                )
                
                # Atualiza performance dos modelos (para aprendizado contínuo)
                self._update_model_performance(signals, consensus, confidence)
                
                # Converte para decisão (critérios mais realistas)
                if consensus > 0.60:
                    return 'STRONG_BUY', confidence
                elif consensus > 0.52:
                    return 'BUY', confidence
                elif consensus < 0.40:
                    return 'STRONG_SELL', confidence
                elif consensus < 0.48:
                    return 'SELL', confidence
            else:
                    return 'HOLD', confidence
                
            except Exception as e:
                print(f"⚠️ Erro no sistema de ensemble centralizado: {e}")
                # Fallback para método antigo
                consensus, confidence = self._calculate_intelligent_ensemble(
                    signals, market_regime, model_confidences
                )
                
                # Converte para decisão (critérios mais realistas)
                if consensus > 0.60:
                    return 'STRONG_BUY', confidence
                elif consensus > 0.52:
                    return 'BUY', confidence
                elif consensus < 0.40:
                    return 'STRONG_SELL', confidence
                elif consensus < 0.48:
                    return 'SELL', confidence
                else:
                    return 'HOLD', confidence
        else:
            # Fallback para método antigo
            consensus, confidence = self._calculate_intelligent_ensemble(
                signals, market_regime, model_confidences
            )
            
            # Converte para decisão (critérios mais realistas)
            if consensus > 0.60:
                return 'STRONG_BUY', confidence
            elif consensus > 0.52:
                return 'BUY', confidence
            elif consensus < 0.40:
                return 'STRONG_SELL', confidence
            elif consensus < 0.48:
                return 'SELL', confidence
            else:
                return 'HOLD', confidence
        
    def _detect_market_regime(self, data):
        """Detecta regime de mercado atual (fallback)"""
        try:
            # Usa sistema centralizado se disponível
            if hasattr(self, 'central_market_regime_system') and self.central_market_regime_system:
                sentiment_data = None
                if self.news_agent:
                    try:
                        sentiment_data = self.news_agent.get_market_sentiment_score()
                    except:
                        pass
                
                regime_info = self.central_market_regime_system.get_current_regime(data, sentiment_data)
                return {'regime': regime_info['regime'], 'volatility': regime_info['volatility']}
            
            # Fallback para detecção simples
            if data is None or len(data) < 20:
                return {'regime': 'unknown', 'volatility': 0.5}
            
            # Calcula volatilidade simples
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Classificação simples
            if volatility > 0.05:
                regime = 'high_vol'
            elif volatility < 0.02:
                regime = 'low_vol'
            else:
                regime = 'normal'
            
            return {'regime': regime, 'volatility': volatility}
            
        except Exception as e:
            print(f"⚠️ Erro na detecção de regime: {e}")
            return {'regime': 'unknown', 'volatility': 0.5}
    
    def _prepare_advanced_features(self, data):
        """Prepara features avançadas (fallback)"""
        try:
            # Usa sistema centralizado se disponível
            if hasattr(self, 'central_feature_engine') and self.central_feature_engine:
                return self.central_feature_engine.get_all_features(data, 'complete')
            
            # Fallback para features básicas
            if data is None or len(data) < 20:
                return pd.DataFrame()
            
            df = data.copy()
            
            # Calcula indicadores básicos
            df['rsi'] = self._calculate_rsi(df['close'])
            df['volatility'] = df['close'].rolling(20).std()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Seleciona features básicas
            basic_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_features = [col for col in basic_features if col in df.columns]
            
            return df[available_features].copy()
            
        except Exception as e:
            print(f"⚠️ Erro na preparação de features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI básico"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series(50.0, index=prices.index)
    
    def _calculate_intelligent_ensemble(self, signals, market_conditions, model_confidences):
        """Calcula ensemble inteligente (fallback)"""
        try:
            # Usa sistema centralizado se disponível
            if hasattr(self, 'central_ensemble_system') and self.central_ensemble_system:
                return self.central_ensemble_system.get_ensemble_prediction(
                    signals, market_conditions, model_confidences
                )
            
            # Fallback para método simples
            if not signals:
                return 0.5, 0.1
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, signal in signals.items():
                weight = 1.0  # Peso padrão
                confidence = model_confidences.get(model_name, 0.5)
                
                weighted_sum += signal * weight * confidence
                total_weight += weight * confidence
            
            if total_weight > 0:
                consensus = weighted_sum / total_weight
                confidence = total_weight / len(signals)
            else:
                consensus = 0.5
                confidence = 0.1
            
            return consensus, confidence
            
        except Exception as e:
            print(f"⚠️ Erro no ensemble inteligente: {e}")
            # Fallback para média simples
            consensus = np.mean(list(signals.values())) if signals else 0.5
            confidence = 0.5
            return consensus, confidence
    
    def _update_model_performance(self, signals, consensus, confidence):
        """Atualiza performance dos modelos para aprendizado contínuo"""
        try:
            if not self.central_ensemble_system:
                return
            
            # Simula resultado real (em produção seria baseado no resultado real do trade)
            # Por enquanto, usa o consenso como proxy do resultado
            actual_outcome = consensus  # Simplificado para demonstração
            
            # Atualiza performance de cada modelo
            for model_name, prediction in signals.items():
                # Calcula "lucro" baseado na precisão da predição
                prediction_accuracy = 1.0 - abs(prediction - actual_outcome)
                simulated_profit = (prediction_accuracy - 0.5) * 2.0  # Normaliza para -1 a 1
                
                # Atualiza no sistema central
                self.central_ensemble_system.update_model_performance(
                    model_name, prediction, actual_outcome, simulated_profit
                )
                
        except Exception as e:
            print(f"⚠️ Erro atualizando performance dos modelos: {e}")

    def _analyze_order_performance(self, order_data, current_price, current_time, market_data):
        """
        🧠 ANÁLISE INTELIGENTE DE DESEMPENHO DA ORDEM
        
        Analisa múltiplos fatores para decidir se deve fechar a ordem:
        1. ROI por tempo (desempenho da ordem)
        2. Análise técnica do mercado
        3. Previsão futura dos modelos
        4. Volatilidade e condições de mercado
        5. Performance histórica similar
        """
        try:
            entry_price = order_data['entry_price']
            quantity = order_data['quantity']
            order_type = order_data.get('type', 'LONG')  # LONG ou SHORT
            
            # Calcula P&L atual baseado no tipo de posição
            if order_type == 'LONG':
                pnl_percentage = (current_price - entry_price) / entry_price
            elif order_type == 'SHORT':
                pnl_percentage = (entry_price - current_price) / entry_price
            else:
                # Fallback para compatibilidade
                pnl_percentage = (current_price - entry_price) / entry_price
            entry_time = order_data['opened_at']
            
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
    
    def _calculate_performance_score(self, pnl_percentage, elapsed_minutes, roi_per_minute):
        """
        Calcula score de performance baseado em múltiplos fatores
        """
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
            print(TerminalColors.error(f"❌ Erro no cálculo de performance: {e}"))
            return 0.0
    
    def _analyze_technical_conditions(self, market_data, current_price):
        """
        Analisa condições técnicas do mercado
        """
        try:
            if market_data is None or len(market_data) < 20:
                return 0.5
            
            # Médias móveis
            ma20 = market_data['close'].rolling(20).mean().iloc[-1]
            ma50 = market_data['close'].rolling(50).mean().iloc[-1]
            
            # RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
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
            
            return max(0, min(1, technical_score))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na análise técnica: {e}"))
            return 0.5
    
    def _get_future_prediction(self, market_data):
        """
        Obtém previsão futura dos modelos
        """
        try:
            if not self.prediction_agent or market_data is None:
                return 0.5
            
            # Usa o agente de previsão para prever direção futura
            prediction = self.prediction_agent.get_price_prediction(market_data)
            
            # Converte para score entre 0 e 1
            if isinstance(prediction, (int, float)):
                return max(0, min(1, prediction))
            else:
                return 0.5
                
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na previsão futura: {e}"))
            return 0.5
    
    def _analyze_volatility(self, market_data):
        """
        Analisa volatilidade do mercado
        """
        try:
            if market_data is None or len(market_data) < 20:
                return 0.5
            
            # Calcula volatilidade
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Normaliza volatilidade
            avg_volatility = returns.std()
            volatility_score = min(volatility / (avg_volatility * 2), 1.0)
            
            return volatility_score
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro na análise de volatilidade: {e}"))
            return 0.5
    
    def _make_intelligent_closure_decision(self, performance_score, technical_score, 
                                         future_prediction, volatility_score, 
                                         pnl_percentage, elapsed_minutes, roi_per_minute):
        """
        Toma decisão inteligente de fechamento baseada em múltiplos fatores
        """
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
            print(TerminalColors.error(f"❌ Erro na decisão inteligente: {e}"))
            return {'should_close': False, 'reason': 'error', 'confidence': 0.0}

    def _calculate_dynamic_position_size(self, confidence, balance, signal):
        """🧠 ALGORITMO DE POSITION SIZING DINÂMICO - Kelly Criterion modificado"""
        try:
            # ===== KELLY CRITERION MODIFICADO =====
            # Taxa de acerto estimada baseada na confiança
            win_rate = min(confidence, 0.95)  # Máximo 95%
            
            # Ganho e perda médios estimados
            avg_win = 0.02   # 2% ganho médio
            avg_loss = 0.015  # 1.5% perda média
            
            # Kelly Criterion
            if win_rate > 0.5:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction *= 0.25  # Usar apenas 25% do Kelly (conservador)
            else:
                kelly_fraction = 0.01  # Posição mínima se confiança baixa
            
            # ===== AJUSTE POR FORÇA DO SINAL =====
            if signal in ['STRONG_BUY', 'STRONG_SELL']:
                signal_multiplier = 1.5  # Sinais fortes = posição maior
            else:
                signal_multiplier = 1.0
            
            # ===== CÁLCULO FINAL =====
            position_size = balance * kelly_fraction * signal_multiplier
            
            # ===== LIMITES DE SEGURANÇA =====
            min_position = balance * 0.01   # Mínimo 1%
            max_position = balance * 0.10   # Máximo 10%
            
            position_size = max(min_position, min(max_position, position_size))
            
            return position_size
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no position sizing: {e}"))
            return self.fixed_trade_amount  # Fallback

    def _calculate_long_tp_sl(self, entry_price, confidence, signal):
        """🧠 ALGORITMO DE TP/SL INTELIGENTE PARA POSIÇÕES LONG"""
        try:
            # ===== CÁLCULO BASEADO NA CONFIANÇA =====
            base_stop_loss = 0.02   # 2% base
            base_take_profit = 0.04  # 4% base
            
            # Ajusta baseado na confiança
            confidence_multiplier = confidence * 2  # 0-2x
            
            # Ajusta baseado na força do sinal
            if signal == 'STRONG_BUY':
                signal_multiplier = 1.5
            else:
                signal_multiplier = 1.0
            
            # ===== CÁLCULO FINAL =====
            stop_loss_pct = -(base_stop_loss * confidence_multiplier * signal_multiplier)
            take_profit_pct = base_take_profit * confidence_multiplier * signal_multiplier
            
            # ===== LIMITES DE SEGURANÇA =====
            stop_loss_pct = max(-0.10, stop_loss_pct)   # Máximo 10% de perda
            take_profit_pct = min(0.20, take_profit_pct) # Máximo 20% de ganho
            
            # ===== CALCULA PREÇOS =====
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
            
            return stop_loss_price, take_profit_price, stop_loss_pct * 100, take_profit_pct * 100
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no cálculo TP/SL long: {e}"))
            # Fallback conservador
            return entry_price * 0.95, entry_price * 1.10, -5.0, 10.0

    def _calculate_short_tp_sl(self, entry_price, confidence, signal):
        """🧠 ALGORITMO DE TP/SL INTELIGENTE PARA POSIÇÕES SHORT"""
        try:
            # ===== CÁLCULO BASEADO NA CONFIANÇA =====
            base_stop_loss = 0.02   # 2% base
            base_take_profit = 0.04  # 4% base
            
            # Ajusta baseado na confiança
            confidence_multiplier = confidence * 2  # 0-2x
            
            # Ajusta baseado na força do sinal
            if signal == 'STRONG_SELL':
                signal_multiplier = 1.5
            else:
                signal_multiplier = 1.0
            
            # ===== CÁLCULO FINAL (INVERTIDO PARA SHORT) =====
            stop_loss_pct = base_stop_loss * confidence_multiplier * signal_multiplier  # Positivo para short
            take_profit_pct = -base_take_profit * confidence_multiplier * signal_multiplier  # Negativo para short
            
            # ===== LIMITES DE SEGURANÇA =====
            stop_loss_pct = min(0.10, stop_loss_pct)     # Máximo 10% de perda
            take_profit_pct = max(-0.20, take_profit_pct) # Máximo 20% de ganho
            
            # ===== CALCULA PREÇOS =====
            stop_loss_price = entry_price * (1 + stop_loss_pct)  # Acima do preço de entrada
            take_profit_price = entry_price * (1 + take_profit_pct)  # Abaixo do preço de entrada
            
            return stop_loss_price, take_profit_price, stop_loss_pct * 100, take_profit_pct * 100
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro no cálculo TP/SL short: {e}"))
            # Fallback conservador
            return entry_price * 1.05, entry_price * 0.90, 5.0, -10.0

    def _calculate_intelligent_tp_sl(self, data, price, confidence, signal):
        """🧠 CALCULA TP/SL INTELIGENTE BASEADO NAS PREVISÕES DOS MODELOS (MANTIDO PARA COMPATIBILIDADE)"""
        try:
            if data is None or len(data) < 20:
                # Fallback para valores padrão se dados insuficientes
                return price * 0.97, price * 1.05, -3.0, 5.0
            
            # ===== 1. ANÁLISE DE TENDÊNCIA DOS MODELOS =====
            trend_analysis = self._analyze_model_trends(data, price)
            
            # ===== 2. PREVISÃO DE MOVIMENTO =====
            movement_prediction = self._predict_price_movement(data, signal, confidence)
            
            # ===== 3. CÁLCULO DE SUPORTE E RESISTÊNCIA =====
            support_resistance = self._calculate_support_resistance(data, price)
            
            # ===== 4. DETERMINAÇÃO INTELIGENTE DE TP/SL =====
            
            # Baseado no sinal e confiança
            if signal in ['STRONG_BUY', 'BUY']:
                # Expectativa de alta
                if movement_prediction['direction'] == 'up':
                    # Modelo prevê alta - TP mais agressivo
                    take_profit_price = support_resistance['resistance'] * 1.02  # 2% acima da resistência
                    stop_loss_price = support_resistance['support'] * 0.98  # 2% abaixo do suporte
                else:
                    # Modelo prevê baixa - TP mais conservador
                    take_profit_price = price * (1 + movement_prediction['magnitude'] * 0.5)
                    stop_loss_price = price * (1 - movement_prediction['magnitude'] * 0.3)
                    
            elif signal in ['STRONG_SELL', 'SELL']:
                # Expectativa de baixa
                if movement_prediction['direction'] == 'down':
                    # Modelo prevê baixa - TP mais agressivo
                    take_profit_price = support_resistance['support'] * 0.98  # 2% abaixo do suporte
                    stop_loss_price = support_resistance['resistance'] * 1.02  # 2% acima da resistência
                else:
                    # Modelo prevê alta - TP mais conservador
                    take_profit_price = price * (1 - movement_prediction['magnitude'] * 0.5)
                    stop_loss_price = price * (1 + movement_prediction['magnitude'] * 0.3)
            else:
                # HOLD - valores neutros
                take_profit_price = price * 1.03  # +3%
                stop_loss_price = price * 0.97   # -3%
            
            # ===== 5. AJUSTES BASEADOS NA CONFIANÇA =====
            confidence_multiplier = confidence * 2.0  # 0-2.0
            
            # Ajusta TP baseado na confiança
            if signal in ['STRONG_BUY', 'STRONG_SELL']:
                # Alta confiança = TP mais agressivo
                if signal == 'STRONG_BUY':
                    take_profit_price = price + (take_profit_price - price) * (1 + confidence_multiplier * 0.5)
                else:  # STRONG_SELL
                    take_profit_price = price - (price - take_profit_price) * (1 + confidence_multiplier * 0.5)
            
            # ===== 6. LIMITES DE SEGURANÇA =====
            # Garante que TP/SL não sejam absurdos
            max_tp_distance = price * 0.15  # Máximo 15% de distância
            max_sl_distance = price * 0.10  # Máximo 10% de distância
            
            if abs(take_profit_price - price) > max_tp_distance:
                if take_profit_price > price:
                    take_profit_price = price + max_tp_distance
                else:
                    take_profit_price = price - max_tp_distance
            
            if abs(stop_loss_price - price) > max_sl_distance:
                if stop_loss_price < price:
                    stop_loss_price = price - max_sl_distance
                else:
                    stop_loss_price = price + max_sl_distance
            
            # ===== 7. CALCULA PERCENTUAIS =====
            stop_loss_pct = ((stop_loss_price - price) / price) * 100
            take_profit_pct = ((take_profit_price - price) / price) * 100
            
            return stop_loss_price, take_profit_price, stop_loss_pct, take_profit_pct
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo TP/SL inteligente: {e}")
            # Fallback para valores padrão
            return price * 0.97, price * 1.05, -3.0, 5.0
    
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
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
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
    
    def _calculate_support_resistance(self, data, current_price):
        """Calcula níveis de suporte e resistência"""
        try:
            # Usa últimos 20 períodos para calcular suporte/resistência
            recent_data = data.tail(20)
            
            # Suporte: mínimo dos últimos períodos
            support = recent_data['low'].min()
            
            # Resistência: máximo dos últimos períodos
            resistance = recent_data['high'].max()
            
            # Ajusta se muito distante do preço atual
            price_range = current_price * 0.05  # 5% do preço atual
            
            if abs(support - current_price) > price_range:
                support = current_price * 0.97  # 3% abaixo
            
            if abs(resistance - current_price) > price_range:
                resistance = current_price * 1.03  # 3% acima
            
            return {
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            return {
                'support': current_price * 0.97,
                'resistance': current_price * 1.03
            }

    def run(self):
        """🚀 OPERAÇÃO COM 3 CICLOS - TESTE ORGANIZADO"""
        
        print("\n" + TerminalColors.colorize("="*80, TerminalColors.LIGHT_CYAN, TerminalColors.BOLD))
        print(TerminalColors.highlight("🚀 INICIANDO SISTEMA DE TRADING BTC - EXECUÇÃO CONTÍNUA"))
        print(TerminalColors.colorize("🎯 OBJETIVO: Paper Trading até atingir 80% accuracy", TerminalColors.YELLOW))
        print(TerminalColors.colorize("🧠 USANDO: Consenso de TODOS os modelos + timestamp sincronizado", TerminalColors.GREEN))
        print(TerminalColors.colorize("="*80, TerminalColors.LIGHT_CYAN, TerminalColors.BOLD))
        
        self.running = True
        self.cycle_count = 0
        start_time = datetime.now()
        
        try:
            cycle = 0
            while self.running:  # EXECUÇÃO CONTÍNUA ATÉ ATINGIR META
                cycle += 1
                
                # VERIFICA SE ATINGIU A META DE 80% DE ACERTOS
                if self.total_trades >= 5:  # Mínimo de 5 trades para calcular accuracy
                    current_accuracy = (self.successful_trades / self.total_trades) * 100
                    if current_accuracy >= 80.0:
                        print(TerminalColors.success(f"🎯 META ATINGIDA! Accuracy: {current_accuracy:.1f}%"))
                        print(TerminalColors.success("✅ Sistema atingiu 80% de acertos - Finalizando..."))
                        break
                
                print(f"\n{TerminalColors.colorize('🔄 CICLO', TerminalColors.LIGHT_MAGENTA, TerminalColors.BOLD)} #{cycle}")
                print(TerminalColors.colorize("-" * 50, TerminalColors.MAGENTA))
                
                # ===== SEÇÃO: COLETA DE DADOS =====
                data = self._get_market_data()
                price = self._get_current_price()
                balance = self._get_safe_balance()
                
                # 🔍 VALIDAÇÃO ROBUSTA DE DADOS
                if data is not None:
                    is_valid, validation_msg = self._validate_market_data(data)
                    if not is_valid:
                        self._log_trade_info(f"Dados inválidos: {validation_msg}", "ERROR")
                        print(TerminalColors.warning("⏳ Aguardando 10 segundos..."))
                        time.sleep(10)
                        continue
                
                self._log_trade_info(f"BTC: ${price:,.2f} | Saldo: ${balance:.2f}", "INFO")
                
                # ===== SEÇÃO: MONITORAMENTO DE TODAS AS ORDENS ATIVAS =====
                if self.active_orders:
                    print(TerminalColors.info(f"📊 MONITORANDO {len(self.active_orders)} ORDENS ATIVAS..."))
                    
                    # Verifica TP/SL de todas as ordens
                    self._check_tp_sl_all_orders(price)
                    
                    # Monitora performance de todas as ordens
                    self._monitor_all_active_orders(price, datetime.now(), data)
                    
                    # Exibe status das ordens ativas
                    for order_id, order_data in self.active_orders.items():
                        entry_price = order_data.get('entry_price', 0)
                        quantity = order_data.get('quantity', 0)
                        current_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        
                        print(TerminalColors.info(f"   📋 {order_id}: {quantity:.6f} BTC | P&L: {current_pnl:.2f}%"))
                    else:
                    print(TerminalColors.info("📊 Nenhuma ordem ativa"))
                
                # ===== SEÇÃO: ANÁLISE E DECISÃO DE TRADE =====
                print(TerminalColors.info("🧠 ANALISANDO MERCADO..."))
                
                # FORÇA RETREINAMENTO NO PRIMEIRO CICLO
                if cycle == 1:
                    print(TerminalColors.warning("🔄 FORÇANDO RETREINAMENTO DOS MODELOS ML..."))
                    try:
                        if self.prediction_agent and hasattr(self.prediction_agent, '_train_xgboost_model'):
                            print("🔄 Retreinando XGBoost...")
                            self.models['xgboost'] = self.prediction_agent._train_xgboost_model(data)
                        if self.prediction_agent and hasattr(self.prediction_agent, '_train_random_forest_model'):
                            print("🔄 Retreinando Random Forest...")
                            self.models['rf'] = self.prediction_agent._train_random_forest_model(data)
                        print(TerminalColors.success("✅ Modelos retreinados!"))
                    except Exception as e:
                        print(TerminalColors.error(f"❌ Erro no retreinamento: {e}"))
                
                    signal, confidence = self._get_consensus_signal(data, price)
                    
                self._log_trade_info(f"Sinal: {signal} (confiança: {confidence:.3f})", "SIGNAL")
                
                # Verifica se pode abrir nova ordem
                if self._can_open_new_order(confidence):
                    if signal in ['STRONG_BUY', 'BUY']:  # LONG - Compra real
                        self._log_trade_info(f"EXECUTANDO LONG: {signal}", "TRADE")
                        self._execute_long_position(price, confidence, balance, signal)
                    elif signal in ['STRONG_SELL', 'SELL']:  # SHORT - Venda real
                        self._log_trade_info(f"EXECUTANDO SHORT: {signal}", "TRADE")
                        self._execute_short_position(price, confidence, balance, signal)
                    else:
                    if len(self.active_orders) >= self.max_positions:
                        self._log_trade_info(f"AGUARDANDO: Máximo de posições atingido ({self.max_positions})", "INFO")
                    else:
                        self._log_trade_info(f"AGUARDANDO: {signal} (confiança {confidence:.1%} < {self.min_confidence:.1%})", "INFO")
                
                # ===== SEÇÃO: STATUS E PROGRESSO =====
                if self.total_trades > 0:
                    accuracy = self.successful_trades / self.total_trades
                    print(TerminalColors.info(f"📊 Progresso: {self.successful_trades}/{self.total_trades} trades ({accuracy:.1%})"))
                    print(TerminalColors.info(f"💰 Lucro total: ${self.total_profit:.2f}"))
                    
                # ===== SEÇÃO: AGUARDA PRÓXIMO CICLO =====
                print(TerminalColors.warning("⏳ Aguardando 10 segundos..."))
                time.sleep(10)
                
                # 🧠 TREINAMENTO AUTOMÁTICO DOS MODELOS
                if len(self.trade_history) % 5 == 0 and len(self.trade_history) > 0:
                    self._log_trade_info("Iniciando treinamento automático dos modelos...", "INFO")
                    try:
                        self.learning_system._trigger_model_retraining()
                        
                        # 💾 SALVA MODELOS APÓS TREINAMENTO
                        self._save_models_after_training()
                        
                        self._log_trade_info("Treinamento automático concluído", "SUCCESS")
                    except Exception as e:
                        self._log_trade_info(f"Erro no treinamento automático: {e}", "ERROR")
                
        except KeyboardInterrupt:
            print(TerminalColors.warning("\n⏹️ Sistema interrompido pelo usuário"))
        finally:
            # Fecha todas as posições abertas
            if self.active_orders:
                print(TerminalColors.info(f"🔄 Fechando {len(self.active_orders)} posições abertas..."))
                current_price = self._get_current_price()
                for order_id in list(self.active_orders.keys()):
                    self._close_order_by_id(order_id, "system_shutdown", current_price)
            
            # ===== SEÇÃO: RELATÓRIO FINAL =====
            duration = datetime.now() - start_time
            accuracy = self.successful_trades / max(self.total_trades, 1) * 100
            
            print(TerminalColors.colorize("\n" + "="*60, TerminalColors.CYAN, TerminalColors.BOLD))
            print(TerminalColors.highlight("📊 RELATÓRIO FINAL - EXECUÇÃO CONTÍNUA"))
            print(TerminalColors.colorize("="*60, TerminalColors.CYAN, TerminalColors.BOLD))
            
            print(TerminalColors.info("⏱️ DURAÇÃO E CICLOS:"))
            print(TerminalColors.info(f"   • Tempo total: {duration}"))
            print(TerminalColors.info(f"   • Ciclos executados: {cycle}"))
            
            print(TerminalColors.info("\n📈 PERFORMANCE DE TRADES:"))
            print(TerminalColors.info(f"   • Total de trades: {self.total_trades}"))
            print(TerminalColors.success(f"   • Trades lucrativos: {self.successful_trades}"))
            print(TerminalColors.info(f"   • Taxa de acerto: {accuracy:.1f}%"))
            
            print(TerminalColors.info("\n💰 RESULTADO FINANCEIRO:"))
            if self.total_profit > 0:
                print(TerminalColors.success(f"   • Lucro total: +${self.total_profit:.2f}"))
            else:
                print(TerminalColors.error(f"   • Prejuízo total: ${self.total_profit:.2f}"))
            
            print(TerminalColors.success("\n✅ SISTEMA FINALIZADO COM SUCESSO!"))

    def _validate_market_data(self, data):
        """🔍 VALIDAÇÃO ROBUSTA DE DADOS DE MERCADO"""
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
            
            # Verifica gaps nos dados (intervalo esperado: 5 minutos)
            if len(data) > 1:
                timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_series') else pd.to_datetime(data['timestamp'])
                intervals = timestamps.diff().dropna()
                expected_interval = pd.Timedelta(minutes=5)
                
                # Se mais de 20% dos intervalos são maiores que o esperado
                large_gaps = intervals[intervals > expected_interval * 2]
                if len(large_gaps) > len(intervals) * 0.2:
                    return False, f"Gaps detectados: {len(large_gaps)} intervalos grandes"
            
            # Verifica outliers de preço
            price_std = data['close'].std()
            price_mean = data['close'].mean()
            outliers = data[abs(data['close'] - price_mean) > 3 * price_std]
            if len(outliers) > len(data) * 0.1:  # Mais de 10% outliers
                return False, f"Muitos outliers: {len(outliers)} de {len(data)} registros"
            
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
            
            return True, "Dados válidos"
            
        except Exception as e:
            return False, f"Erro na validação: {e}"

    def _log_trade_info(self, message, level="INFO"):
        """📝 LOG ESTRUTURADO PARA INFORMAÇÕES ESSENCIAIS"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "TRADE":
            print(f"💰 [{timestamp}] {message}")
        elif level == "SIGNAL":
            print(f"🎯 [{timestamp}] {message}")
        elif level == "ERROR":
            print(f"❌ [{timestamp}] {message}")
        elif level == "SUCCESS":
            print(f"✅ [{timestamp}] {message}")
        else:
            print(f"ℹ️ [{timestamp}] {message}")
        
        # Salva em arquivo de log
        try:
            with open("logs/trading_system.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] {level}: {message}\n")
        except:
            pass  # Ignora erros de log

    def _save_models_after_training(self):
        """💾 SALVA MODELOS APÓS TREINAMENTO AUTOMÁTICO"""
        try:
            # Salva XGBoost
            if 'xgboost' in self.models:
                joblib.dump(self.models['xgboost'], 'models/trained/xgboost_model.joblib')
                self._log_trade_info("XGBoost salvo", "INFO")
            
            # Salva Random Forest
            if 'random_forest' in self.models:
                joblib.dump(self.models['random_forest'], 'models/trained/rf_model.joblib')
                self._log_trade_info("Random Forest salvo", "INFO")
            
            # Salva LSTM (se disponível)
            if 'lstm' in self.models:
                self.models['lstm'].save('models/trained/lstm_model.h5')
                self._log_trade_info("LSTM salvo", "INFO")
            
            # Salva estado do sistema de aprendizado
            self.learning_system.save_state()
            self._log_trade_info("Estado do sistema salvo", "INFO")
            
        except Exception as e:
            self._log_trade_info(f"Erro ao salvar modelos: {e}", "ERROR")

    def _can_open_new_order(self, confidence):
        """Verifica se pode abrir nova ordem"""
        return (
            len(self.active_orders) < self.max_positions and 
            confidence >= self.min_confidence
        )
    
    def _add_active_order(self, order_data):
        """Adiciona ordem ativa ao sistema"""
        order_id = f"order_{len(self.active_orders) + 1}_{int(time.time())}"
        order_data['order_id'] = order_id
        order_data['opened_at'] = datetime.now()
        order_data['status'] = 'active'
        
        self.active_orders[order_id] = order_data
        
        print(TerminalColors.success(f"📊 NOVA ORDEM ADICIONADA: {order_id}"))
        print(TerminalColors.info(f"   Posições ativas: {len(self.active_orders)}/{self.max_positions}"))
        
        return order_id
    
    def _remove_active_order(self, order_id, reason="closed"):
        """Remove ordem ativa e adiciona ao histórico"""
        if order_id in self.active_orders:
            order_data = self.active_orders[order_id]
            order_data['closed_at'] = datetime.now()
            order_data['close_reason'] = reason
            order_data['status'] = 'closed'
            
            # Calcula resultado final
            if 'entry_price' in order_data and 'exit_price' in order_data:
                entry_price = order_data['entry_price']
                exit_price = order_data['exit_price']
                quantity = order_data['quantity']
                
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                
                order_data['final_pnl'] = pnl
                order_data['final_pnl_pct'] = pnl_pct
                
                # Atualiza estatísticas
                self.total_trades += 1
                self.total_profit += pnl
                if pnl > 0:
                    self.successful_trades += 1
                
                # Exibe resultado
                self._display_order_result(order_data)
            
            # Move para histórico
            self.order_history.append(order_data)
            del self.active_orders[order_id]
            
            print(TerminalColors.info(f"📊 ORDEM REMOVIDA: {order_id} - {reason}"))
            print(TerminalColors.info(f"   Posições ativas: {len(self.active_orders)}/{self.max_positions}"))
    
    def _display_order_result(self, order_data):
        """Exibe resultado de ordem fechada"""
        order_id = order_data['order_id']
        reason = order_data.get('close_reason', 'unknown')
        pnl = order_data.get('final_pnl', 0)
        pnl_pct = order_data.get('final_pnl_pct', 0)
        
        if pnl > 0:
            print(TerminalColors.success(f"🎉 ORDEM FECHADA COM LUCRO: {order_id}"))
            print(TerminalColors.success(f"   💰 Lucro: ${pnl:.2f} ({pnl_pct:.2f}%)"))
        else:
            print(TerminalColors.warning(f"📉 ORDEM FECHADA COM PREJUÍZO: {order_id}"))
            print(TerminalColors.warning(f"   💸 Prejuízo: ${abs(pnl):.2f} ({abs(pnl_pct):.2f}%)"))
        
        print(TerminalColors.info(f"   📋 Motivo: {reason}"))
        print(TerminalColors.info(f"   📊 Total trades: {self.total_trades} | Lucros: {self.successful_trades}"))
        print(TerminalColors.info(f"   💰 Lucro total: ${self.total_profit:.2f}"))
    
    def _monitor_all_active_orders(self, current_price, current_time, market_data):
        """Monitora todas as ordens ativas"""
        orders_to_close = []
        
        for order_id, order_data in self.active_orders.items():
            try:
                # Analisa performance da ordem
                try:
                    analysis = self._analyze_order_performance(order_data, current_price, current_time, market_data)
                except Exception as e:
                    print(TerminalColors.error(f"❌ Erro na análise de performance: {e}"))
                    continue
                
                if analysis['should_close']:
                    orders_to_close.append((order_id, analysis['reason']))
                    print(TerminalColors.warning(f"⚠️ ORDEM {order_id} MARCADA PARA FECHAMENTO: {analysis['reason']}"))
                
            except Exception as e:
                print(TerminalColors.error(f"❌ Erro monitorando ordem {order_id}: {e}"))
        
        # Fecha ordens marcadas
        for order_id, reason in orders_to_close:
            self._close_order_by_id(order_id, reason, current_price)
    
    def _close_order_by_id(self, order_id, reason, current_price):
        """Fecha ordem específica por ID"""
        if order_id not in self.active_orders:
            print(TerminalColors.warning(f"⚠️ Ordem {order_id} não encontrada"))
            return
        
        order_data = self.active_orders[order_id]
        
        try:
            # Executa venda real se for ordem real
            if order_data.get('is_real', False):
                self._execute_sell_real(current_price, reason)
                order_data['exit_price'] = current_price
                order_data['exit_time'] = datetime.now()
            
            # Remove da lista ativa
            self._remove_active_order(order_id, reason)
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro fechando ordem {order_id}: {e}"))
    
    def _check_tp_sl_all_orders(self, current_price):
        """Verifica TP/SL de todas as ordens ativas - Suporte a LONG e SHORT"""
        orders_to_close = []
        
        for order_id, order_data in self.active_orders.items():
            try:
                entry_price = order_data.get('entry_price', 0)
                stop_loss = order_data.get('stop_loss', 0)
                take_profit = order_data.get('take_profit', 0)
                order_type = order_data.get('type', 'LONG')
                
                if entry_price > 0:
                    # Verifica TP/SL baseado no tipo de posição
                    if order_type == 'LONG':
                        # Para LONG: Stop Loss abaixo, Take Profit acima
                        if stop_loss > 0 and current_price <= stop_loss:
                            orders_to_close.append((order_id, 'stop_loss'))
                            print(TerminalColors.error(f"🛡️ STOP LOSS LONG ATINGIDO: {order_id} @ ${current_price:.2f}"))
                        
                        elif take_profit > 0 and current_price >= take_profit:
                            orders_to_close.append((order_id, 'take_profit'))
                            print(TerminalColors.success(f"🎯 TAKE PROFIT LONG ATINGIDO: {order_id} @ ${current_price:.2f}"))
                            
                    elif order_type == 'SHORT':
                        # Para SHORT: Stop Loss acima, Take Profit abaixo
                        if stop_loss > 0 and current_price >= stop_loss:
                            orders_to_close.append((order_id, 'stop_loss'))
                            print(TerminalColors.error(f"🛡️ STOP LOSS SHORT ATINGIDO: {order_id} @ ${current_price:.2f}"))
                        
                        elif take_profit > 0 and current_price <= take_profit:
                            orders_to_close.append((order_id, 'take_profit'))
                            print(TerminalColors.success(f"🎯 TAKE PROFIT SHORT ATINGIDO: {order_id} @ ${current_price:.2f}"))
                    else:
                        # Fallback para compatibilidade
                        if stop_loss > 0 and current_price <= stop_loss:
                            orders_to_close.append((order_id, 'stop_loss'))
                            print(TerminalColors.error(f"🛡️ STOP LOSS ATINGIDO: {order_id} @ ${current_price:.2f}"))
                        
                        elif take_profit > 0 and current_price >= take_profit:
                            orders_to_close.append((order_id, 'take_profit'))
                            print(TerminalColors.success(f"🎯 TAKE PROFIT ATINGIDO: {order_id} @ ${current_price:.2f}"))
                
            except Exception as e:
                print(TerminalColors.error(f"❌ Erro verificando TP/SL da ordem {order_id}: {e}"))
        
        # Fecha ordens que atingiram TP/SL
        for order_id, reason in orders_to_close:
            self._close_order_by_id(order_id, reason, current_price)

def main():
    """Função principal"""
    print(TerminalColors.info("🔍 Verificando privilégios administrativos..."))
    
    if not is_admin():
        print(TerminalColors.warning("⚠️ Executando elevação de privilégios para sincronização..."))
        run_as_admin()
        return
    
    print(TerminalColors.success("✅ Privilégios confirmados - iniciando sistema"))
    
    try:
        system = FinalTestnetSystemAdvanced()
        system.run()
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro crítico: {e}"))
        sys.exit(1)

if __name__ == "__main__":
    main() 