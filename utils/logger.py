"""
Sistema de Logging Configurado
Logger padrão com níveis, timestamps e rotação de arquivos

Funcionalidades:
- Logging para console e arquivo
- Rotação automática de logs
- Formatação padronizada
- Diferentes níveis por componente
- Cores e filtros para melhor visualização
"""

# ===== CONFIGURAÇÃO GLOBAL DE LOGS =====
import os
import logging

# Configura variáveis de ambiente para suprimir logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configura loggers globais para mostrar apenas ERROS
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('data-collector').setLevel(logging.WARNING)
logging.getLogger('news-agent').setLevel(logging.WARNING)
logging.getLogger('prediction-agent').setLevel(logging.WARNING)
logging.getLogger('vision-agent').setLevel(logging.WARNING)
logging.getLogger('decision-agent').setLevel(logging.WARNING)
logging.getLogger('continuous-learning').setLevel(logging.WARNING)
logging.getLogger('advanced-strategies-2025').setLevel(logging.WARNING)

# ===== IMPORTS =====
import logging.handlers
from datetime import datetime
import sys

# Códigos de cores ANSI
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class TradingFormatter(logging.Formatter):
    """Formatter com cores e filtros para trading"""
    
    def __init__(self):
        super().__init__()
        self.last_news_time = 0
        self.last_data_time = 0
        
    def format(self, record):
        # Formatação base
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Filtra spam repetitivo
        if self._should_filter(record):
            return ""  # Retorna string vazia em vez de None
            
        # Cores por tipo de mensagem
        color = self._get_color(record)
        
        # Formatação específica por tipo
        if "COMPRA" in record.msg or "BUY" in record.msg:
            return f"{color}{Colors.BOLD}🟢 {timestamp} | {record.msg}{Colors.RESET}"
        elif "VENDA" in record.msg or "SELL" in record.msg:
            return f"{color}{Colors.BOLD}🔴 {timestamp} | {record.msg}{Colors.RESET}"
        elif "LUCRO" in record.msg or "PROFIT" in record.msg:
            return f"{Colors.GREEN}{Colors.BOLD}💰 {timestamp} | {record.msg}{Colors.RESET}"
        elif "PREJUÍZO" in record.msg or "LOSS" in record.msg:
            return f"{Colors.RED}{Colors.BOLD}📉 {timestamp} | {record.msg}{Colors.RESET}"
        elif "HOLD" in record.msg:
            return f"{Colors.YELLOW}⏸️  {timestamp} | {record.msg}{Colors.RESET}"
        elif "DECISÃO" in record.msg or "DECISION" in record.msg:
            return f"{Colors.MAGENTA}{Colors.BOLD}🧠 {timestamp} | {record.msg}{Colors.RESET}"
        elif "PROGRESSO" in record.msg or "PROGRESS" in record.msg:
            return f"{Colors.CYAN}{Colors.BOLD}📊 {timestamp} | {record.msg}{Colors.RESET}"
        elif record.levelname == "ERROR":
            return f"{Colors.RED}{Colors.BOLD}❌ {timestamp} | {record.msg}{Colors.RESET}"
        elif record.levelname == "WARNING":
            return f"{Colors.YELLOW}[WARNING] {timestamp} | {record.msg}{Colors.RESET}"
        else:
            return f"{color}{timestamp} | {record.msg}{Colors.RESET}"
    
    def _should_filter(self, record):
        """Filtra mensagens repetitivas"""
        current_time = record.created
        
        # Filtra coleta de notícias (máximo 1 por 2 minutos)
        if any(x in record.msg for x in ["Coletando notícias", "RSS", "feedparser", "[UP]", "[NEWS2]"]):
            if current_time - self.last_news_time < 120:  # 2 minutos
                return True
            self.last_news_time = current_time
            # Substitui por mensagem resumida menos frequente
            record.msg = "📰 Análise de notícias atualizada"
            
        # Filtra coleta de dados (máximo 1 por minuto)
        if any(x in record.msg for x in ["Coletando dados", "market data", "API call", "dados mercado"]):
            if current_time - self.last_data_time < 60:  # 1 minuto
                return True
            self.last_data_time = current_time
            record.msg = "Dados de mercado atualizados"  # Removido emoji
        
        # Filtra mensagens muito repetitivas completamente
        repetitive_patterns = [
            "Calculando sentimento", "análise completa", "Score de visão",
            "Força do sinal", "Analisando sinais", "Estado carregado",
            "inicializado", "carregado", "adicionado"
        ]
        
        if any(pattern in record.msg for pattern in repetitive_patterns):
            return True
            
        return False
    
    def _get_color(self, record):
        """Retorna cor baseada no logger name"""
        if "trading" in record.name or "main" in record.name:
            return Colors.CYAN
        elif "decision" in record.name:
            return Colors.MAGENTA
        elif "news" in record.name:
            return Colors.BLUE
        elif "vision" in record.name:
            return Colors.GREEN
        elif "prediction" in record.name:
            return Colors.YELLOW
        else:
            return Colors.WHITE

class TradingFilter(logging.Filter):
    """Filtro para logs de trading"""
    
    def filter(self, record):
        # Sempre mostra logs críticos
        if record.levelname in ["ERROR", "WARNING"]:
            return True
            
        # Filtra debug excessivo
        if record.levelname == "DEBUG":
            return False
            
        # Logs importantes de trading (sempre mostrar)
        critical_keywords = [
            "COMPRA", "VENDA", "BUY", "SELL", "LUCRO", "PREJUÍZO", 
            "PROFIT", "LOSS", "TRADE", "ORDEM", "APROVADO", "APPROVED",
            "INICIANDO SISTEMA", "RELATÓRIO FINAL", "Sistema finalizado"
        ]
        
        if any(keyword in record.msg.upper() for keyword in critical_keywords):
            return True
            
        # Logs importantes de decisão (filtrar alguns)
        decision_keywords = ["DECISÃO", "DECISION", "TARGET"]
        if any(keyword in record.msg for keyword in decision_keywords):
            # Só mostra se não for HOLD
            if "HOLD" not in record.msg:
                return True
            else:
                return False  # Filtra HOLD
        
        # Logs de progresso/setup importantes
        setup_keywords = [
            "Configurações de TREINAMENTO", "MODO TREINAMENTO", "carregado",
            "SISTEMA PRONTO", "BTC:", "Saldo:", "CICLO #", "Duração:", "Total trades:"
        ]
        
        if any(keyword in record.msg for keyword in setup_keywords):
            return True
            
        # Filtra completamente logs técnicos
        spam_keywords = [
            "[ANALYZE]", "[DATA]", "[SIGNAL]", "[OK]", "[BRAIN]", "[PREDICT]",
            "[NEWS]", "[EYE]", "[UP]", "[NOTE]", "[TARGET]",
            "Score de", "Força do sinal", "Executando análise", "Inicializando",
            "Analisando sinais", "Calculando sentimento", "Previsão:", "carregado",
            "inicializado", "Estado carregado", "Modelo", "adicionado", "Sub-agentes",
            "FinBERT", "TextBlob", "VisionAgent", "NewsAgent", "DecisionAgent",
            "PredictionAgent", "Sistema de Aprendizado", "COMPONENTES CARREGADOS",
            "CONFIGURAÇÕES DE TRADING", "PESOS DOS AGENTES"
        ]
        
        if any(keyword in record.msg for keyword in spam_keywords):
            return False
            
        return True

# Configuração global de logging para reduzir poluição
import logging
import os
import sys
from datetime import datetime

# Suprime logs de bibliotecas externas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = ''    # Força CPU
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('binance').setLevel(logging.ERROR)

# Filtro para eliminar logs duplicados e desnecessários
class CleanLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen_messages = set()
        self.spam_patterns = [
            'INFO:data-collector:',
            'INFO:news-agent:',
            'INFO:prediction-agent:',
            'INFO:vision-agent:',
            'INFO:decision-agent:',
            'INFO:btc-trading-system:',
            'INFO:continuous-learning:',
            'INFO:advanced-strategies-2025:',
            '[FORCE] Forçando atualização',
            '[SUCCESS] Dados atualizados',
            '[DATA] Coletando dados',
            '[DATA] Dados obtidos',
            '[BALANCE] Arquivo .env carregado',
            '[BALANCE] Credenciais da testnet encontradas',
            '[BALANCE] Saldo USDT REAL',
            '[NEWS2] Coletando notícias',
            '[DATA] Coletadas',
            '[UP] Calculando sentimento',
            '[SUCCESS] Noticias atualizadas',
            'Device set to use cpu',
            '✅ ✅ Ordem executada',
            '✅ ✅ Ordem executada via API',
            '✅ ✅ SHORT aberto',
            '✅ ✅ LONG aberto',
            'ℹ️ 📊 MONITORANDO',
            'ℹ️ 🔍 Analisando ordem',
            'ℹ️    📋',
            'ℹ️       💰 Entry:',
            'ℹ️       ⏱️ Tempo:',
            '✅ ✅ Nenhuma ordem marcada',
            'ℹ️ 🔍 MONITORAMENTO CONTÍNUO',
            'ℹ️ 📊 RESUMO DO PORTFÓLIO',
            'ℹ️    📋 Ordens ativas:',
            'ℹ️    📈 Win rate:',
            'ℹ️    💰 P&L total:',
            'ℹ️    🚨 Alertas:',
            'ℹ️ 📊',
            '💰 📉 EXECUTANDO SHORT:',
            '💰 📊 TRADE REGISTRADO:',
            'ℹ️ 💾 Histórico salvo:',
            '✅ ✅ Ordem',
            '✅ ✅ Modelos salvos',
            '✅ Dados de performance salvos',
            '✅ Relatório de performance gerado',
            '💾 Métricas salvas',
            '💾 Estado do sistema salvo'
        ]
    
    def filter(self, record):
        # Elimina mensagens duplicadas
        message = record.getMessage()
        if message in self.seen_messages:
            return False
        
        # Elimina padrões de spam
        for pattern in self.spam_patterns:
            if pattern in message:
                return False
        
        # Mantém apenas logs essenciais
        essential_patterns = [
            '🚀 SISTEMA MODULARIZADO',
            '✅ Sistema modularizado inicializado',
            '🔄 CICLO',
            '💰 Preço:',
            '📊 Sinal:',
            '🛡️ Risco:',
            '📋 Ordens ativas:',
            '📉 Executando SHORT:',
            '🚀 Executando LONG:',
            '❌ Erro na execução',
            '⚠️ Limite de risco',
            '🎯 LIMITE DE',
            '✅ Sistema finalizando',
            '📊 RELATÓRIO FINAL',
            '⏱️ DURAÇÃO E CICLOS:',
            '📈 PERFORMANCE DE TRADES:',
            '💰 RESULTADO FINANCEIRO:',
            '✅ SISTEMA MODULARIZADO FINALIZADO'
        ]
        
        keep_log = False
        for pattern in essential_patterns:
            if pattern in message:
                keep_log = True
                break
        
        if keep_log:
            self.seen_messages.add(message)
            return True
        
        return False

# Configuração do logger principal
def setup_clean_logging():
    """Configura logging limpo e organizado"""
    # Remove handlers existentes
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configura handler limpo
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.addFilter(CleanLogFilter())
    
    # Formato limpo
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    # Aplica configuração
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Suprime logs de bibliotecas específicas
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('binance').setLevel(logging.ERROR)

# Função para log de trades (mantida limpa)
def log_trade_info(message, level='INFO'):
    """Função de log para informações de trading"""
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
        print(f"[INFO] [{timestamp}] {message}")

def setup_logger(name, level=logging.INFO):
    """Configura logger com filtros e formatação"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Aplica formatter e filtros
    formatter = TradingFormatter()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(TradingFilter())
    console_handler.addFilter(CleanLogFilter())
    
    logger.addHandler(console_handler)
    
    # Handler para arquivo (opcional)
    try:
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            f'logs/{name}.log',
            maxBytes=1024*1024,  # 1MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ Erro configurando log de arquivo: {e}")
    
    return logger

# Inicializa logging limpo
setup_clean_logging() 

def setup_trading_logger(log_file='trading.log'):
    logger = logging.getLogger('trading')
    logger.setLevel(logging.INFO)
    # Evita múltiplos handlers
    if not logger.handlers:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger 