"""
Configurações do Sistema de Trading
===================================

Configurações para Binance testnet e outros parâmetros do sistema.
"""

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações da Binance Testnet
BINANCE_TESTNET_CONFIG = {
    'base_url': 'https://testnet.binance.vision',
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'testnet': True
}

# Configurações de Trading
TRADING_CONFIG = {
    'symbol': 'BTCUSDT',
    'default_balance': 1000.0,  # Saldo padrão da testnet
    'min_order_size': 0.001,    # Tamanho mínimo da ordem BTC
    'max_order_size': 0.1,      # Tamanho máximo da ordem BTC
    'default_quantity': 0.01    # Quantidade padrão para trades
}

# Configurações de Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'file_logging': True,
    'console_logging': True,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Configurações de Performance
PERFORMANCE_CONFIG = {
    'accuracy_threshold': 0.80,  # 80%
    'profit_target': 0.02,       # 2%
    'stop_loss': 0.01,           # 1%
    'max_trades_per_day': 10,
    'min_confidence': 0.60       # 60%
}

def get_testnet_credentials():
    """Retorna credenciais da testnet se configuradas"""
    api_key = BINANCE_TESTNET_CONFIG['api_key']
    api_secret = BINANCE_TESTNET_CONFIG['api_secret']
    
    if api_key and api_secret:
        return {
            'api_key': api_key,
            'api_secret': api_secret,
            'testnet': True
        }
    else:
        return None

def is_testnet_configured():
    """Verifica se as credenciais da testnet estão configuradas"""
    return BINANCE_TESTNET_CONFIG['api_key'] is not None and BINANCE_TESTNET_CONFIG['api_secret'] is not None 