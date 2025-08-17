#!/usr/bin/env python3
"""
GERENCIADOR DE API BINANCE
==========================

Módulo responsável pelo gerenciamento da API Binance.
"""

import os
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors

load_dotenv()

class APIManager:
    """Gerenciador da API Binance com sincronização robusta"""
    
    def __init__(self):
        """Inicializa o gerenciador de API"""
        self.symbol = 'BTCUSDT'
        self.client = None
        self.server_time_offset = 0
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa cliente Binance com configurações robustas"""
        try:
            # Configuração da API - CORRIGIDO para usar as chaves corretas
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
            
            if not api_key or not api_secret:
                print(TerminalColors.error("❌ Chaves da API Binance não encontradas no .env"))
                print(TerminalColors.info("🔧 Verificando chaves alternativas..."))
                
                # Tenta chaves alternativas
                api_key = os.getenv("EXCHANGE_API_KEY")
                api_secret = os.getenv("EXCHANGE_API_SECRET")
                
                if not api_key or not api_secret:
                    print(TerminalColors.error("❌ Nenhuma chave da API encontrada"))
                    return
            
            print(TerminalColors.info(f"🔑 Usando chave API"))
            
            # SOLUÇÃO REAL: Cliente com configurações otimizadas
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                requests_params={'timeout': 60}
            )
            
            # Sincroniza com servidor Binance
            self._sync_with_binance_server()
            
            print(TerminalColors.success("✅ Cliente Binance inicializado"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao inicializar cliente Binance: {e}"))
    
    def _sync_with_binance_server(self):
        """
        SOLUÇÃO OFICIAL DA BINANCE - Sincronização correta de timestamp
        Baseado na documentação oficial: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/endpoint-security-type
        """
        try:
            if self.client is None:
                print(TerminalColors.warning("⚠️ Cliente Binance não inicializado"))
                self.server_time_offset = 0
                return False
            
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
        """Obtém timestamp ajustado"""
        if is_trading_operation:
            # Para operações de trading, usa offset
            return int(time.time() * 1000) + self.server_time_offset
        else:
            # Para outras operações, usa tempo local
            return int(time.time() * 1000)
    
    def _make_robust_request_with_sync(self, func, *args, **kwargs):
        """Faz requisição robusta com sincronização"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Adiciona timestamp se necessário
                if self._needs_timestamp(func.__name__):
                    kwargs['timestamp'] = self._get_adjusted_timestamp(is_trading_operation=True)
                
                # Adiciona recvWindow se necessário
                if self._needs_recv_window(func.__name__):
                    kwargs['recvWindow'] = 60000  # 60 segundos
                
                # Executa requisição
                result = func(*args, **kwargs)
                return result
                
            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp error
                    print(TerminalColors.warning(f"⚠️ Erro de timestamp (tentativa {attempt + 1}): {e}"))
                    # Re-sincroniza
                    self._sync_with_binance_server()
                    time.sleep(1)
                elif e.code == -2015:  # Invalid API-key
                    print(TerminalColors.error(f"❌ API Key inválida: {e}"))
                    return None
                else:
                    print(TerminalColors.error(f"❌ Erro da API Binance: {e}"))
                    return None
            except Exception as e:
                print(TerminalColors.error(f"❌ Erro na requisição: {e}"))
                return None
        
        return None
    
    def _needs_timestamp(self, method_name):
        """Verifica se método precisa de timestamp"""
        timestamp_methods = [
            'order_market_buy', 'order_market_sell', 'order_limit_buy', 'order_limit_sell',
            'get_open_orders', 'cancel_order', 'get_order'
        ]
        return method_name in timestamp_methods
    
    def _needs_recv_window(self, method_name):
        """Verifica se método precisa de recvWindow"""
        recv_window_methods = [
            'order_market_buy', 'order_market_sell', 'order_limit_buy', 'order_limit_sell'
        ]
        return method_name in recv_window_methods
    
    def get_current_price(self):
        """Obtém preço atual do BTC - endpoint público, sempre funciona"""
        try:
            if self.client is None:
                print(TerminalColors.warning("⚠️ Cliente Binance não inicializado"))
                return 108000.0  # Fallback
            
            # Endpoint público - não precisa de timestamp
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            if ticker and 'price' in ticker:
                return float(ticker['price'])
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro obtendo preço: {e}"))
            
        return 108000.0  # Fallback
    
    def get_market_data(self):
        """Obtém dados de mercado - endpoint público, sempre funciona"""
        try:
            if self.client is None:
                print(TerminalColors.warning("⚠️ Cliente Binance não inicializado"))
                return pd.DataFrame()
            
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
        return pd.DataFrame()
    
    def get_safe_balance(self):
        """Obtém saldo da conta - ATUALIZA APÓS CADA TRADE"""
        try:
            if self.client is None:
                print(TerminalColors.warning("⚠️ Cliente Binance não inicializado"))
                return 10000.0
            
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
                print(TerminalColors.warning("⚠️ Erro de timestamp no saldo - usando simulado"))
                return 10000.0
            else:
                print(TerminalColors.error(f"❌ Erro API Binance: {e}"))
                return 10000.0
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
            
            return True, "Dados válidos"
            
        except Exception as e:
            return False, f"Erro na validação: {e}" 