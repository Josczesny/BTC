 # executors/exchange_api.py
"""
Interface com APIs de Exchanges
Integração com Binance e outras exchanges para execução de ordens

Funcionalidades:
- Conexão segura com APIs
- Execução de ordens (market/limit)
- Consulta de saldo e posições
- Gestão de rate limits
"""

import hashlib
import hmac
import time
import requests
from datetime import datetime
import os
from utils.logger import setup_logger

logger = setup_logger("exchange-api")

class ExchangeAPI:
    def __init__(self, exchange="binance"):
        """
        Inicializa conexão com exchange
        
        Args:
            exchange (str): Nome da exchange ("binance", "bybit", etc.)
        """
        logger.info(f"[CONN] Inicializando conexão com {exchange.upper()}")
        
        self.exchange = exchange.lower()
        
        # Configurações da API
        self.api_key = os.getenv("EXCHANGE_API_KEY")
        self.api_secret = os.getenv("EXCHANGE_API_SECRET")
        
        # URLs base por exchange
        self.base_urls = {
            "binance": "https://api.binance.com",
            "binance_testnet": "https://testnet.binance.vision",
            "bybit": "https://api.bybit.com",
            "coinbase": "https://api.exchange.coinbase.com"
        }
        
        # Seleciona URL base
        if exchange in self.base_urls:
            self.base_url = self.base_urls[exchange]
        else:
            self.base_url = self.base_urls["binance_testnet"]  # Fallback para testnet
            logger.warning(f"[WARN]  Exchange {exchange} não suportada, usando Binance testnet")
        
        # Configurações de rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms entre requests
        
        # Sincronização de timestamp
        self.time_offset = 0
        self._calibrate_time_offset()
        
        # Testa conexão
        self._test_connection()
        
        logger.info("[OK] ExchangeAPI inicializada")

    def _calibrate_time_offset(self):
        """
        Calibra offset de tempo com servidor Binance
        """
        try:
            local_time_before = int(time.time() * 1000)
            response = requests.get(f"{self.base_url}/api/v3/time", timeout=5)
            local_time_after = int(time.time() * 1000)
            
            if response.status_code == 200:
                server_time = response.json()['serverTime']
                local_time_avg = (local_time_before + local_time_after) // 2
                
                # Calcula offset (servidor - local)
                self.time_offset = server_time - local_time_avg
                
                logger.info(f"[SYNC] Offset calculado: {self.time_offset}ms")
            else:
                self.time_offset = 0
                logger.warning("[WARN] Falha na calibração de tempo")
                
        except Exception as e:
            self.time_offset = 0
            logger.warning(f"[WARN] Erro na calibração: {e}")

    def _test_connection(self):
        """
        Testa conectividade com a exchange
        """
        try:
            if self.exchange == "binance":
                response = requests.get(f"{self.base_url}/api/v3/time", timeout=5)
                if response.status_code == 200:
                    logger.info("[OK] Conexão com Binance estabelecida")
                else:
                    logger.error("[ERROR] Falha na conexão com Binance")
            else:
                logger.warning("[WARN]  Teste de conexão não implementado para esta exchange")
                
        except Exception as e:
            logger.error(f"[ERROR] Erro no teste de conexão: {e}")

    def _rate_limit_wait(self):
        """
        Controla rate limiting entre requests
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _sign_request(self, params):
        """
        Assina requisição para autenticação (Binance)
        
        Args:
            params (dict): Parâmetros da requisição
            
        Returns:
            dict: Parâmetros com assinatura
        """
        if not self.api_secret:
            return params
        
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        params["signature"] = signature
        return params

    def _get_server_time(self):
        """
        Obtém timestamp sincronizado com servidor
        
        Returns:
            int: Timestamp sincronizado
        """
        try:
            # Usa timestamp local + offset calibrado
            local_time = int(time.time() * 1000)
            server_time = local_time + self.time_offset
            
            # Subtrai pequena margem de segurança (200ms)
            return server_time - 200
            
        except:
            # Fallback para tempo local
            return int(time.time() * 1000)

    def _make_request(self, method, endpoint, params=None, signed=False):
        """
        Faz requisição para a API da exchange
        
        Args:
            method (str): Método HTTP
            endpoint (str): Endpoint da API
            params (dict): Parâmetros da requisição
            signed (bool): Se requer assinatura
            
        Returns:
            dict: Resposta da API ou None se erro
        """
        self._rate_limit_wait()
        
        if params is None:
            params = {}
        
        # Adiciona timestamp para requests assinados
        if signed:
            # Sincroniza com servidor Binance
            server_time = self._get_server_time()
            params["timestamp"] = server_time
            params = self._sign_request(params)
        
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Erro na requisição para {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Erro inesperado na requisição: {e}")
            return None

    def get_current_price(self, symbol="BTCUSDT"):
        """
        Obtém preço atual do Bitcoin
        
        Args:
            symbol (str): Par de trading
            
        Returns:
            float: Preço atual ou None se erro
        """
        try:
            if self.exchange == "binance":
                endpoint = "/api/v3/ticker/price"
                params = {"symbol": symbol}
                
                response = self._make_request("GET", endpoint, params)
                
                if response and "price" in response:
                    price = float(response["price"])
                    logger.debug(f"[MONEY] Preço atual {symbol}: ${price:,.2f}")
                    return price
            
            logger.error("[ERROR] Falha ao obter preço atual")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter preço: {e}")
            return None

    def get_account_balance(self, asset="USDT"):
        """
        Método de compatibilidade para get_balance
        
        Args:
            asset (str): Ativo a consultar
            
        Returns:
            float: Saldo disponível
        """
        return self.get_balance(asset)

    def get_balance(self, asset="USDT"):
        """
        Obtém saldo da conta
        
        Args:
            asset (str): Ativo a consultar
            
        Returns:
            float: Saldo disponível ou None se erro
        """
        try:
            if not self.api_key or not self.api_secret:
                logger.warning("[WARN]  Credenciais não configuradas")
                return None
            
            if self.exchange == "binance":
                endpoint = "/api/v3/account"
                
                response = self._make_request("GET", endpoint, signed=True)
                
                if response and "balances" in response:
                    for balance in response["balances"]:
                        if balance["asset"] == asset:
                            free_balance = float(balance["free"])
                            logger.debug(f"💳 Saldo {asset}: {free_balance}")
                            return free_balance
            
            logger.error("[ERROR] Falha ao obter saldo")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter saldo: {e}")
            return None

    def place_order(self, side, quantity, symbol="BTCUSDT", order_type="MARKET"):
        """
        Executa ordem na exchange
        
        Args:
            side (str): "BUY" ou "SELL"
            quantity (float): Quantidade a negociar
            symbol (str): Par de trading
            order_type (str): Tipo da ordem
            
        Returns:
            dict: Dados da ordem executada ou None se erro
        """
        logger.info(f"[UP] Executando ordem {side} - {quantity} {symbol}")
        
        try:
            if not self.api_key or not self.api_secret:
                logger.error("[ERROR] Credenciais não configuradas")
                return None
            
            if self.exchange == "binance":
                endpoint = "/api/v3/order"
                
                params = {
                    "symbol": symbol,
                    "side": side.upper(),
                    "type": order_type.upper(),
                    "quantity": quantity
                }
                
                # Para ordens market, remove price
                if order_type.upper() == "MARKET":
                    params.pop("price", None)
                
                response = self._make_request("POST", endpoint, params, signed=True)
                
                if response and "orderId" in response:
                    order_data = {
                        "id": response["orderId"],
                        "symbol": response["symbol"],
                        "side": response["side"],
                        "quantity": float(response["executedQty"]),
                        "price": float(response.get("price", 0)),
                        "status": response["status"],
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"[OK] Ordem executada - ID: {order_data['id']}")
                    return order_data
            
            logger.error("[ERROR] Falha na execução da ordem")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na execução da ordem: {e}")
            return None

    def get_open_orders(self, symbol="BTCUSDT"):
        """
        Obtém ordens abertas
        
        Args:
            symbol (str): Par de trading
            
        Returns:
            list: Lista de ordens abertas
        """
        try:
            if not self.api_key or not self.api_secret:
                return []
            
            if self.exchange == "binance":
                endpoint = "/api/v3/openOrders"
                params = {"symbol": symbol}
                
                response = self._make_request("GET", endpoint, params, signed=True)
                
                if response:
                    open_orders = []
                    for order in response:
                        open_orders.append({
                            "id": order["orderId"],
                            "symbol": order["symbol"],
                            "side": order["side"],
                            "quantity": float(order["origQty"]),
                            "price": float(order["price"]),
                            "status": order["status"],
                            "type": order["type"]
                        })
                    
                    logger.debug(f"[REPORT] {len(open_orders)} ordens abertas")
                    return open_orders
            
            return []
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter ordens abertas: {e}")
            return []

    def cancel_order(self, order_id, symbol="BTCUSDT"):
        """
        Cancela ordem específica
        
        Args:
            order_id (str): ID da ordem
            symbol (str): Par de trading
            
        Returns:
            bool: True se cancelada com sucesso
        """
        try:
            if not self.api_key or not self.api_secret:
                return False
            
            if self.exchange == "binance":
                endpoint = "/api/v3/order"
                params = {
                    "symbol": symbol,
                    "orderId": order_id
                }
                
                response = self._make_request("DELETE", endpoint, params, signed=True)
                
                if response and response.get("status") == "CANCELED":
                    logger.info(f"[OK] Ordem {order_id} cancelada")
                    return True
            
            logger.error(f"[ERROR] Falha ao cancelar ordem {order_id}")
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao cancelar ordem: {e}")
            return False

    def get_order_history(self, symbol="BTCUSDT", limit=100):
        """
        Obtém histórico de ordens
        
        Args:
            symbol (str): Par de trading
            limit (int): Número máximo de ordens
            
        Returns:
            list: Histórico de ordens
        """
        try:
            if not self.api_key or not self.api_secret:
                return []
            
            if self.exchange == "binance":
                endpoint = "/api/v3/allOrders"
                params = {
                    "symbol": symbol,
                    "limit": limit
                }
                
                response = self._make_request("GET", endpoint, params, signed=True)
                
                if response:
                    order_history = []
                    for order in response:
                        order_history.append({
                            "id": order["orderId"],
                            "symbol": order["symbol"],
                            "side": order["side"],
                            "quantity": float(order["origQty"]),
                            "executed_qty": float(order["executedQty"]),
                            "price": float(order["price"]),
                            "status": order["status"],
                            "type": order["type"],
                            "time": datetime.fromtimestamp(order["time"] / 1000)
                        })
                    
                    logger.debug(f"📜 Recuperadas {len(order_history)} ordens do histórico")
                    return order_history
            
            return []
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter histórico: {e}")
            return []

    def get_fake_balance(self):
        """
        Retorna balance fictício para desenvolvimento
        
        Returns:
            dict: Balance simulado
        """
        logger.info("[MONEY] Gerando balance fictício")
        
        return {
            'BTC': {
                'free': 0.5,
                'locked': 0.0,
                'total': 0.5
            },
            'USDT': {
                'free': 10000.0,
                'locked': 0.0,
                'total': 10000.0
            },
            'total_value_usd': 32500.0,
            'timestamp': datetime.now()
        }

    def get_fake_price(self, symbol='BTCUSDT'):
        """
        Retorna preço fictício realista
        
        Args:
            symbol (str): Símbolo da criptomoeda
            
        Returns:
            float: Preço fictício
        """
        logger.info(f"💲 Gerando preço fictício para {symbol}")
        
        # Preços base realistas
        base_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2800.0,
            'ADAUSDT': 0.45,
            'DOTUSDT': 6.50
        }
        
        base_price = base_prices.get(symbol, 45000.0)
        
        # Adiciona variação aleatória de ±2%
        import numpy as np
        variation = np.random.uniform(-0.02, 0.02)
        fake_price = base_price * (1 + variation)
        
        return round(fake_price, 2)

    def place_fake_order(self, order):
        """
        Simula execução de ordem para desenvolvimento
        
        Args:
            order (dict): Ordem a ser executada
            
        Returns:
            dict: Resultado da execução
        """
        logger.info(f"[NOTE] Simulando execução: {order.get('action')} {order.get('amount')} @ ${order.get('price', 0):,.2f}")
        
        try:
            # Simula latência da exchange
            import time
            import numpy as np
            time.sleep(0.1)
            
            # Simula sucesso em 95% dos casos
            success_rate = 0.95
            is_successful = np.random.random() < success_rate
            
            if is_successful:
                # Simula slippage pequeno
                slippage = np.random.uniform(-0.001, 0.001)  # 0.1% slippage
                executed_price = order.get('price', 45000) * (1 + slippage)
                
                result = {
                    'status': 'filled',
                    'order_id': f"fake_{int(datetime.now().timestamp())}",
                    'symbol': 'BTCUSDT',
                    'side': order.get('action', 'BUY').lower(),
                    'amount': order.get('amount', 0),
                    'price': order.get('price', 45000),
                    'executed_price': round(executed_price, 2),
                    'executed_amount': order.get('amount', 0),
                    'slippage': round(slippage * 100, 4),
                    'timestamp': datetime.now(),
                    'fees': round(order.get('amount', 0) * executed_price * 0.001, 6)  # 0.1% fee
                }
                
                logger.info(f"[OK] Ordem executada: slippage {slippage*100:.3f}%")
                
            else:
                result = {
                    'status': 'rejected',
                    'reason': 'Insufficient liquidity (simulated)',
                    'timestamp': datetime.now()
                }
                
                logger.warning("[WARN]  Ordem rejeitada (simulação)")
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na simulação de ordem: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': datetime.now()
            }