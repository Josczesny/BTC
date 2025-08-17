 # agents/order_agent.py
"""
Agente de Gerenciamento de Ordens
Controla ordens abertas, gestão de risco e execução de trades

Funcionalidades:
- Abertura e fechamento de ordens via API
- Monitoramento contínuo de posições
- Stop loss/take profit dinâmicos
- Gestão de tempo e risco
"""

from datetime import datetime, timedelta
from executors.exchange_api import ExchangeAPI
from monitor.performance_logger import PerformanceLogger
from utils.logger import setup_logger

logger = setup_logger("order-agent")

class OrderAgent:
    def __init__(self):
        """
        Inicializa o agente de ordens
        """
        logger.info("[REPORT] Inicializando OrderAgent")
        
        # Integração com exchange
        self.exchange = ExchangeAPI()
        self.performance_logger = PerformanceLogger()
        
        # Configurações de gestão de risco
        self.max_loss_per_trade = 0.02  # 2% máximo de perda por trade
        self.max_gain_target = 0.05     # 5% target de lucro
        self.max_trade_duration = 24    # 24 horas máximo por trade
        
        # Cache de ordens ativas
        self.active_orders = {}
        
        # Configurações de validação
        self.max_position_size = 1.0    # 100% do capital
        self.max_order_value = 50000.0  # $50k máximo por ordem
        
        logger.info("[OK] OrderAgent inicializado")

    def validate_order(self, order):
        """
        Valida uma ordem antes da execução
        
        Args:
            order (dict): Ordem a ser validada
            
        Returns:
            dict: Resultado da validação
        """
        logger.info(f"[VALIDATE] Validando ordem: {order.get('action', 'UNKNOWN')}")
        
        try:
            # Validações básicas
            required_fields = ['action', 'amount', 'price']
            for field in required_fields:
                if field not in order:
                    return {
                        'is_valid': False,
                        'reason': f'Campo obrigatório ausente: {field}'
                    }
            
            action = order['action']
            amount = order['amount']
            price = order['price']
            
            # Valida ação
            valid_actions = ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']
            if action not in valid_actions:
                return {
                    'is_valid': False,
                    'reason': f'Ação inválida: {action}. Permitidas: {valid_actions}'
                }
            
            # Valida quantidade
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {
                    'is_valid': False,
                    'reason': f'Quantidade inválida: {amount}'
                }
            
            if amount > self.max_position_size:
                return {
                    'is_valid': False,
                    'reason': f'Quantidade excede limite: {amount} > {self.max_position_size}'
                }
            
            # Valida preço
            if not isinstance(price, (int, float)) or price <= 0:
                return {
                    'is_valid': False,
                    'reason': f'Preço inválido: {price}'
                }
            
            # Valida limites de risco
            risk_amount = amount * price
            if risk_amount > self.max_order_value:
                return {
                    'is_valid': False,
                    'reason': f'Valor da ordem excede limite: ${risk_amount:,.2f} > ${self.max_order_value:,.2f}'
                }
            
            # Todas as validações passaram
            logger.info(f"[OK] Ordem válida: {action} {amount} @ ${price:,.2f}")
            
            return {
                'is_valid': True,
                'reason': 'Ordem passou em todas as validações',
                'risk_amount': risk_amount,
                'validated_order': {
                    'action': action,
                    'amount': round(amount, 6),
                    'price': round(price, 2),
                    'validation_timestamp': datetime.now()
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na validação da ordem: {e}")
            return {
                'is_valid': False,
                'reason': f'Erro interno na validação: {str(e)}'
            }

    def manage_open_orders(self, market_data):
        """
        Gerencia ordens abertas baseado em dados de mercado atualizados
        
        Args:
            market_data (pd.DataFrame): Dados de mercado atuais
        """
        logger.info("[REFRESH] Gerenciando ordens abertas")
        
        try:
            if not self.active_orders:
                logger.debug("📭 Nenhuma ordem aberta para gerenciar")
                return
            
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 45000.0
            current_time = datetime.now()
            
            orders_to_close = []
            
            for order_id, order_data in self.active_orders.items():
                # Analisa cada ordem aberta
                decision = self._analyze_order_for_closure(order_data, current_price, current_time, market_data)
                
                if decision['should_close']:
                    orders_to_close.append((order_id, decision['reason']))
                    logger.info(f"[SEND] Ordem {order_id} marcada para fechamento: {decision['reason']}")
            
            # Executa fechamentos
            for order_id, reason in orders_to_close:
                success = self.close_order(order_id, reason)
                if success:
                    logger.info(f"[OK] Ordem {order_id} fechada com sucesso")
                else:
                    logger.error(f"[ERROR] Falha ao fechar ordem {order_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no gerenciamento de ordens: {e}")

    def _analyze_order_for_closure(self, order_data, current_price, current_time, market_data):
        """
        Analisa se uma ordem deve ser fechada
        
        Args:
            order_data (dict): Dados da ordem
            current_price (float): Preço atual
            current_time (datetime): Tempo atual
            market_data (pd.DataFrame): Dados de mercado
            
        Returns:
            dict: Decisão de fechamento
        """
        try:
            entry_price = order_data.get('entry_price', current_price)
            side = order_data.get('side', 'buy')
            entry_time = order_data.get('timestamp', current_time)
            
            # Calcula P&L atual
            if side == 'buy':
                pnl_percentage = (current_price - entry_price) / entry_price
            else:
                pnl_percentage = (entry_price - current_price) / entry_price
            
            # Tempo decorrido
            time_elapsed = (current_time - entry_time).total_seconds() / 3600  # em horas
            
            # === CRITÉRIOS DE FECHAMENTO ===
            
            # 1. Stop Loss
            if pnl_percentage <= -self.max_loss_per_trade:
                return {'should_close': True, 'reason': 'stop_loss'}
            
            # 2. Take Profit
            if pnl_percentage >= self.max_gain_target:
                return {'should_close': True, 'reason': 'take_profit'}
            
            # 3. Tempo máximo
            if time_elapsed >= self.max_trade_duration:
                return {'should_close': True, 'reason': 'max_time'}
            
            # 4. Análise técnica adversa (se temos dados suficientes)
            if len(market_data) >= 5:
                recent_trend = self._analyze_recent_trend(market_data)
                if self._is_trend_against_position(recent_trend, side):
                    return {'should_close': True, 'reason': 'trend_reversal'}
            
            # 5. Volatilidade excessiva
            if len(market_data) >= 10:
                volatility = market_data['close'].pct_change().tail(10).std()
                if volatility > 0.05:  # 5% volatilidade
                    return {'should_close': True, 'reason': 'high_volatility'}
            
            return {'should_close': False, 'reason': 'holding'}
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de fechamento: {e}")
            return {'should_close': False, 'reason': 'analysis_error'}

    def _analyze_recent_trend(self, market_data):
        """
        Analisa tendência recente dos dados
        """
        try:
            if len(market_data) < 5:
                return 'neutral'
            
            recent_prices = market_data['close'].tail(5)
            first_price = recent_prices.iloc[0]
            last_price = recent_prices.iloc[-1]
            
            change = (last_price - first_price) / first_price
            
            if change > 0.01:  # 1% alta
                return 'bullish'
            elif change < -0.01:  # 1% baixa
                return 'bearish'
            else:
                return 'neutral'
            
        except Exception:
            return 'neutral'

    def _is_trend_against_position(self, trend, position_side):
        """
        Verifica se a tendência está contra a posição
        """
        if position_side == 'buy' and trend == 'bearish':
            return True
        elif position_side == 'sell' and trend == 'bullish':
            return True
        return False

    def open_order(self, side, size, reason="manual"):
        """
        Abre nova ordem no mercado
        
        Args:
            side (str): "buy" ou "sell"
            size (float): Tamanho da posição (% do capital)
            reason (str): Motivo da abertura
            
        Returns:
            dict: Dados da ordem aberta ou None se falhou
        """
        logger.info(f"[UP] Abrindo ordem {side.upper()} - Tamanho: {size:.2%}")
        
        try:
            # Calcula quantidade baseada no capital disponível
            quantity = self._calculate_quantity(size)
            
            if quantity <= 0:
                logger.warning("[WARN]  Quantidade inválida para ordem")
                return None
            
            # Coleta preço atual para referência
            current_price = self.exchange.get_current_price()
            
            # Executa ordem no exchange
            order_result = self.exchange.place_order(
                side=side,
                quantity=quantity,
                order_type="market"
            )
            
            if order_result and order_result.get("status") == "filled":
                # Registra ordem aberta
                order_data = {
                    "id": order_result["id"],
                    "side": side,
                    "quantity": quantity,
                    "entry_price": order_result["price"],
                    "timestamp": datetime.now(),
                    "reason": reason,
                    "stop_loss": self._calculate_stop_loss(order_result["price"], side),
                    "take_profit": self._calculate_take_profit(order_result["price"], side),
                    "status": "active"
                }
                
                self.active_orders[order_result["id"]] = order_data
                
                # Log da abertura
                self.performance_logger.log_order_open(order_data)
                
                logger.info(f"[OK] Ordem aberta - ID: {order_result['id']}")
                return order_data
            
            else:
                logger.error("[ERROR] Falha na execução da ordem")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Erro ao abrir ordem: {e}")
            return None

    def close_order(self, order_id, reason="manual"):
        """
        Fecha ordem específica
        
        Args:
            order_id (str): ID da ordem a fechar
            reason (str): Motivo do fechamento
            
        Returns:
            bool: True se fechou com sucesso
        """
        logger.info(f"[DOWN] Fechando ordem {order_id}")
        
        try:
            if order_id not in self.active_orders:
                logger.warning(f"[WARN]  Ordem {order_id} não encontrada")
                return False
            
            order_data = self.active_orders[order_id]
            
            # Determina lado oposto para fechamento
            close_side = "sell" if order_data["side"] == "buy" else "buy"
            
            # Executa ordem de fechamento
            close_result = self.exchange.place_order(
                side=close_side,
                quantity=order_data["quantity"],
                order_type="market"
            )
            
            if close_result and close_result.get("status") == "filled":
                # Atualiza dados da ordem
                order_data["exit_price"] = close_result["price"]
                order_data["exit_timestamp"] = datetime.now()
                order_data["close_reason"] = reason
                order_data["status"] = "closed"
                
                # Calcula P&L
                pnl = self._calculate_pnl(order_data)
                order_data["pnl"] = pnl
                order_data["pnl_percentage"] = pnl / order_data["entry_price"]
                
                # Log do fechamento
                self.performance_logger.log_order_close(order_data)
                
                # Remove da lista de ordens ativas
                del self.active_orders[order_id]
                
                logger.info(f"[OK] Ordem fechada - P&L: {pnl:.2f} ({order_data['pnl_percentage']:.2%})")
                return True
            
            else:
                logger.error("[ERROR] Falha no fechamento da ordem")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Erro ao fechar ordem: {e}")
            return False

    def evaluate_open_orders(self):
        """
        Avalia todas as ordens abertas para fechamento automático
        """
        if not self.active_orders:
            return
        
        logger.debug(f"[EVAL] Avaliando {len(self.active_orders)} ordens abertas")
        
        orders_to_close = []
        current_price = self.exchange.get_current_price()
        
        for order_id, order_data in self.active_orders.items():
            close_reason = self._should_close_order(order_data, current_price)
            
            if close_reason:
                orders_to_close.append((order_id, close_reason))
        
        # Fecha ordens identificadas
        for order_id, reason in orders_to_close:
            self.close_order(order_id, reason)

    def _should_close_order(self, order_data, current_price):
        """
        Determina se uma ordem deve ser fechada
        
        Args:
            order_data (dict): Dados da ordem
            current_price (float): Preço atual
            
        Returns:
            str: Motivo do fechamento ou None se deve manter
        """
        entry_price = order_data["entry_price"]
        side = order_data["side"]
        entry_time = order_data["timestamp"]
        
        # Calcula P&L atual
        if side == "buy":
            current_pnl = current_price - entry_price
            pnl_percentage = current_pnl / entry_price
        else:  # sell
            current_pnl = entry_price - current_price
            pnl_percentage = current_pnl / entry_price
        
        # Verifica stop loss
        if pnl_percentage <= -self.max_loss_per_trade:
            return "stop_loss"
        
        # Verifica take profit
        if pnl_percentage >= self.max_gain_target:
            return "take_profit"
        
        # Verifica tempo máximo
        time_elapsed = datetime.now() - entry_time
        if time_elapsed.total_seconds() / 3600 > self.max_trade_duration:
            return "max_time_reached"
        
        # TODO: Implementar critérios adicionais:
        # - Reversão de sinal dos agentes
        # - Volatilidade excessiva
        # - Condições de mercado adversas
        
        return None

    def _calculate_quantity(self, size_percentage):
        """
        Calcula quantidade baseada no capital disponível
        
        Args:
            size_percentage (float): Percentual do capital a usar
            
        Returns:
            float: Quantidade a negociar
        """
        try:
            # TODO: Implementar cálculo real baseado no saldo
            # TODO: Considerar margem, alavancagem, etc.
            
            available_balance = self.exchange.get_balance()
            current_price = self.exchange.get_current_price()
            
            if available_balance and current_price:
                capital_to_use = available_balance * size_percentage
                quantity = capital_to_use / current_price
                return quantity
            
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no cálculo de quantidade: {e}")
            return 0.0

    def _calculate_stop_loss(self, entry_price, side):
        """
        Calcula preço de stop loss
        """
        if side == "buy":
            return entry_price * (1 - self.max_loss_per_trade)
        else:  # sell
            return entry_price * (1 + self.max_loss_per_trade)

    def _calculate_take_profit(self, entry_price, side):
        """
        Calcula preço de take profit
        """
        if side == "buy":
            return entry_price * (1 + self.max_gain_target)
        else:  # sell
            return entry_price * (1 - self.max_gain_target)

    def _calculate_pnl(self, order_data):
        """
        Calcula P&L realizado da ordem
        """
        entry_price = order_data["entry_price"]
        exit_price = order_data["exit_price"]
        quantity = order_data["quantity"]
        side = order_data["side"]
        
        if side == "buy":
            return (exit_price - entry_price) * quantity
        else:  # sell
            return (entry_price - exit_price) * quantity

    def get_open_positions(self):
        """
        Retorna lista de posições abertas
        
        Returns:
            list: Lista de ordens ativas
        """
        return list(self.active_orders.values())

    def get_portfolio_status(self):
        """
        Retorna status geral do portfólio
        
        Returns:
            dict: Estatísticas do portfólio
        """
        try:
            open_positions = self.get_open_positions()
            current_price = self.exchange.get_current_price()
            
            total_unrealized_pnl = 0.0
            total_exposure = 0.0
            
            for order in open_positions:
                entry_price = order["entry_price"]
                quantity = order["quantity"]
                side = order["side"]
                
                # Calcula P&L não realizado
                if side == "buy":
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                total_unrealized_pnl += unrealized_pnl
                total_exposure += entry_price * quantity
            
            return {
                "open_positions": len(open_positions),
                "total_exposure": total_exposure,
                "unrealized_pnl": total_unrealized_pnl,
                "available_balance": self.exchange.get_balance(),
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter status do portfólio: {e}")
            return {}