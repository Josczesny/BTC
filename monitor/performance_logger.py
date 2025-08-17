"""
Logger de Performance e Métricas
Registra desempenho dos trades, lucro/prejuízo e estatísticas

Funcionalidades:
- Log detalhado de cada trade
- Cálculo de métricas de performance
- Relatórios de desempenho
- Alertas para drawdowns excessivos
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import json
from utils.logger import setup_logger

logger = setup_logger("performance-logger")

class PerformanceLogger:
    def __init__(self):
        """
        Inicializa o logger de performance
        """
        logger.info("[DATA] Inicializando PerformanceLogger")
        
        # Configurações do banco de dados
        self.db_path = "data/performance.db"
        self.ensure_database()
        
        # Configurações de alertas
        self.max_drawdown_alert = 0.10  # 10% de drawdown máximo
        self.min_win_rate_alert = 0.40   # 40% de win rate mínimo
        
        # Cache de métricas
        self.metrics_cache = {}
        self.cache_duration = 300  # 5 minutos
        
        logger.info("[OK] PerformanceLogger inicializado")

    def ensure_database(self):
        """
        Garante que o banco de dados existe com as tabelas necessárias
        """
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Tabela para registros de trades
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT UNIQUE,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        entry_price REAL,
                        exit_price REAL,
                        entry_timestamp INTEGER,
                        exit_timestamp INTEGER,
                        pnl REAL,
                        pnl_percentage REAL,
                        duration_hours REAL,
                        reason_open TEXT,
                        reason_close TEXT,
                        fees REAL,
                        agent_signals TEXT
                    )
                """)
                
                # Tabela para métricas diárias
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_metrics (
                        date TEXT PRIMARY KEY,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        total_pnl REAL,
                        total_fees REAL,
                        win_rate REAL,
                        avg_win REAL,
                        avg_loss REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        portfolio_value REAL
                    )
                """)
                
                # Tabela para logs de sistema
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER,
                        level TEXT,
                        component TEXT,
                        message TEXT,
                        data TEXT
                    )
                """)
                
                conn.commit()
                
            logger.info("[OK] Banco de performance configurado")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na configuração do banco: {e}")

    def log_order_open(self, order_data):
        """
        Registra abertura de ordem
        
        Args:
            order_data (dict): Dados da ordem aberta
        """
        try:
            logger.info(f"[UP] Registrando abertura de ordem {order_data['id']}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Serializa sinais dos agentes se disponível
                agent_signals = json.dumps(order_data.get("agent_signals", {}))
                
                conn.execute("""
                    INSERT OR REPLACE INTO trades 
                    (order_id, symbol, side, quantity, entry_price, entry_timestamp, 
                     reason_open, agent_signals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order_data["id"],
                    order_data.get("symbol", "BTCUSDT"),
                    order_data["side"],
                    order_data["quantity"],
                    order_data["entry_price"],
                    int(order_data["timestamp"].timestamp()),
                    order_data.get("reason", "unknown"),
                    agent_signals
                ))
                
                conn.commit()
                
            logger.debug("[OK] Abertura registrada com sucesso")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao registrar abertura: {e}")

    def log_order_close(self, order_data):
        """
        Registra fechamento de ordem e calcula métricas
        
        Args:
            order_data (dict): Dados da ordem fechada
        """
        try:
            logger.info(f"[DOWN] Registrando fechamento de ordem {order_data['id']}")
            
            # Calcula duração
            duration_hours = 0
            if "exit_timestamp" in order_data and "timestamp" in order_data:
                duration = order_data["exit_timestamp"] - order_data["timestamp"]
                duration_hours = duration.total_seconds() / 3600
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE trades SET
                        exit_price = ?,
                        exit_timestamp = ?,
                        pnl = ?,
                        pnl_percentage = ?,
                        duration_hours = ?,
                        reason_close = ?,
                        fees = ?
                    WHERE order_id = ?
                """, (
                    order_data["exit_price"],
                    int(order_data["exit_timestamp"].timestamp()),
                    order_data["pnl"],
                    order_data.get("pnl_percentage", 0),
                    duration_hours,
                    order_data.get("close_reason", "unknown"),
                    order_data.get("fees", 0),
                    order_data["id"]
                ))
                
                conn.commit()
            
            # Atualiza métricas diárias
            self._update_daily_metrics()
            
            # Verifica alertas
            self._check_alerts()
            
            logger.info(f"[OK] Fechamento registrado - P&L: {order_data['pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao registrar fechamento: {e}")

    def _update_daily_metrics(self):
        """
        Atualiza métricas agregadas do dia atual
        """
        try:
            today = datetime.now().date().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Calcula métricas do dia
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        SUM(COALESCE(fees, 0)) as total_fees,
                        AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
                    FROM trades 
                    WHERE date(exit_timestamp, 'unixepoch') = ?
                    AND exit_timestamp IS NOT NULL
                """, (today,))
                
                metrics = cursor.fetchone()
                
                if metrics and metrics[0] > 0:  # Se há trades no dia
                    total_trades = metrics[0]
                    winning_trades = metrics[1] or 0
                    losing_trades = metrics[2] or 0
                    total_pnl = metrics[3] or 0
                    total_fees = metrics[4] or 0
                    avg_win = metrics[5] or 0
                    avg_loss = metrics[6] or 0
                    
                    # Calcula win rate
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    # TODO: Calcular Sharpe ratio e max drawdown
                    sharpe_ratio = 0.0
                    max_drawdown = 0.0
                    portfolio_value = 0.0
                    
                    # Salva métricas diárias
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_metrics 
                        (date, total_trades, winning_trades, losing_trades, total_pnl,
                         total_fees, win_rate, avg_win, avg_loss, max_drawdown, 
                         sharpe_ratio, portfolio_value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        today, total_trades, winning_trades, losing_trades, total_pnl,
                        total_fees, win_rate, avg_win, avg_loss, max_drawdown,
                        sharpe_ratio, portfolio_value
                    ))
                    
                    conn.commit()
                    
                    logger.debug(f"[DATA] Métricas diárias atualizadas - P&L: {total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao atualizar métricas diárias: {e}")

    def _check_alerts(self):
        """
        Verifica condições para alertas de performance
        """
        try:
            current_metrics = self.get_current_performance()
            
            # Alerta de drawdown excessivo
            max_drawdown = current_metrics.get("max_drawdown", 0)
            if max_drawdown > self.max_drawdown_alert:
                self._send_alert("high_drawdown", {
                    "current_drawdown": max_drawdown,
                    "threshold": self.max_drawdown_alert
                })
            
            # Alerta de win rate baixo
            win_rate = current_metrics.get("win_rate", 0)
            if win_rate < self.min_win_rate_alert:
                self._send_alert("low_win_rate", {
                    "current_win_rate": win_rate,
                    "threshold": self.min_win_rate_alert
                })
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na verificação de alertas: {e}")

    def _send_alert(self, alert_type, data):
        """
        Envia alerta de performance
        
        Args:
            alert_type (str): Tipo do alerta
            data (dict): Dados do alerta
        """
        logger.warning(f"[ALERT] ALERTA {alert_type.upper()}: {data}")
        
        # TODO: Implementar notificações (email, Telegram, etc.)
        # TODO: Salvar alertas no banco para histórico

    def get_current_performance(self):
        """
        Obtém métricas de performance atuais
        
        Returns:
            dict: Métricas de performance
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Métricas gerais
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        SUM(COALESCE(fees, 0)) as total_fees,
                        AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
                        AVG(duration_hours) as avg_duration
                    FROM trades 
                    WHERE exit_timestamp IS NOT NULL
                """)
                
                metrics = cursor.fetchone()
                
                if metrics and metrics[0] > 0:
                    total_trades = metrics[0]
                    winning_trades = metrics[1] or 0
                    losing_trades = metrics[2] or 0
                    total_pnl = metrics[3] or 0
                    total_fees = metrics[4] or 0
                    avg_win = metrics[5] or 0
                    avg_loss = metrics[6] or 0
                    avg_duration = metrics[7] or 0
                    
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
                    
                    return {
                        "total_trades": total_trades,
                        "winning_trades": winning_trades,
                        "losing_trades": losing_trades,
                        "win_rate": win_rate,
                        "total_pnl": total_pnl,
                        "total_fees": total_fees,
                        "net_pnl": total_pnl - total_fees,
                        "avg_win": avg_win,
                        "avg_loss": avg_loss,
                        "profit_factor": profit_factor,
                        "avg_duration_hours": avg_duration,
                        "max_drawdown": 0.0,  # TODO: Calcular drawdown real
                        "sharpe_ratio": 0.0,   # TODO: Calcular Sharpe ratio
                        "updated_at": datetime.now()
                    }
                
                else:
                    return {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                        "net_pnl": 0.0,
                        "updated_at": datetime.now()
                    }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao obter performance: {e}")
            return {}

    def generate_summary(self, days=30):
        """
        Gera relatório de performance dos últimos N dias
        
        Args:
            days (int): Número de dias para incluir no relatório
            
        Returns:
            dict: Relatório de performance
        """
        logger.info(f"[REPORT] Gerando relatório dos últimos {days} dias")
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).timestamp()
            
            with sqlite3.connect(self.db_path) as conn:
                # Trades no período
                df_trades = pd.read_sql_query("""
                    SELECT * FROM trades 
                    WHERE exit_timestamp >= ? 
                    AND exit_timestamp IS NOT NULL
                    ORDER BY exit_timestamp
                """, conn, params=(cutoff_date,))
                
                if df_trades.empty:
                    return {
                        "period_days": days,
                        "total_trades": 0,
                        "message": "Nenhum trade encontrado no período"
                    }
                
                # Calcula métricas do período
                total_trades = len(df_trades)
                winning_trades = len(df_trades[df_trades['pnl'] > 0])
                losing_trades = len(df_trades[df_trades['pnl'] < 0])
                
                total_pnl = df_trades['pnl'].sum()
                total_fees = df_trades['fees'].fillna(0).sum()
                net_pnl = total_pnl - total_fees
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                
                best_trade = df_trades['pnl'].max()
                worst_trade = df_trades['pnl'].min()
                
                avg_duration = df_trades['duration_hours'].mean()
                
                # Análise por dia
                df_trades['date'] = pd.to_datetime(df_trades['exit_timestamp'], unit='s').dt.date
                daily_pnl = df_trades.groupby('date')['pnl'].sum()
                
                profitable_days = len(daily_pnl[daily_pnl > 0])
                total_days = len(daily_pnl)
                
                summary = {
                    "period_days": days,
                    "period_start": df_trades.iloc[0]['exit_timestamp'],
                    "period_end": df_trades.iloc[-1]['exit_timestamp'],
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "total_fees": total_fees,
                    "net_pnl": net_pnl,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "best_trade": best_trade,
                    "worst_trade": worst_trade,
                    "avg_duration_hours": avg_duration,
                    "profitable_days": profitable_days,
                    "total_trading_days": total_days,
                    "daily_win_rate": profitable_days / total_days if total_days > 0 else 0,
                    "profit_factor": abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0,
                    "generated_at": datetime.now()
                }
                
                logger.info(f"[OK] Relatório gerado - Net P&L: {net_pnl:.2f}")
                
                return summary
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na geração do relatório: {e}")
            return {}

    def log_system_event(self, level, component, message, data=None):
        """
        Registra evento do sistema para debugging
        
        Args:
            level (str): Nível do log (INFO, WARNING, ERROR)
            component (str): Componente que gerou o log
            message (str): Mensagem do log
            data (dict): Dados adicionais opcionais
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_logs 
                    (timestamp, level, component, message, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    int(datetime.now().timestamp()),
                    level.upper(),
                    component,
                    message,
                    json.dumps(data, default=str) if data else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"[ERROR] Erro ao registrar evento do sistema: {e}")

    def export_data(self, format="csv", days=30):
        """
        Exporta dados de performance
        
        Args:
            format (str): Formato de exportação (csv, json)
            days (int): Número de dias para exportar
            
        Returns:
            str: Caminho do arquivo exportado ou None se erro
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).timestamp()
            
            with sqlite3.connect(self.db_path) as conn:
                df_trades = pd.read_sql_query("""
                    SELECT * FROM trades 
                    WHERE exit_timestamp >= ? 
                    AND exit_timestamp IS NOT NULL
                    ORDER BY exit_timestamp
                """, conn, params=(cutoff_date,))
                
                if df_trades.empty:
                    logger.warning("[WARN]  Nenhum dado para exportar")
                    return None
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if format.lower() == "csv":
                    filename = f"performance_export_{timestamp}.csv"
                    df_trades.to_csv(filename, index=False)
                elif format.lower() == "json":
                    filename = f"performance_export_{timestamp}.json"
                    df_trades.to_json(filename, orient="records", date_format="iso")
                else:
                    raise ValueError(f"Formato não suportado: {format}")
                
                logger.info(f"📁 Dados exportados para {filename}")
                return filename
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na exportação: {e}")
            return None

    def log_metrics(self, metrics):
        """
        Registra métricas de performance
        
        Args:
            metrics (dict): Métricas a serem registradas
        """
        logger.info(f"[DATA] Registrando métricas: {list(metrics.keys())}")
        
        try:
            # Adiciona timestamp se não existir
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now()
            
            # Converte para JSON para armazenamento
            metrics_json = json.dumps(metrics, default=str)
            
            # Salva no banco de logs do sistema
            self.log_system_event(
                level='INFO',
                component='metrics',
                message='Performance metrics logged',
                data=metrics
            )
            
            # Atualiza cache
            if not hasattr(self, 'metrics_cache_list'):
                self.metrics_cache_list = []
            
            self.metrics_cache_list.append({
                'timestamp': metrics['timestamp'],
                'data': metrics
            })
            
            # Mantém apenas últimas 1000 métricas no cache
            if len(self.metrics_cache_list) > 1000:
                self.metrics_cache_list = self.metrics_cache_list[-1000:]
            
            logger.debug(f"[OK] Métricas registradas: {len(metrics)} campos")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao registrar métricas: {e}")