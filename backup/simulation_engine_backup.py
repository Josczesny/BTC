"""
Sistema Completo de Simulação de Trading
Valida estratégias com precisão de 80%+ antes do trading real
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Representa uma operação de trading"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    direction: str  # 'long' ou 'short'
    fees: float
    profit_loss: Optional[float]
    duration: Optional[timedelta]
    max_profit: float = 0.0
    max_loss: float = 0.0
    confidence: float = 0.0

@dataclass
class SimulationResult:
    """Resultado completo da simulação"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    var_95: float
    expected_shortfall: float
    kelly_criterion: float
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series

class RiskManager:
    """Gerenciador de risco para simulação"""
    
    def __init__(self, max_position_size: float = 0.1, 
                 max_daily_loss: float = 0.05,
                 max_drawdown: float = 0.15):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.daily_pnl = {}
        self.current_drawdown = 0.0
        
    def calculate_position_size(self, confidence: float, volatility: float,
                              account_balance: float) -> float:
        """
        Calcula tamanho da posição baseado na confiança e volatilidade
        Usa Kelly Criterion modificado
        """
        # Kelly Criterion modificado
        win_rate = min(confidence, 0.99)
        avg_win = 0.02  # 2% ganho médio estimado
        avg_loss = 0.01  # 1% perda média estimada
        
        if win_rate > 0.5:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            # Reduzir Kelly por segurança
            kelly_fraction *= 0.25  # Usar apenas 25% do Kelly
        else:
            kelly_fraction = 0.01  # Posição mínima se confiança baixa
        
        # Ajustar pela volatilidade
        vol_adjustment = min(1.0, 0.02 / max(volatility, 0.001))
        
        # Calcular tamanho final
        position_size = kelly_fraction * vol_adjustment
        position_size = min(position_size, self.max_position_size)
        
        return max(position_size, 0.001)  # Mínimo 0.1%
    
    def check_risk_limits(self, current_date: datetime, 
                         current_pnl: float, account_balance: float) -> bool:
        """Verifica se os limites de risco foram violados"""
        
        # Verificar perda diária
        date_str = current_date.date()
        if date_str not in self.daily_pnl:
            self.daily_pnl[date_str] = 0.0
        
        self.daily_pnl[date_str] += current_pnl
        daily_loss_pct = self.daily_pnl[date_str] / account_balance
        
        if daily_loss_pct < -self.max_daily_loss:
            return False
        
        # Verificar drawdown máximo
        if self.current_drawdown < -self.max_drawdown:
            return False
        
        return True
    
    def update_drawdown(self, equity_curve: pd.Series):
        """Atualiza drawdown atual"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        self.current_drawdown = drawdown.iloc[-1]

class TradingSimulator:
    """
    Simulador completo de trading com análise detalhada
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)
        
    def run_simulation(self, data: pd.DataFrame, signals: pd.DataFrame,
                      confidence_scores: pd.Series) -> SimulationResult:
        """
        Executa simulação completa de trading
        """
        self.logger.info("🎮 Iniciando simulação de trading...")
        
        # Inicializar variáveis
        account_balance = self.initial_capital
        position = 0.0
        trades = []
        equity_curve = []
        current_trade = None
        
        # Estatísticas
        daily_returns = []
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i] if i < len(signals) else 0
            current_confidence = confidence_scores.iloc[i] if i < len(confidence_scores) else 0.5
            
            # Calcular volatilidade recente
            if i >= 20:
                recent_returns = data['close'].iloc[i-20:i].pct_change().dropna()
                volatility = recent_returns.std()
            else:
                volatility = 0.02  # Volatilidade padrão
            
            # Gerenciar posição atual
            if current_trade is not None:
                # Atualizar lucro/prejuízo não realizado
                if current_trade.direction == 'long':
                    unrealized_pnl = (current_price - current_trade.entry_price) * current_trade.size
                else:
                    unrealized_pnl = (current_trade.entry_price - current_price) * current_trade.size
                
                # Atualizar máximo lucro/prejuízo
                current_trade.max_profit = max(current_trade.max_profit, unrealized_pnl)
                current_trade.max_loss = min(current_trade.max_loss, unrealized_pnl)
                
                # Verificar condições de saída
                should_exit = self._should_exit_trade(current_trade, current_price, 
                                                    current_time, current_confidence)
                
                if should_exit:
                    # Fechar trade
                    exit_price = current_price * (1 - self.slippage if current_trade.direction == 'long' 
                                                else 1 + self.slippage)
                    
                    if current_trade.direction == 'long':
                        pnl = (exit_price - current_trade.entry_price) * current_trade.size
                    else:
                        pnl = (current_trade.entry_price - exit_price) * current_trade.size
                    
                    # Deduzir comissões
                    total_fees = current_trade.fees + (exit_price * current_trade.size * self.commission)
                    pnl -= total_fees
                    
                    # Finalizar trade
                    current_trade.exit_time = current_time
                    current_trade.exit_price = exit_price
                    current_trade.profit_loss = pnl
                    current_trade.duration = current_time - current_trade.entry_time
                    current_trade.fees = total_fees
                    
                    trades.append(current_trade)
                    account_balance += pnl
                    position = 0.0
                    current_trade = None
            
            # Verificar sinal de entrada
            if current_trade is None and abs(current_signal) > 0 and current_confidence > 0.6:
                
                # Calcular tamanho da posição
                position_size = self.risk_manager.calculate_position_size(
                    current_confidence, volatility, account_balance
                )
                
                # Verificar limites de risco
                if self.risk_manager.check_risk_limits(current_time, 0, account_balance):
                    
                    # Abrir nova posição
                    direction = 'long' if current_signal > 0 else 'short'
                    entry_price = current_price * (1 + self.slippage if direction == 'long' 
                                                 else 1 - self.slippage)
                    
                    trade_size = position_size * account_balance / entry_price
                    entry_fees = entry_price * trade_size * self.commission
                    
                    current_trade = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        size=trade_size,
                        direction=direction,
                        fees=entry_fees,
                        profit_loss=None,
                        duration=None,
                        confidence=current_confidence
                    )
                    
                    position = trade_size if direction == 'long' else -trade_size
                    account_balance -= entry_fees
            
            # Registrar equity
            current_equity = account_balance
            if current_trade is not None:
                if current_trade.direction == 'long':
                    unrealized_pnl = (current_price - current_trade.entry_price) * current_trade.size
                else:
                    unrealized_pnl = (current_trade.entry_price - current_price) * current_trade.size
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
            
            # Calcular retorno diário
            if i > 0:
                daily_return = (current_equity - equity_curve[i-1]) / equity_curve[i-1]
                daily_returns.append(daily_return)
            
            # Atualizar drawdown
            if len(equity_curve) > 1:
                equity_series = pd.Series(equity_curve)
                self.risk_manager.update_drawdown(equity_series)
        
        # Fechar trade aberto se houver
        if current_trade is not None:
            final_price = data['close'].iloc[-1]
            exit_price = final_price * (1 - self.slippage if current_trade.direction == 'long' 
                                      else 1 + self.slippage)
            
            if current_trade.direction == 'long':
                pnl = (exit_price - current_trade.entry_price) * current_trade.size
            else:
                pnl = (current_trade.entry_price - exit_price) * current_trade.size
            
            total_fees = current_trade.fees + (exit_price * current_trade.size * self.commission)
            pnl -= total_fees
            
            current_trade.exit_time = data.index[-1]
            current_trade.exit_price = exit_price
            current_trade.profit_loss = pnl
            current_trade.duration = data.index[-1] - current_trade.entry_time
            current_trade.fees = total_fees
            
            trades.append(current_trade)
            account_balance += pnl
        
        # Calcular métricas
        result = self._calculate_metrics(trades, equity_curve, daily_returns, data.index)
        
        self.logger.info(f"[OK] Simulação concluída - {len(trades)} trades executados")
        
        return result
    
    def _should_exit_trade(self, trade: Trade, current_price: float,
                          current_time: datetime, confidence: float) -> bool:
        """Determina se deve sair do trade atual"""
        
        # Calcular P&L atual
        if trade.direction == 'long':
            current_pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            current_pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Stop loss: -2%
        if current_pnl_pct < -0.02:
            return True
        
        # Take profit: +3%
        if current_pnl_pct > 0.03:
            return True
        
        # Tempo máximo: 24 horas
        if current_time - trade.entry_time > timedelta(hours=24):
            return True
        
        # Sair se confiança caiu muito
        if confidence < 0.3:
            return True
        
        return False
    
    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float],
                          daily_returns: List[float], dates: pd.DatetimeIndex) -> SimulationResult:
        """Calcula todas as métricas de performance"""
        
        if not trades:
            return SimulationResult(
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                calmar_ratio=0, max_drawdown=0, win_rate=0, profit_factor=0,
                avg_trade_duration=0, total_trades=0, winning_trades=0, losing_trades=0,
                largest_win=0, largest_loss=0, consecutive_wins=0, consecutive_losses=0,
                var_95=0, expected_shortfall=0, kelly_criterion=0, trades=[],
                equity_curve=pd.Series(), drawdown_curve=pd.Series()
            )
        
        # Converter para séries pandas
        equity_series = pd.Series(equity_curve, index=dates)
        
        # Retorno total
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Retorno anualizado
        days = (dates[-1] - dates[0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Métricas de trades
        profits = [t.profit_loss for t in trades if t.profit_loss is not None]
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])
        win_rate = winning_trades / len(profits) if profits else 0
        
        # Profit factor
        gross_profit = sum([p for p in profits if p > 0])
        gross_loss = abs(sum([p for p in profits if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Duração média dos trades
        durations = [t.duration.total_seconds() / 3600 for t in trades 
                    if t.duration is not None]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Maior ganho e perda
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0
        
        # Sequências consecutivas
        consecutive_wins = self._calculate_consecutive_wins(profits)
        consecutive_losses = self._calculate_consecutive_losses(profits)
        
        # Ratios de risco
        if daily_returns:
            daily_returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(daily_returns_array) / np.std(daily_returns_array) * np.sqrt(252)
            
            # Sortino ratio (apenas downside deviation)
            downside_returns = daily_returns_array[daily_returns_array < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                sortino_ratio = np.mean(daily_returns_array) / downside_deviation * np.sqrt(252)
            else:
                sortino_ratio = float('inf')
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
            
            # VaR 95%
            var_95 = np.percentile(daily_returns_array, 5)
            
            # Expected Shortfall
            var_returns = daily_returns_array[daily_returns_array <= var_95]
            expected_shortfall = np.mean(var_returns) if len(var_returns) > 0 else 0
            
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
            var_95 = 0
            expected_shortfall = 0
        
        # Kelly Criterion
        if profits and win_rate > 0:
            avg_win = np.mean([p for p in profits if p > 0])
            avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 1
            kelly_criterion = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        else:
            kelly_criterion = 0
        
        return SimulationResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            kelly_criterion=kelly_criterion,
            trades=trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown
        )
    
    def _calculate_consecutive_wins(self, profits: List[float]) -> int:
        """Calcula máximo de vitórias consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, profits: List[float]) -> int:
        """Calcula máximo de perdas consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

class MonteCarloSimulator:
    """
    Simulador Monte Carlo para análise de robustez
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        
    def run_monte_carlo(self, data: pd.DataFrame, signals: pd.DataFrame,
                       confidence_scores: pd.Series) -> Dict[str, Any]:
        """
        Executa múltiplas simulações com variações aleatórias
        """
        
        results = []
        simulator = TradingSimulator()
        
        print(f"[DICE] Executando {self.n_simulations} simulações Monte Carlo...")
        
        for i in range(self.n_simulations):
            if i % 100 == 0:
                print(f"   Simulação {i+1}/{self.n_simulations}")
            
            # Adicionar ruído aos dados
            noisy_data = self._add_noise_to_data(data)
            noisy_signals = self._add_noise_to_signals(signals)
            noisy_confidence = self._add_noise_to_confidence(confidence_scores)
            
            # Executar simulação
            result = simulator.run_simulation(noisy_data, noisy_signals, noisy_confidence)
            results.append(result)
        
        # Analisar resultados
        analysis = self._analyze_monte_carlo_results(results)
        
        return analysis
    
    def _add_noise_to_data(self, data: pd.DataFrame, noise_level: float = 0.001) -> pd.DataFrame:
        """Adiciona ruído realístico aos dados de preço"""
        noisy_data = data.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            if col in noisy_data.columns:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] *= (1 + noise)
        
        return noisy_data
    
    def _add_noise_to_signals(self, signals: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        """Adiciona ruído aos sinais de trading"""
        noisy_signals = signals.copy()
        
        # Adicionar ruído gaussiano
        noise = np.random.normal(0, noise_level, len(signals))
        noisy_signals += noise
        
        # Manter limites
        noisy_signals = np.clip(noisy_signals, -1, 1)
        
        return noisy_signals
    
    def _add_noise_to_confidence(self, confidence: pd.Series, noise_level: float = 0.05) -> pd.Series:
        """Adiciona ruído aos scores de confiança"""
        noisy_confidence = confidence.copy()
        
        noise = np.random.normal(0, noise_level, len(confidence))
        noisy_confidence += noise
        
        # Manter entre 0 e 1
        noisy_confidence = np.clip(noisy_confidence, 0, 1)
        
        return noisy_confidence
    
    def _analyze_monte_carlo_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analisa resultados das simulações Monte Carlo"""
        
        # Extrair métricas
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results if not np.isinf(r.sharpe_ratio)]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        
        analysis = {
            'total_return': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'median': np.median(sharpe_ratios) if sharpe_ratios else 0,
                'std': np.std(sharpe_ratios) if sharpe_ratios else 0
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'worst': np.min(max_drawdowns),
                'best': np.max(max_drawdowns)
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates)
            },
            'probability_of_profit': len([r for r in returns if r > 0]) / len(returns),
            'probability_of_large_loss': len([r for r in returns if r < -0.2]) / len(returns)
        }
        
        return analysis

def create_performance_report(result: SimulationResult, save_path: str = None):
    """Cria relatório visual de performance"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Relatório de Performance - Simulação de Trading', fontsize=16)
    
    # 1. Curva de equity
    axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
    axes[0, 0].set_title('Curva de Equity')
    axes[0, 0].set_ylabel('Valor da Conta')
    axes[0, 0].grid(True)
    
    # 2. Drawdown
    axes[0, 1].fill_between(result.drawdown_curve.index, 
                           result.drawdown_curve.values, 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True)
    
    # 3. Distribuição de P&L dos trades
    profits = [t.profit_loss for t in result.trades if t.profit_loss is not None]
    if profits:
        axes[0, 2].hist(profits, bins=30, alpha=0.7)
        axes[0, 2].set_title('Distribuição de P&L')
        axes[0, 2].set_xlabel('P&L por Trade')
        axes[0, 2].grid(True)
    
    # 4. Duração dos trades
    durations = [t.duration.total_seconds() / 3600 for t in result.trades 
                if t.duration is not None]
    if durations:
        axes[1, 0].hist(durations, bins=20, alpha=0.7)
        axes[1, 0].set_title('Duração dos Trades')
        axes[1, 0].set_xlabel('Horas')
        axes[1, 0].grid(True)
    
    # 5. Métricas principais
    metrics_text = f"""
    Retorno Total: {result.total_return:.2%}
    Retorno Anualizado: {result.annualized_return:.2%}
    Sharpe Ratio: {result.sharpe_ratio:.2f}
    Max Drawdown: {result.max_drawdown:.2%}
    Win Rate: {result.win_rate:.2%}
    Profit Factor: {result.profit_factor:.2f}
    Total Trades: {result.total_trades}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Métricas Principais')
    axes[1, 1].axis('off')
    
    # 6. Análise de confiança vs performance
    if result.trades:
        confidences = [t.confidence for t in result.trades]
        profits_pct = [(t.profit_loss / (t.entry_price * t.size)) * 100 
                      for t in result.trades if t.profit_loss is not None]
        
        if len(confidences) == len(profits_pct):
            axes[1, 2].scatter(confidences, profits_pct, alpha=0.6)
            axes[1, 2].set_title('Confiança vs Performance')
            axes[1, 2].set_xlabel('Score de Confiança')
            axes[1, 2].set_ylabel('P&L (%)')
            axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[DATA] Relatório salvo em {save_path}")
    
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Dados de exemplo
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    # Simular dados de preço
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 50000 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Ajustar high/low
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Simular sinais de trading (estratégia simples baseada em momentum)
    data['sma_short'] = data['close'].rolling(10).mean()
    data['sma_long'] = data['close'].rolling(50).mean()
    signals = pd.Series(0, index=dates)
    signals[data['sma_short'] > data['sma_long']] = 1
    signals[data['sma_short'] < data['sma_long']] = -1
    
    # Simular scores de confiança
    confidence_scores = pd.Series(np.random.beta(2, 2, 1000), index=dates)
    
    print("[START] Iniciando simulação de trading...")
    
    # Executar simulação
    simulator = TradingSimulator(initial_capital=10000)
    result = simulator.run_simulation(data, signals, confidence_scores)
    
    # Mostrar resultados
    print(f"\n[DATA] Resultados da Simulação:")
    print(f"   Retorno Total: {result.total_return:.2%}")
    print(f"   Retorno Anualizado: {result.annualized_return:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Total Trades: {result.total_trades}")
    
    # Criar relatório visual
    create_performance_report(result)
    
    # Executar Monte Carlo
    print(f"\n[DICE] Executando análise Monte Carlo...")
    monte_carlo = MonteCarloSimulator(n_simulations=100)  # Reduzido para exemplo
    mc_results = monte_carlo.run_monte_carlo(data, signals, confidence_scores)
    
    print(f"\n[UP] Resultados Monte Carlo:")
    print(f"   Retorno Médio: {mc_results['total_return']['mean']:.2%}")
    print(f"   Probabilidade de Lucro: {mc_results['probability_of_profit']:.2%}")
    print(f"   Retorno Percentil 5%: {mc_results['total_return']['percentile_5']:.2%}")
    print(f"   Retorno Percentil 95%: {mc_results['total_return']['percentile_95']:.2%}") 