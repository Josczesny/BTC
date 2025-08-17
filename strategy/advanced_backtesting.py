#!/usr/bin/env python3
"""
SISTEMA DE BACKTESTING ROBUSTO COM VALIDAÇÃO ESTATÍSTICA
========================================================

Implementa:
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing
- Out-of-sample validation
- Métricas avançadas de performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    """Sistema de backtesting avançado com validação estatística"""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        
    def walk_forward_analysis(self, strategy_signals: pd.DataFrame, 
                            price_data: pd.DataFrame, 
                            window_size: int = 30,
                            step_size: int = 7) -> Dict:
        """
        Walk-forward analysis para validação robusta
        
        Args:
            strategy_signals: Sinais da estratégia
            price_data: Dados de preço
            window_size: Tamanho da janela de treinamento (dias)
            step_size: Passo para próxima janela (dias)
            
        Returns:
            Dict com resultados da análise
        """
        print("🔄 Executando Walk-Forward Analysis...")
        
        results = {
            'windows': [],
            'train_periods': [],
            'test_periods': [],
            'train_returns': [],
            'test_returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
        
        # Converte para datetime se necessário
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        
        start_date = price_data.index[0]
        end_date = price_data.index[-1]
        
        current_start = start_date
        window_count = 0
        
        while current_start + timedelta(days=window_size) < end_date:
            # Define períodos
            train_end = current_start + timedelta(days=window_size)
            test_end = train_end + timedelta(days=step_size)
            
            # Filtra dados
            train_data = price_data[current_start:train_end]
            test_data = price_data[train_end:test_end]
            
            if len(train_data) < 20 or len(test_data) < 5:
                current_start += timedelta(days=step_size)
                continue
            
            # Simula trades no período de teste
            test_returns = self._simulate_trades_period(test_data, strategy_signals)
            
            # Calcula métricas
            if len(test_returns) > 0:
                sharpe = self._calculate_sharpe_ratio(test_returns)
                max_dd = self._calculate_max_drawdown(test_returns)
                
                results['windows'].append(window_count)
                results['train_periods'].append((current_start, train_end))
                results['test_periods'].append((train_end, test_end))
                results['test_returns'].append(test_returns)
                results['sharpe_ratios'].append(sharpe)
                results['max_drawdowns'].append(max_dd)
            
            window_count += 1
            current_start += timedelta(days=step_size)
        
        # Calcula métricas agregadas
        results['avg_sharpe'] = np.mean(results['sharpe_ratios'])
        results['avg_max_dd'] = np.mean(results['max_drawdowns'])
        results['win_rate'] = len([r for r in results['test_returns'] if np.sum(r) > 0]) / len(results['test_returns'])
        
        print(f"✅ Walk-Forward concluído: {len(results['windows'])} janelas analisadas")
        print(f"📊 Sharpe médio: {results['avg_sharpe']:.3f}")
        print(f"📉 Drawdown médio: {results['avg_max_dd']:.2%}")
        print(f"🎯 Win rate: {results['win_rate']:.1%}")
        
        return results
    
    def monte_carlo_simulation(self, returns: List[float], 
                             n_simulations: int = 1000,
                             time_horizon: int = 252) -> Dict:
        """
        Simulação de Monte Carlo para stress testing
        
        Args:
            returns: Retornos históricos
            n_simulations: Número de simulações
            time_horizon: Horizonte temporal (dias)
            
        Returns:
            Dict com resultados da simulação
        """
        print(f"🎲 Executando Monte Carlo ({n_simulations} simulações)...")
        
        if len(returns) < 10:
            return {'error': 'Dados insuficientes para simulação'}
        
        # Calcula parâmetros dos retornos
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Gera simulações
        simulations = []
        for _ in range(n_simulations):
            # Gera caminho aleatório
            path = np.random.normal(mean_return, std_return, time_horizon)
            cumulative_return = np.cumprod(1 + path) - 1
            simulations.append(cumulative_return)
        
        simulations = np.array(simulations)
        
        # Calcula percentis
        percentiles = [5, 25, 50, 75, 95]
        percentile_returns = {}
        
        for p in percentiles:
            percentile_returns[f'p{p}'] = np.percentile(simulations[:, -1], p)
        
        # Calcula probabilidade de perda
        prob_loss = np.mean(simulations[:, -1] < 0)
        
        results = {
            'simulations': simulations,
            'percentile_returns': percentile_returns,
            'prob_loss': prob_loss,
            'mean_final_return': np.mean(simulations[:, -1]),
            'std_final_return': np.std(simulations[:, -1])
        }
        
        print(f"✅ Monte Carlo concluído")
        print(f"📊 Probabilidade de perda: {prob_loss:.1%}")
        print(f"📈 Retorno médio final: {results['mean_final_return']:.2%}")
        
        return results
    
    def stress_test(self, strategy_signals: pd.DataFrame, 
                   price_data: pd.DataFrame) -> Dict:
        """
        Stress testing com cenários extremos
        
        Args:
            strategy_signals: Sinais da estratégia
            price_data: Dados de preço
            
        Returns:
            Dict com resultados dos testes de stress
        """
        print("🔥 Executando Stress Testing...")
        
        results = {}
        
        # Cenário 1: Alta volatilidade
        high_vol_data = price_data.copy()
        high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.05, len(high_vol_data)))
        high_vol_returns = self._simulate_trades_period(high_vol_data, strategy_signals)
        results['high_volatility'] = {
            'returns': high_vol_returns,
            'total_return': np.sum(high_vol_returns),
            'sharpe': self._calculate_sharpe_ratio(high_vol_returns),
            'max_dd': self._calculate_max_drawdown(high_vol_returns)
        }
        
        # Cenário 2: Tendência de baixa
        bear_market_data = price_data.copy()
        trend = np.linspace(0, -0.3, len(bear_market_data))  # -30% em tendência
        bear_market_data['close'] = bear_market_data['close'] * (1 + trend)
        bear_returns = self._simulate_trades_period(bear_market_data, strategy_signals)
        results['bear_market'] = {
            'returns': bear_returns,
            'total_return': np.sum(bear_returns),
            'sharpe': self._calculate_sharpe_ratio(bear_returns),
            'max_dd': self._calculate_max_drawdown(bear_returns)
        }
        
        # Cenário 3: Gaps de preço
        gap_data = price_data.copy()
        # Adiciona gaps aleatórios
        gap_indices = np.random.choice(len(gap_data), size=len(gap_data)//10, replace=False)
        for idx in gap_indices:
            gap_data.iloc[idx, gap_data.columns.get_loc('close')] *= np.random.uniform(0.9, 1.1)
        
        gap_returns = self._simulate_trades_period(gap_data, strategy_signals)
        results['price_gaps'] = {
            'returns': gap_returns,
            'total_return': np.sum(gap_returns),
            'sharpe': self._calculate_sharpe_ratio(gap_returns),
            'max_dd': self._calculate_max_drawdown(gap_returns)
        }
        
        print("✅ Stress Testing concluído")
        for scenario, metrics in results.items():
            print(f"📊 {scenario}: Retorno {metrics['total_return']:.2%}, Sharpe {metrics['sharpe']:.3f}")
        
        return results
    
    def _simulate_trades_period(self, price_data: pd.DataFrame, 
                              strategy_signals: pd.DataFrame) -> List[float]:
        """Simula trades em um período específico"""
        returns = []
        position = 0
        entry_price = 0
        
        for i in range(len(price_data)):
            if i >= len(strategy_signals):
                break
                
            current_price = price_data.iloc[i]['close']
            signal = strategy_signals.iloc[i] if i < len(strategy_signals) else 0
            
            # Lógica de trading simples
            if signal > 0.6 and position == 0:  # Compra
                position = 1
                entry_price = current_price
            elif signal < 0.4 and position == 1:  # Vende
                returns.append((current_price - entry_price) / entry_price)
                position = 0
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calcula Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252  # Diário
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calcula máximo drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def generate_report(self, results: Dict) -> str:
        """Gera relatório completo de backtesting"""
        report = []
        report.append("=" * 60)
        report.append("📊 RELATÓRIO DE BACKTESTING AVANÇADO")
        report.append("=" * 60)
        
        # Walk-forward results
        if 'walk_forward' in results:
            wf = results['walk_forward']
            report.append(f"\n🔄 WALK-FORWARD ANALYSIS:")
            report.append(f"   • Janelas analisadas: {len(wf['windows'])}")
            report.append(f"   • Sharpe médio: {wf['avg_sharpe']:.3f}")
            report.append(f"   • Drawdown médio: {wf['avg_max_dd']:.2%}")
            report.append(f"   • Win rate: {wf['win_rate']:.1%}")
        
        # Monte Carlo results
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            report.append(f"\n🎲 MONTE CARLO SIMULATION:")
            report.append(f"   • Probabilidade de perda: {mc['prob_loss']:.1%}")
            report.append(f"   • Retorno médio final: {mc['mean_final_return']:.2%}")
            report.append(f"   • P5: {mc['percentile_returns']['p5']:.2%}")
            report.append(f"   • P95: {mc['percentile_returns']['p95']:.2%}")
        
        # Stress test results
        if 'stress_test' in results:
            st = results['stress_test']
            report.append(f"\n🔥 STRESS TESTING:")
            for scenario, metrics in st.items():
                report.append(f"   • {scenario}: {metrics['total_return']:.2%} retorno, {metrics['sharpe']:.3f} Sharpe")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report) 