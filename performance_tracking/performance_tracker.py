#!/usr/bin/env python3
"""
RASTREADOR DE PERFORMANCE
=========================

Módulo responsável pelo rastreamento e análise de performance do sistema.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors

class PerformanceTracker:
    """Rastreador de performance do sistema"""
    
    def __init__(self):
        """Inicializa o rastreador de performance"""
        self.performance_data = {
            'start_time': datetime.now(),
            'trades': [],
            'daily_stats': {},
            'model_performance': {},
            'system_metrics': {}
        }
        
        # Cria diretório de logs se não existir
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def log_trade(self, trade_data):
        """Registra trade no histórico"""
        try:
            trade_entry = {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_data.get('id', 'unknown'),
                'type': trade_data.get('type', 'UNKNOWN'),
                'signal': trade_data.get('signal', 'UNKNOWN'),
                'entry_price': trade_data.get('entry_price', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'quantity': trade_data.get('quantity', 0),
                'position_size': trade_data.get('position_size', 0),
                'pnl': trade_data.get('pnl', 0),
                'pnl_percentage': trade_data.get('pnl_percentage', 0),
                'confidence': trade_data.get('confidence', 0),
                'duration_minutes': trade_data.get('duration_minutes', 0),
                'close_reason': trade_data.get('close_reason', 'unknown')
            }
            
            self.performance_data['trades'].append(trade_entry)
            
            # Atualiza estatísticas diárias
            self._update_daily_stats(trade_entry)
            
            # Salva dados periodicamente
            if len(self.performance_data['trades']) % 10 == 0:
                self.save_performance_data()
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao registrar trade: {e}"))
    
    def log_model_performance(self, model_name, prediction, actual_outcome, confidence):
        """Registra performance do modelo"""
        try:
            if model_name not in self.performance_data['model_performance']:
                self.performance_data['model_performance'][model_name] = {
                    'predictions': [],
                    'accuracy': 0.0,
                    'total_predictions': 0,
                    'correct_predictions': 0
                }
            
            model_data = self.performance_data['model_performance'][model_name]
            
            prediction_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual_outcome': actual_outcome,
                'confidence': confidence,
                'correct': (prediction > 0.5 and actual_outcome > 0.5) or (prediction < 0.5 and actual_outcome < 0.5)
            }
            
            model_data['predictions'].append(prediction_entry)
            model_data['total_predictions'] += 1
            
            if prediction_entry['correct']:
                model_data['correct_predictions'] += 1
            
            model_data['accuracy'] = model_data['correct_predictions'] / model_data['total_predictions']
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao registrar performance do modelo: {e}"))
    
    def log_system_metric(self, metric_name, value, category='general'):
        """Registra métrica do sistema"""
        try:
            if category not in self.performance_data['system_metrics']:
                self.performance_data['system_metrics'][category] = {}
            
            self.performance_data['system_metrics'][category][metric_name] = {
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao registrar métrica: {e}"))
    
    def _update_daily_stats(self, trade_entry):
        """Atualiza estatísticas diárias"""
        try:
            trade_date = datetime.fromisoformat(trade_entry['timestamp']).date()
            date_str = trade_date.isoformat()
            
            if date_str not in self.performance_data['daily_stats']:
                self.performance_data['daily_stats'][date_str] = {
                    'trades': 0,
                    'total_pnl': 0.0,
                    'profitable_trades': 0,
                    'total_volume': 0.0,
                    'avg_confidence': 0.0,
                    'signals': {}
                }
            
            daily_stats = self.performance_data['daily_stats'][date_str]
            
            # Atualiza contadores
            daily_stats['trades'] += 1
            daily_stats['total_pnl'] += trade_entry['pnl']
            daily_stats['total_volume'] += trade_entry['position_size']
            
            if trade_entry['pnl'] > 0:
                daily_stats['profitable_trades'] += 1
            
            # Atualiza confiança média
            total_confidence = daily_stats['avg_confidence'] * (daily_stats['trades'] - 1) + trade_entry['confidence']
            daily_stats['avg_confidence'] = total_confidence / daily_stats['trades']
            
            # Atualiza distribuição de sinais
            signal = trade_entry['signal']
            if signal not in daily_stats['signals']:
                daily_stats['signals'][signal] = 0
            daily_stats['signals'][signal] += 1
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao atualizar estatísticas diárias: {e}"))
    
    def get_performance_summary(self):
        """Obtém resumo da performance"""
        try:
            if not self.performance_data['trades']:
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'total_volume': 0.0,
                    'avg_confidence': 0.0
                }
            
            trades = self.performance_data['trades']
            
            # Estatísticas básicas
            total_trades = len(trades)
            total_pnl = sum(t['pnl'] for t in trades)
            profitable_trades = len([t for t in trades if t['pnl'] > 0])
            total_volume = sum(t['position_size'] for t in trades)
            
            # P&L por trade
            pnls = [t['pnl'] for t in trades]
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            best_trade = max(pnls) if pnls else 0
            worst_trade = min(pnls) if pnls else 0
            
            # Confiança média
            avg_confidence = sum(t['confidence'] for t in trades) / total_trades if total_trades > 0 else 0
            
            # Win rate
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'total_volume': total_volume,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter resumo de performance: {e}"))
            return {}
    
    def get_model_performance_summary(self):
        """Obtém resumo da performance dos modelos"""
        try:
            model_summary = {}
            
            for model_name, model_data in self.performance_data['model_performance'].items():
                if model_data['total_predictions'] > 0:
                    model_summary[model_name] = {
                        'accuracy': model_data['accuracy'],
                        'total_predictions': model_data['total_predictions'],
                        'correct_predictions': model_data['correct_predictions']
                    }
            
            return model_summary
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter resumo dos modelos: {e}"))
            return {}
    
    def get_daily_performance(self, days=7):
        """Obtém performance dos últimos dias"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            daily_performance = []
            
            for date_str, daily_stats in self.performance_data['daily_stats'].items():
                date = datetime.fromisoformat(date_str).date()
                if start_date <= date <= end_date:
                    daily_performance.append({
                        'date': date_str,
                        'trades': daily_stats['trades'],
                        'pnl': daily_stats['total_pnl'],
                        'win_rate': daily_stats['profitable_trades'] / daily_stats['trades'] if daily_stats['trades'] > 0 else 0,
                        'volume': daily_stats['total_volume'],
                        'avg_confidence': daily_stats['avg_confidence']
                    })
            
            return sorted(daily_performance, key=lambda x: x['date'])
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao obter performance diária: {e}"))
            return []
    
    def save_performance_data(self):
        """Salva dados de performance em arquivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/performance_data_{timestamp}.json"
            
            # Converte datetime para string para serialização
            data_to_save = self._convert_datetime_to_string(self.performance_data.copy())
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)
            
            print(TerminalColors.success(f"✅ Dados de performance salvos: {filename}"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao salvar dados de performance: {e}"))
    
    def _convert_datetime_to_string(self, obj):
        """Converte objetos datetime para string para serialização JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_datetime_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_string(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def load_performance_data(self, filename):
        """Carrega dados de performance de arquivo"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.performance_data = data
            print(TerminalColors.success(f"✅ Dados de performance carregados: {filename}"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao carregar dados de performance: {e}"))
    
    def generate_performance_report(self):
        """Gera relatório completo de performance"""
        try:
            summary = self.get_performance_summary()
            model_summary = self.get_model_performance_summary()
            daily_performance = self.get_daily_performance()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': summary,
                'model_performance': model_summary,
                'daily_performance': daily_performance,
                'system_metrics': self.performance_data['system_metrics']
            }
            
            # Salva relatório
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/performance_report_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(TerminalColors.success(f"✅ Relatório de performance gerado: {filename}"))
            
            return report
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao gerar relatório: {e}"))
            return {}
    
    def print_performance_summary(self):
        """Exibe resumo da performance no terminal"""
        try:
            summary = self.get_performance_summary()
            
            if summary['total_trades'] == 0:
                print(TerminalColors.info("📊 Nenhum trade registrado ainda"))
                return
            
            print(TerminalColors.highlight("\n📊 RESUMO DE PERFORMANCE"))
            print(TerminalColors.colorize("="*50, TerminalColors.CYAN))
            
            print(TerminalColors.info(f"📈 Total de trades: {summary['total_trades']}"))
            print(TerminalColors.info(f"💰 P&L total: ${summary['total_pnl']:.2f}"))
            print(TerminalColors.info(f"🎯 Taxa de acerto: {summary['win_rate']:.1%}"))
            print(TerminalColors.info(f"📊 P&L médio: ${summary['avg_pnl']:.2f}"))
            print(TerminalColors.info(f"🚀 Melhor trade: ${summary['best_trade']:.2f}"))
            print(TerminalColors.info(f"📉 Pior trade: ${summary['worst_trade']:.2f}"))
            print(TerminalColors.info(f"💼 Volume total: ${summary['total_volume']:.2f}"))
            print(TerminalColors.info(f"🎯 Confiança média: {summary['avg_confidence']:.1%}"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao exibir resumo: {e}"))
    
    def update_metrics(self, total_trades, successful_trades, total_profit):
        """Atualiza métricas do sistema"""
        try:
            # Registra métricas do sistema
            self.log_system_metric('total_trades', total_trades, 'trading')
            self.log_system_metric('successful_trades', successful_trades, 'trading')
            self.log_system_metric('total_profit', total_profit, 'trading')
            
            # Calcula e registra win rate
            win_rate = successful_trades / max(total_trades, 1)
            self.log_system_metric('win_rate', win_rate, 'trading')
            
            # Registra timestamp da atualização
            self.log_system_metric('last_update', datetime.now().isoformat(), 'system')
            
        except Exception as e:
            print(TerminalColors.warning(f"⚠️ Erro ao atualizar métricas: {e}"))
    
    def save_metrics(self):
        """Salva métricas do sistema"""
        try:
            # Salva dados de performance
            self.save_performance_data()
            
            # Gera relatório
            self.generate_performance_report()
            
            print(TerminalColors.success("💾 Métricas salvas com sucesso"))
            
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro ao salvar métricas: {e}")) 