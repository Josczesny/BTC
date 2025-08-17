"""
Sistema de Auto-Otimização de Hiperparâmetros
Otimiza automaticamente modelos para máxima precisão
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings('ignore')

class TradingObjective:
    """
    Função objetivo customizada para otimização de trading
    Foca em maximizar precisão, Sharpe ratio e minimizar drawdown
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def calculate_trading_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula score customizado para trading
        Combina accuracy, Sharpe ratio e drawdown
        """
        try:
            # 1. Accuracy baseada em R²
            r2 = r2_score(y_true, y_pred)
            accuracy_score = max(0, r2)
            
            # 2. Simular retornos baseados nas predições
            # Assumir estratégia simples: comprar se predição > 0, vender se < 0
            signals = np.sign(y_pred)
            strategy_returns = signals * y_true
            
            # 3. Calcular Sharpe ratio
            if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns)
                sharpe_score = max(0, min(sharpe_ratio / 3, 1))  # Normalizar para 0-1
            else:
                sharpe_score = 0
            
            # 4. Calcular máximo drawdown
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown)
            drawdown_score = max(0, 1 + max_drawdown / 0.1)  # Penalizar drawdown > 10%
            
            # 5. Calcular win rate
            winning_trades = np.sum(strategy_returns > 0)
            total_trades = len(strategy_returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Combinar scores com pesos
            final_score = (
                accuracy_score * 0.3 +
                sharpe_score * 0.25 +
                drawdown_score * 0.25 +
                win_rate * 0.2
            )
            
            return final_score
            
        except Exception as e:
            logging.warning(f"Erro no cálculo do trading score: {e}")
            return 0.0

class AutoOptimizer:
    """
    Otimizador automático de hiperparâmetros usando Optuna
    """
    
    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout  # 1 hora por padrão
        self.best_params = {}
        self.study_results = {}
        self.logger = logging.getLogger(__name__)
        
    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Otimiza hiperparâmetros do Random Forest"""
        
        def objective(trial):
            # Definir espaço de busca
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            try:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                # Usar função objetivo customizada
                objective_func = TradingObjective(X_train, y_train, X_val, y_val)
                score = objective_func.calculate_trading_score(y_val.values, y_pred)
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Erro na otimização RF: {e}")
                return 0.0
        
        # Criar e executar estudo
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params['random_forest'] = study.best_params
        self.study_results['random_forest'] = {
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        return study.best_params
    
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Otimiza hiperparâmetros do XGBoost"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            try:
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                objective_func = TradingObjective(X_train, y_train, X_val, y_val)
                score = objective_func.calculate_trading_score(y_val.values, y_pred)
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Erro na otimização XGB: {e}")
                return 0.0
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params['xgboost'] = study.best_params
        self.study_results['xgboost'] = {
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        return study.best_params
    
    def optimize_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Otimiza hiperparâmetros da Rede Neural"""
        
        def objective(trial):
            # Arquitetura da rede
            n_layers = trial.suggest_int('n_layers', 1, 4)
            layers = []
            
            for i in range(n_layers):
                n_units = trial.suggest_int(f'n_units_l{i}', 32, 512)
                layers.append(n_units)
            
            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 1000),
                'early_stopping': True,
                'validation_fraction': 0.1
            }
            
            try:
                model = MLPRegressor(**params, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                objective_func = TradingObjective(X_train, y_train, X_val, y_val)
                score = objective_func.calculate_trading_score(y_val.values, y_pred)
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Erro na otimização NN: {e}")
                return 0.0
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params['neural_network'] = study.best_params
        self.study_results['neural_network'] = {
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        return study.best_params
    
    def optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Otimiza todos os modelos e retorna os melhores parâmetros"""
        
        self.logger.info("[FIX] Iniciando otimização automática de hiperparâmetros...")
        
        results = {}
        
        # Otimizar Random Forest
        self.logger.info("   Otimizando Random Forest...")
        try:
            rf_params = self.optimize_random_forest(X_train, y_train, X_val, y_val)
            results['random_forest'] = rf_params
            self.logger.info(f"   [OK] RF otimizado - Score: {self.study_results['random_forest']['best_score']:.4f}")
        except Exception as e:
            self.logger.error(f"   [ERROR] Erro na otimização RF: {e}")
        
        # Otimizar XGBoost
        self.logger.info("   Otimizando XGBoost...")
        try:
            xgb_params = self.optimize_xgboost(X_train, y_train, X_val, y_val)
            results['xgboost'] = xgb_params
            self.logger.info(f"   [OK] XGB otimizado - Score: {self.study_results['xgboost']['best_score']:.4f}")
        except Exception as e:
            self.logger.error(f"   [ERROR] Erro na otimização XGB: {e}")
        
        # Otimizar Neural Network
        self.logger.info("   Otimizando Neural Network...")
        try:
            nn_params = self.optimize_neural_network(X_train, y_train, X_val, y_val)
            results['neural_network'] = nn_params
            self.logger.info(f"   [OK] NN otimizado - Score: {self.study_results['neural_network']['best_score']:.4f}")
        except Exception as e:
            self.logger.error(f"   [ERROR] Erro na otimização NN: {e}")
        
        return results
    
    def get_best_model_type(self) -> str:
        """Retorna o tipo de modelo com melhor performance"""
        
        if not self.study_results:
            return 'random_forest'  # Padrão
        
        best_model = max(self.study_results.items(), 
                        key=lambda x: x[1]['best_score'])
        
        return best_model[0]
    
    def save_optimization_results(self, filepath: str):
        """Salva resultados da otimização"""
        try:
            results = {
                'best_params': self.best_params,
                'study_results': self.study_results
            }
            
            joblib.dump(results, filepath)
            self.logger.info(f"💾 Resultados salvos em {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {e}")
    
    def load_optimization_results(self, filepath: str):
        """Carrega resultados salvos"""
        try:
            results = joblib.load(filepath)
            
            self.best_params = results['best_params']
            self.study_results = results['study_results']
            
            self.logger.info(f"📂 Resultados carregados de {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar resultados: {e}")

class AdaptiveParameterTuner:
    """
    Ajustador adaptativo que modifica parâmetros baseado na performance recente
    """
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.parameter_history = []
        
    def suggest_parameter_adjustment(self, current_params: Dict, 
                                   recent_performance: float) -> Dict:
        """
        Sugere ajustes nos parâmetros baseado na performance recente
        """
        
        # Adicionar ao histórico
        self.performance_history.append(recent_performance)
        self.parameter_history.append(current_params.copy())
        
        # Se não temos histórico suficiente, manter parâmetros
        if len(self.performance_history) < 5:
            return current_params
        
        # Analisar tendência de performance
        recent_trend = np.mean(self.performance_history[-3:]) - np.mean(self.performance_history[-6:-3])
        
        adjusted_params = current_params.copy()
        
        # Se performance está piorando, fazer ajustes
        if recent_trend < -0.05:  # Performance caiu mais de 5%
            
            # Ajustar learning rate se disponível
            if 'learning_rate' in adjusted_params:
                if adjusted_params['learning_rate'] > 0.01:
                    adjusted_params['learning_rate'] *= (1 - self.adaptation_rate)
                else:
                    adjusted_params['learning_rate'] *= (1 + self.adaptation_rate)
            
            # Ajustar regularização
            if 'alpha' in adjusted_params:
                adjusted_params['alpha'] *= (1 + self.adaptation_rate)
            
            # Ajustar número de estimadores
            if 'n_estimators' in adjusted_params:
                adjusted_params['n_estimators'] = min(
                    adjusted_params['n_estimators'] + 50, 500
                )
        
        return adjusted_params
    
    def reset_history(self):
        """Reseta histórico de performance"""
        self.performance_history = []
        self.parameter_history = []

class HyperparameterScheduler:
    """
    Agendador que ajusta hiperparâmetros ao longo do tempo
    """
    
    def __init__(self):
        self.schedules = {}
        
    def create_learning_rate_schedule(self, initial_lr: float, 
                                    decay_rate: float = 0.95,
                                    decay_steps: int = 100) -> callable:
        """Cria schedule para learning rate"""
        
        def schedule(step):
            return initial_lr * (decay_rate ** (step // decay_steps))
        
        return schedule
    
    def create_regularization_schedule(self, initial_alpha: float,
                                     increase_rate: float = 1.05,
                                     increase_steps: int = 50) -> callable:
        """Cria schedule para regularização"""
        
        def schedule(step):
            return initial_alpha * (increase_rate ** (step // increase_steps))
        
        return schedule
    
    def apply_schedules(self, params: Dict, step: int) -> Dict:
        """Aplica todos os schedules aos parâmetros"""
        
        adjusted_params = params.copy()
        
        for param_name, schedule_func in self.schedules.items():
            if param_name in adjusted_params:
                adjusted_params[param_name] = schedule_func(step)
        
        return adjusted_params

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(X.sum(axis=1) + np.random.randn(n_samples) * 0.1)
    
    # Dividir dados
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Criar otimizador
    optimizer = AutoOptimizer(n_trials=20, timeout=300)  # 5 minutos para teste
    
    print("[START] Iniciando otimização automática...")
    
    # Otimizar todos os modelos
    best_params = optimizer.optimize_all_models(X_train, y_train, X_val, y_val)
    
    # Mostrar resultados
    print(f"\n[DATA] Resultados da Otimização:")
    for model_name, results in optimizer.study_results.items():
        print(f"   {model_name}: Score = {results['best_score']:.4f}")
    
    # Melhor modelo
    best_model_type = optimizer.get_best_model_type()
    print(f"\n[WIN] Melhor modelo: {best_model_type}")
    print(f"   Parâmetros: {best_params.get(best_model_type, {})}")
    
    # Salvar resultados
    optimizer.save_optimization_results('optimization_results.joblib') 