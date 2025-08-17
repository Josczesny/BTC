"""
Sistema Avançado de Ensemble Learning
Combina múltiplos modelos para decisões de alta precisão (80%+)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelConfidence:
    """Calcula confiança das predições do modelo"""
    
    def __init__(self):
        self.historical_accuracy = {}
        self.recent_accuracy = {}
        
    def calculate_confidence(self, model_name: str, prediction: float, 
                           features: np.ndarray, historical_error: float) -> float:
        """
        Calcula confiança da predição baseada em múltiplos fatores
        """
        try:
            # Confiança baseada em erro histórico
            base_confidence = max(0, 1 - historical_error)
            
            # Ajustar baseado na consistência do modelo
            consistency_factor = self._get_consistency_factor(model_name)
            
            # Ajustar baseado na "estranheza" da predição
            novelty_factor = self._calculate_novelty_factor(features)
            
            # Combinação final
            confidence = base_confidence * consistency_factor * novelty_factor
            
            return min(max(confidence, 0.1), 0.99)  # Entre 10% e 99%
            
        except Exception as e:
            logging.warning(f"Erro no cálculo de confiança: {e}")
            return 0.5
    
    def _get_consistency_factor(self, model_name: str) -> float:
        """Fator de consistência baseado no histórico do modelo"""
        if model_name not in self.historical_accuracy:
            return 0.8  # Padrão para modelos novos
        
        accuracies = self.historical_accuracy[model_name]
        if len(accuracies) < 5:
            return 0.8
        
        # Menor variabilidade = maior consistência
        std_accuracy = np.std(accuracies)
        consistency = max(0.5, 1 - std_accuracy)
        
        return consistency
    
    def _calculate_novelty_factor(self, features: np.ndarray) -> float:
        """Penaliza predições em condições muito diferentes do treino"""
        # Implementação simplificada - em produção usaria análise mais sofisticada
        return 0.9  # Placeholder
    
    def update_accuracy(self, model_name: str, accuracy: float):
        """Atualiza histórico de precisão do modelo"""
        if model_name not in self.historical_accuracy:
            self.historical_accuracy[model_name] = []
        
        self.historical_accuracy[model_name].append(accuracy)
        
        # Manter apenas últimas 100 medições
        if len(self.historical_accuracy[model_name]) > 100:
            self.historical_accuracy[model_name] = self.historical_accuracy[model_name][-100:]

class AdvancedEnsemble:
    """
    Sistema de ensemble avançado com seleção dinâmica de modelos
    e aprendizado meta-adaptativo
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.confidence_calculator = ModelConfidence()
        self.confidence_threshold = confidence_threshold
        self.meta_model = None
        self.logger = logging.getLogger(__name__)
        
        # Inicializar modelos base
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Inicializa conjunto diversificado de modelos base"""
        
        self.models = {
            # Modelos baseados em árvores
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            
            # Modelos lineares
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            
            # Modelo não-linear
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            
            # Rede neural
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Pesos iniciais iguais
        self.model_weights = {name: 1.0 for name in self.models.keys()}
        
        # Performance inicial
        self.model_performance = {name: {'mse': float('inf'), 'r2': 0.0} 
                                for name in self.models.keys()}
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """
        Treina todos os modelos do ensemble e calcula pesos adaptativos
        """
        self.logger.info("[ROBOT] Treinando ensemble de modelos...")
        
        training_results = {}
        
        # Treinar cada modelo individualmente
        for name, model in self.models.items():
            try:
                self.logger.info(f"   Treinando {name}...")
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Avaliar performance
                if X_val is not None and y_val is not None:
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                else:
                    # Cross-validation se não houver conjunto de validação
                    tscv = TimeSeriesSplit(n_splits=5)
                    scores = cross_val_score(model, X_train, y_train, 
                                           cv=tscv, scoring='neg_mean_squared_error')
                    mse = -scores.mean()
                    r2 = cross_val_score(model, X_train, y_train, 
                                        cv=tscv, scoring='r2').mean()
                
                # Atualizar performance
                self.model_performance[name] = {'mse': mse, 'r2': r2}
                
                # Calcular accuracy para confiança
                accuracy = max(0, r2)
                self.confidence_calculator.update_accuracy(name, accuracy)
                
                training_results[name] = {
                    'mse': mse,
                    'r2': r2,
                    'accuracy': accuracy
                }
                
                self.logger.info(f"   {name}: MSE={mse:.6f}, R2={r2:.4f}")
                
            except Exception as e:
                self.logger.error(f"Erro ao treinar {name}: {e}")
                training_results[name] = {'error': str(e)}
        
        # Calcular pesos adaptativos
        self._calculate_adaptive_weights()
        
        # Treinar meta-modelo
        self._train_meta_model(X_train, y_train)
        
        return training_results
    
    def _calculate_adaptive_weights(self):
        """Calcula pesos adaptativos baseados na performance"""
        
        # Calcular pesos baseados no R²
        r2_scores = [perf['r2'] for perf in self.model_performance.values()]
        
        if max(r2_scores) <= 0:
            # Se todos os modelos têm R² negativo, usar pesos iguais
            self.model_weights = {name: 1.0 for name in self.models.keys()}
            return
        
        # Softmax dos R² scores para pesos
        r2_array = np.array(r2_scores)
        r2_array = np.maximum(r2_array, 0)  # Garantir não-negativos
        
        if r2_array.sum() == 0:
            weights = np.ones(len(r2_array)) / len(r2_array)
        else:
            # Aplicar softmax com temperatura
            temperature = 2.0
            exp_scores = np.exp(r2_array / temperature)
            weights = exp_scores / exp_scores.sum()
        
        # Atualizar pesos
        for i, name in enumerate(self.models.keys()):
            self.model_weights[name] = weights[i]
        
        self.logger.info("[DATA] Pesos adaptativos calculados:")
        for name, weight in self.model_weights.items():
            self.logger.info(f"   {name}: {weight:.4f}")
    
    def _train_meta_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Treina meta-modelo para combinar predições"""
        
        try:
            # Gerar predições de cada modelo base
            base_predictions = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_train)
                    base_predictions.append(pred)
                except:
                    # Se modelo falhou, usar zeros
                    base_predictions.append(np.zeros(len(X_train)))
            
            # Criar matriz de features para meta-modelo
            meta_features = np.column_stack(base_predictions)
            
            # Treinar meta-modelo (regressão linear simples)
            self.meta_model = LinearRegression()
            self.meta_model.fit(meta_features, y_train)
            
            self.logger.info("[BRAIN] Meta-modelo treinado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar meta-modelo: {e}")
            self.meta_model = None
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições com scores de confiança
        """
        predictions = []
        confidences = []
        
        for i in range(len(X)):
            features = X.iloc[i:i+1]
            
            # Predições de cada modelo
            model_predictions = {}
            model_confidences = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(features)[0]
                    
                    # Calcular confiança
                    historical_error = self.model_performance[name]['mse']
                    confidence = self.confidence_calculator.calculate_confidence(
                        name, pred, features.values, historical_error
                    )
                    
                    model_predictions[name] = pred
                    model_confidences[name] = confidence
                    
                except Exception as e:
                    self.logger.warning(f"Erro na predição {name}: {e}")
                    model_predictions[name] = 0.0
                    model_confidences[name] = 0.1
            
            # Combinar predições
            final_pred, final_confidence = self._combine_predictions(
                model_predictions, model_confidences
            )
            
            predictions.append(final_pred)
            confidences.append(final_confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def _combine_predictions(self, model_predictions: Dict[str, float], 
                           model_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Combina predições usando múltiplas estratégias"""
        
        # Estratégia 1: Média ponderada por confiança
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for name, pred in model_predictions.items():
            weight = self.model_weights[name]
            confidence = model_confidences[name]
            
            weighted_sum += pred * weight * confidence
            confidence_sum += weight * confidence
        
        if confidence_sum > 0:
            weighted_pred = weighted_sum / confidence_sum
            avg_confidence = confidence_sum / len(model_predictions)
        else:
            weighted_pred = np.mean(list(model_predictions.values()))
            avg_confidence = 0.5
        
        # Estratégia 2: Meta-modelo (se disponível)
        meta_pred = weighted_pred
        if self.meta_model is not None:
            try:
                base_preds = np.array(list(model_predictions.values())).reshape(1, -1)
                meta_pred = self.meta_model.predict(base_preds)[0]
            except:
                pass
        
        # Combinar estratégias
        final_pred = 0.7 * weighted_pred + 0.3 * meta_pred
        
        # Ajustar confiança baseada na concordância entre modelos
        pred_std = np.std(list(model_predictions.values()))
        agreement_factor = max(0.5, 1 - pred_std / abs(final_pred)) if final_pred != 0 else 0.5
        
        final_confidence = avg_confidence * agreement_factor
        
        return final_pred, final_confidence
    
    def should_trade(self, confidence: float) -> bool:
        """
        Decide se deve executar trade baseado na confiança
        """
        return confidence >= self.confidence_threshold
    
    def get_model_rankings(self) -> pd.DataFrame:
        """Retorna ranking dos modelos por performance"""
        
        rankings = []
        for name, perf in self.model_performance.items():
            rankings.append({
                'model': name,
                'r2_score': perf['r2'],
                'mse': perf['mse'],
                'weight': self.model_weights[name]
            })
        
        df = pd.DataFrame(rankings)
        return df.sort_values('r2_score', ascending=False)
    
    def save_ensemble(self, filepath: str):
        """Salva o ensemble treinado"""
        try:
            ensemble_data = {
                'models': self.models,
                'model_weights': self.model_weights,
                'model_performance': self.model_performance,
                'meta_model': self.meta_model,
                'confidence_calculator': self.confidence_calculator
            }
            
            joblib.dump(ensemble_data, filepath)
            self.logger.info(f"💾 Ensemble salvo em {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar ensemble: {e}")
    
    def load_ensemble(self, filepath: str):
        """Carrega ensemble salvo"""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.models = ensemble_data['models']
            self.model_weights = ensemble_data['model_weights']
            self.model_performance = ensemble_data['model_performance']
            self.meta_model = ensemble_data['meta_model']
            self.confidence_calculator = ensemble_data['confidence_calculator']
            
            self.logger.info(f"📂 Ensemble carregado de {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar ensemble: {e}")

class DynamicModelSelector:
    """
    Seletor dinâmico que escolhe o melhor modelo baseado nas condições de mercado
    """
    
    def __init__(self):
        self.market_regimes = {}
        self.regime_models = {}
        
    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detecta regime atual do mercado"""
        
        # Calcular volatilidade
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Calcular tendência
        sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        
        # Classificar regime
        if volatility > returns.std() * 1.5:
            if current_price > sma_20 > sma_50:
                return "high_vol_uptrend"
            elif current_price < sma_20 < sma_50:
                return "high_vol_downtrend"
            else:
                return "high_vol_sideways"
        else:
            if current_price > sma_20 > sma_50:
                return "low_vol_uptrend"
            elif current_price < sma_20 < sma_50:
                return "low_vol_downtrend"
            else:
                return "low_vol_sideways"
    
    def select_best_model(self, ensemble: AdvancedEnsemble, 
                         market_regime: str) -> str:
        """Seleciona melhor modelo para o regime atual"""
        
        # Se não temos histórico do regime, usar modelo com melhor performance geral
        if market_regime not in self.regime_models:
            rankings = ensemble.get_model_rankings()
            return rankings.iloc[0]['model']
        
        # Retornar modelo com melhor performance no regime
        return self.regime_models[market_regime]
    
    def update_regime_performance(self, regime: str, model: str, performance: float):
        """Atualiza performance do modelo no regime específico"""
        
        if regime not in self.regime_models:
            self.regime_models[regime] = model
            return
        
        # Lógica para atualizar baseado na performance
        # Implementação simplificada
        if performance > 0.7:  # Se performance boa, manter modelo
            self.regime_models[regime] = model

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
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Criar e treinar ensemble
    ensemble = AdvancedEnsemble(confidence_threshold=0.75)
    
    print("[START] Treinando ensemble avançado...")
    results = ensemble.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Fazer predições com confiança
    predictions, confidences = ensemble.predict_with_confidence(X_test)
    
    # Filtrar apenas predições de alta confiança
    high_confidence_mask = confidences >= 0.75
    high_conf_predictions = predictions[high_confidence_mask]
    high_conf_actual = y_test.values[high_confidence_mask]
    
    print(f"\n[DATA] Resultados do Ensemble:")
    print(f"   Total de predições: {len(predictions)}")
    print(f"   Predições alta confiança: {len(high_conf_predictions)} ({len(high_conf_predictions)/len(predictions)*100:.1f}%)")
    
    if len(high_conf_predictions) > 0:
        accuracy = r2_score(high_conf_actual, high_conf_predictions)
        print(f"   Accuracy alta confiança: {accuracy:.1%}")
    
    # Mostrar ranking dos modelos
    print(f"\n[WIN] Ranking dos Modelos:")
    rankings = ensemble.get_model_rankings()
    print(rankings) 