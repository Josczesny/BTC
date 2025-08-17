#!/usr/bin/env python3
"""
TREINADOR DE MODELOS ML REAIS
============================

Treina modelos de machine learning com dados históricos reais
para atingir 80%+ de precisão no trading de Bitcoin.

Modelos implementados:
- XGBoost (Gradient Boosting)
- Random Forest
- LSTM (Deep Learning)
- Ensemble combinado
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import sys
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.feature_selection import VarianceThreshold

# Deep Learning
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow não disponível - modelos LSTM desabilitados")

# Adiciona projeto ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logger = setup_logger("ml-trainer")

class EnsembleModel:
    """Modelo ensemble que combina múltiplos modelos"""
    def __init__(self, models_dict):
        self.models = models_dict
    
    def predict(self, X):
        predictions = []
        
        for name, model in self.models.items():
            if 'xgboost' in name or 'random_forest' in name:
                pred = model.predict(X)
                predictions.append(pred)
        
        if predictions:
            # Voto majoritário para classificação
            ensemble_pred = np.round(np.mean(predictions, axis=0))
            return ensemble_pred.astype(int)
        
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        probabilities = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[:, 1]  # Classe positiva
                probabilities.append(prob)
        
        if probabilities:
            return np.mean(probabilities, axis=0)
        
        return np.full(len(X), 0.5)

class RealModelTrainer:
    """Treinador de modelos ML com dados reais"""
    
    def __init__(self):
        """Inicializa o treinador"""
        self.db_path = "data/historical/btc_historical.db"
        self.models_dir = "models/trained/"
        self.results_dir = "results/training/"
        
        # Cria diretórios
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configurações de treinamento
        self.target_accuracy = 0.80  # 80% de precisão
        self.validation_split = 0.2
        self.test_split = 0.1
        
        # Modelos treinados
        self.trained_models = {}
        self.model_performance = {}
        
        logger.info("Treinador de Modelos ML inicializado")
        logger.info("Meta de precisão: {:.1%}".format(self.target_accuracy))
    
    def load_historical_data(self, timeframe: str = "1h", limit: int = None):
        """Carrega dados históricos do banco"""
        try:
            logger.info(f"Carregando dados históricos {timeframe}...")
            
            if not os.path.exists(self.db_path):
                logger.error("Banco de dados históricos não encontrado!")
                logger.info("Execute primeiro: python data/historical_data_collector.py")
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                # Query principal
                query = """
                    SELECT 
                        h.timestamp,
                        h.open, h.high, h.low, h.close, h.volume,
                        h.trades, h.quote_volume
                    FROM historical_data h
                    WHERE h.timeframe = ?
                    ORDER BY h.timestamp
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=(timeframe,))
                
                if df.empty:
                    logger.warning(f"Nenhum dado encontrado para {timeframe}")
                    return None
                
                # Converte timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Carrega indicadores técnicos
                indicators_query = """
                    SELECT timestamp, indicator_name, value
                    FROM technical_indicators
                    WHERE timeframe = ?
                """
                
                indicators_df = pd.read_sql_query(indicators_query, conn, params=(timeframe,))
                
                if not indicators_df.empty:
                    # Pivot indicadores
                    indicators_pivot = indicators_df.pivot(
                        index='timestamp',
                        columns='indicator_name', 
                        values='value'
                    )
                    
                    # Converte index
                    indicators_pivot.index = pd.to_datetime(indicators_pivot.index, unit='s')
                    
                    # Merge com dados principais
                    df = df.merge(indicators_pivot, left_index=True, right_index=True, how='left')
                
                logger.info(f"Carregados {len(df):,} registros com {len(df.columns)} features")
                return df
                
        except Exception as e:
            logger.error(f"Erro carregando dados: {e}")
            return None
    
    def select_top_features(self, X, y, n_top=20):
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        # Seleção por mutual information
        mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        mi_indices = np.argsort(mi)[::-1][:max(n_top*2, 30)]  # pega mais para o RF filtrar depois
        mi_features = [X.columns[i] for i in mi_indices if mi[i] > 0]
        print(f"Features selecionadas por mutual information: {mi_features}")
        # Seleção final por RandomForest
        X_mi = X[mi_features]
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_mi, y)
        importances = rf.feature_importances_
        indices = importances.argsort()[::-1][:n_top]
        top_features = [X_mi.columns[i] for i in indices]
        print(f"Top {n_top} features finais: {top_features}")
        return top_features

    def prepare_features_and_targets(self, df: pd.DataFrame, prediction_horizon: int = 1):
        """Prepara features e targets para treinamento"""
        try:
            from utils.technical_indicators import (
                calculate_cci, calculate_obv, calculate_williams_r, calculate_mom, calculate_trix, calculate_ultosc
            )
            logger.info("Preparando features e targets...")
            if df is None or df.empty:
                return None, None
            # Remove linhas com muitos NaN
            df_clean = df.dropna(thresh=int(len(df.columns) * 0.7))  # 70% dos dados devem existir
            # Fill NaN restantes com métodos apropriados
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            features_list = []
            price_features = ['open', 'high', 'low', 'close', 'volume']
            features_list.extend(price_features)
            # Features derivadas já existentes
            df_clean['price_change'] = df_clean['close'].pct_change()
            df_clean['volume_change'] = df_clean['volume'].pct_change()
            df_clean['high_low_ratio'] = df_clean['high'] / df_clean['low']
            df_clean['open_close_ratio'] = df_clean['open'] / df_clean['close']
            derived_features = ['price_change', 'volume_change', 'high_low_ratio', 'open_close_ratio']
            features_list.extend(derived_features)
            # Indicadores técnicos extras (só se não existem)
            if 'cci_20' not in df_clean.columns:
                df_clean['cci_20'] = calculate_cci(df_clean, period=20)
            if 'obv' not in df_clean.columns:
                df_clean['obv'] = calculate_obv(df_clean)
            if 'williams_r_14' not in df_clean.columns:
                df_clean['williams_r_14'] = calculate_williams_r(df_clean, period=14)
            if 'mom_10' not in df_clean.columns:
                df_clean['mom_10'] = calculate_mom(df_clean, period=10)
            if 'trix_15' not in df_clean.columns:
                df_clean['trix_15'] = calculate_trix(df_clean, period=15)
            if 'ultosc' not in df_clean.columns:
                df_clean['ultosc'] = calculate_ultosc(df_clean)
            extra_indicators = ['cci_20', 'obv', 'williams_r_14', 'mom_10', 'trix_15', 'ultosc']
            features_list.extend([col for col in extra_indicators if col in df_clean.columns])
            # Retornos e volatilidade em múltiplas janelas
            for window in [3, 7, 14, 30]:
                col_ret = f'return_{window}'
                col_vol = f'volatility_{window}'
                if col_ret not in df_clean.columns:
                    df_clean[col_ret] = df_clean['close'].pct_change(window)
                if col_vol not in df_clean.columns:
                    df_clean[col_vol] = df_clean['close'].rolling(window).std()
                features_list.extend([col_ret, col_vol])
            # Diferença entre preço e médias móveis
            for window in [5, 10, 20, 50]:
                sma_col = f'sma_{window}'
                if sma_col in df_clean.columns:
                    diff_col = f'close_sma_diff_{window}'
                    df_clean[diff_col] = df_clean['close'] - df_clean[sma_col]
                    features_list.append(diff_col)
            # Relação volume/volume médio
            for window in [5, 10, 20]:
                vol_ma_col = f'volume_ma_{window}'
                if vol_ma_col not in df_clean.columns:
                    df_clean[vol_ma_col] = df_clean['volume'].rolling(window).mean()
                vol_ratio_col = f'volume_ratio_{window}'
                df_clean[vol_ratio_col] = df_clean['volume'] / df_clean[vol_ma_col]
                features_list.extend([vol_ma_col, vol_ratio_col])
            # Cruzamento de médias
            if 'sma_5' in df_clean.columns and 'sma_20' in df_clean.columns:
                df_clean['sma5_above_sma20'] = (df_clean['sma_5'] > df_clean['sma_20']).astype(int)
                features_list.append('sma5_above_sma20')
            # Price action: corpo, sombra, relação corpo/sombra
            df_clean['candle_body'] = abs(df_clean['close'] - df_clean['open'])
            df_clean['upper_shadow'] = df_clean['high'] - df_clean[['close', 'open']].max(axis=1)
            df_clean['lower_shadow'] = df_clean[['close', 'open']].min(axis=1) - df_clean['low']
            df_clean['body_to_shadow_ratio'] = df_clean['candle_body'] / (df_clean['upper_shadow'] + df_clean['lower_shadow'] + 1e-6)
            features_list.extend(['candle_body', 'upper_shadow', 'lower_shadow', 'body_to_shadow_ratio'])
            # Candles seguidos de alta/baixa
            df_clean['bull_candles'] = (df_clean['close'] > df_clean['open']).astype(int)
            df_clean['bear_candles'] = (df_clean['close'] < df_clean['open']).astype(int)
            df_clean['bull_run'] = df_clean['bull_candles'].rolling(5).sum()
            df_clean['bear_run'] = df_clean['bear_candles'].rolling(5).sum()
            features_list.extend(['bull_run', 'bear_run'])
            # Features de indicadores técnicos já disponíveis
            technical_indicators = [col for col in df_clean.columns if col not in features_list]
            if technical_indicators:
                features_list.extend(technical_indicators[:50])  # Limita a 50 indicadores extras
            # Features de lag (valores passados)
            lag_periods = [1, 2, 3, 5, 10]
            for lag in lag_periods:
                df_clean[f'close_lag_{lag}'] = df_clean['close'].shift(lag)
                df_clean[f'volume_lag_{lag}'] = df_clean['volume'].shift(lag)
                features_list.extend([f'close_lag_{lag}', f'volume_lag_{lag}'])
            # Features de janelas móveis
            windows = [5, 10, 20]
            for window in windows:
                df_clean[f'close_ma_{window}'] = df_clean['close'].rolling(window).mean()
                df_clean[f'close_std_{window}'] = df_clean['close'].rolling(window).std()
                df_clean[f'volume_ma_{window}'] = df_clean['volume'].rolling(window).mean()
                features_list.extend([f'close_ma_{window}', f'close_std_{window}', f'volume_ma_{window}'])
            # ========== TARGETS ==========
            df_clean['price_direction'] = np.where(
                df_clean['close'].shift(-prediction_horizon) > df_clean['close'], 1, 0
            )
            df_clean['future_return'] = (
                df_clean['close'].shift(-prediction_horizon) / df_clean['close'] - 1
            )
            df_clean = df_clean.dropna(subset=['future_return'])
            future_returns = df_clean['future_return']
            df_clean['movement_category'] = pd.cut(
                future_returns,
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=[0, 1, 2, 3, 4]  # strong_down, down, sideways, up, strong_up
            ).astype(int)
            df_final = df_clean.dropna()
            # Selecionar apenas colunas numéricas
            X = df_final[features_list].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
            # Remove features constantes/baixa variância
            selector = VarianceThreshold(threshold=1e-3)
            X_sel = selector.fit_transform(X)
            mask = selector.get_support()
            features_selected = [f for i, f in enumerate(X.columns) if mask[i]]
            removed_low_var = [f for i, f in enumerate(X.columns) if not mask[i]]
            if removed_low_var:
                logger.info(f"Features removidas por baixa variância: {removed_low_var}")
                print(f"Features removidas por baixa variância: {removed_low_var}")
            # Remover features altamente correlacionadas
            X_df = pd.DataFrame(X_sel, columns=features_selected, index=df_final.index)
            corr_matrix = X_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # 2. Ajuste do threshold de correlação (menos agressivo)
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95) and column not in ['close', 'high', 'low', 'sma_20', 'ema_50', 'ema_200', 'bb_upper', 'bb_middle', 'bb_lower']]
            if to_drop:
                logger.info(f"Features removidas por alta correlação: {to_drop}")
                print(f"Features removidas por alta correlação: {to_drop}")
            X_df = X_df.drop(columns=to_drop)
            features_selected = [col for col in features_selected if col not in to_drop]
            # Padroniza as features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df)
            y_direction = df_final['price_direction']
            y_return = df_final['future_return']
            y_category = df_final['movement_category']
            # Seleção automática das top 30 features
            X_top = pd.DataFrame(X_scaled, columns=features_selected, index=df_final.index)
            top_features = self.select_top_features(X_top, y_direction, n_top=30)  # Aumentado de 20 para 30
            X_top = X_top[top_features]
            logger.info(f"Variância das features selecionadas: {X_top.var().to_dict()}")
            print(f"Variância das features selecionadas: {X_top.var().to_dict()}")
            logger.info(f"Features preparadas: {X_top.shape}")
            logger.info(f"   {len(top_features)} features após seleção automática")
            logger.info(f"   {len(y_direction)} amostras")
            logger.info(f"   Classes balanceadas: {y_direction.value_counts().to_dict()}")
            print(f"Features finais utilizadas: {top_features}")
            # Loga se o target está desbalanceado
            class_counts = y_direction.value_counts(normalize=True)
            if (class_counts < 0.4).any() or (class_counts > 0.6).any():
                logger.warning(f"Target desbalanceado: {class_counts.to_dict()}")
                print(f"Target desbalanceado: {class_counts.to_dict()}")
            return {
                'X': X_top,
                'y_direction': y_direction,
                'y_return': y_return,
                'y_category': y_category,
                'feature_names': top_features,
                'scaler': scaler
            }, df_final
        except Exception as e:
            logger.error(f"Erro preparando features: {e}")
            return None, None
    
    def train_xgboost_model(self, data: dict, task: str = "classification", target_type: str = 'binary', dataset_name=None):
        """Treina modelo XGBoost com balanceamento, tuning, walk-forward e early stopping (apenas no fit final)"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            logger.info("Treinando modelo XGBoost...")
            X = data['X']
            X = X.loc[:, ~X.columns.duplicated()]
            if target_type == 'binary':
                y = data['y_direction']
            elif target_type == 'multiclass':
                y = data['y_category']
            else:
                y = data['y_return']
            # Calcular scale_pos_weight para balanceamento dinâmico
            X_bal, y_bal = self.get_balanced_data(X, y)
            scale_pos_weight = sum(y_bal == 0) / sum(y_bal == 1) if sum(y_bal == 1) > 0 else 1.0
            
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'scale_pos_weight': [scale_pos_weight],  # Usar peso dinâmico
                'eval_metric': ['logloss'],
                'gamma': [0, 0.1, 0.5, 1.0],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0, 0.1, 1.0]
            }
            # Split para avaliação final e early stopping
            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
            
            # Walk-forward validation no conjunto de treino
            avg_score, best_params = self.walk_forward_validation(xgb.XGBClassifier, param_dist, X_train_full, y_train_full, n_splits=5, task='classification')
            logger.info(f"XGBoost Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            print(f"XGBoost Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            
            # Balancear dados de treino
            X_bal, y_bal = self.get_balanced_data(X_train_full, y_train_full)
            # Dividir em treino/validação para early stopping
            X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
            model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=10
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            # Opcional: refit em todos os dados com o melhor número de árvores
            best_n = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else best_params.get('n_estimators', 100)
            params_final = dict(best_params)
            params_final.pop('n_estimators', None)
            final_model = xgb.XGBClassifier(
                **params_final,
                n_estimators=best_n,
                random_state=42,
                n_jobs=-1
            )
            final_model.fit(X_bal, y_bal, verbose=False)
            
            # Avaliação final no conjunto de teste
            y_pred_test = final_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            logger.info(f"XGBoost - Acurácia no conjunto de teste final: {test_accuracy:.4f}")
            print(f"XGBoost - Acurácia no conjunto de teste final: {test_accuracy:.4f}")
            
            model_path = f"{self.models_dir}xgboost_{task}_{target_type}_{dataset_name}_model.joblib" if dataset_name else f"{self.models_dir}xgboost_{task}_{target_type}_model.joblib"
            import joblib
            joblib.dump(final_model, model_path)
            logger.info(f"Modelo salvo: {model_path}")
            self.trained_models[f'xgboost_{task}_{target_type}'] = final_model
            self.model_performance[f'xgboost_{task}_{target_type}'] = {
                'walkforward_score': avg_score, 
                'test_accuracy': test_accuracy,
                'best_params': best_params
            }
            return final_model, {'walkforward_score': avg_score, 'test_accuracy': test_accuracy, 'best_params': best_params}
        except Exception as e:
            logger.error(f"Erro treinando XGBoost: {e}")
            print(f"Erro treinando XGBoost: {e}")
            return None, None
    
    def get_balanced_data(self, X, y):
        # Garantir que class_counts seja float
        class_counts = pd.Series(y).value_counts(normalize=True).astype(float)
        if (class_counts < 0.4).any() or (class_counts > 0.6).any():
            logger.warning(f"Aplicando SMOTE para balancear target: {class_counts.to_dict()}")
            print(f"Aplicando SMOTE para balancear target: {class_counts.to_dict()}")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"Após SMOTE: {pd.Series(y_res).value_counts(normalize=True).astype(float).to_dict()}")
            return X_res, y_res
        return X, y

    def walk_forward_validation(self, model_class, param_dist, X, y, n_splits=5, task='classification', **fit_kwargs):
        """Validação walk-forward com TimeSeriesSplit e RandomizedSearchCV melhorado"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        best_params = None
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Aplicar balanceamento
            X_train, y_train = self.get_balanced_data(X_train, y_train)
            
            # RandomizedSearchCV com mais iterações para melhor exploração
            search = RandomizedSearchCV(
                model_class(), 
                param_distributions=param_dist, 
                n_iter=100,  # Aumentado de 50 para 100
                cv=3, 
                scoring='accuracy', 
                random_state=42, 
                n_jobs=-1, 
                verbose=0
            )
            search.fit(X_train, y_train, **fit_kwargs)
            
            model = search.best_estimator_
            y_pred = model.predict(X_test)
            
            if task == 'classification':
                acc = accuracy_score(y_test, y_pred)
                logger.info(f"Fold {fold+1} Accuracy: {acc:.4f}")
                print(f"Fold {fold+1} Accuracy: {acc:.4f}")
                scores.append(acc)
            elif task == 'regression':
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(y_test, y_pred)
                print(f"Fold {fold+1} MSE: {mse:.4f}")
                scores.append(mse)
            
            if best_params is None:
                best_params = search.best_params_
        
        avg_score = np.mean(scores)
        print(f"Walk-forward média: {avg_score:.4f}")
        return avg_score, best_params

    def train_random_forest_model(self, data: dict, task: str = "classification", target_type: str = 'binary', dataset_name=None):
        """Treina modelo Random Forest com balanceamento, tuning e walk-forward"""
        try:
            logger.info("Treinando modelo Random Forest...")
            X = data['X']
            if target_type == 'binary':
                y = data['y_direction']
            elif target_type == 'multiclass':
                y = data['y_category']
            else:
                y = data['y_return']
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced']
            }
            avg_score, best_params = self.walk_forward_validation(RandomForestClassifier, param_dist, X, y, n_splits=5, task='classification')
            logger.info(f"RandomForest Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            print(f"RandomForest Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            # Treina no conjunto completo com melhores params
            X_bal, y_bal = self.get_balanced_data(X, y)
            model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_bal, y_bal)
            model_path = f"{self.models_dir}random_forest_{task}_{target_type}_{dataset_name}_model.joblib" if dataset_name else f"{self.models_dir}random_forest_{task}_{target_type}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Modelo salvo: {model_path}")
            self.trained_models[f'random_forest_{task}_{target_type}'] = model
            self.model_performance[f'random_forest_{task}_{target_type}'] = {'walkforward_score': avg_score, 'best_params': best_params}
            return model, {'walkforward_score': avg_score, 'best_params': best_params}
        except Exception as e:
            logger.error(f"Erro treinando Random Forest: {e}")
            print(f"Erro treinando Random Forest: {e}")
            return None, None
    
    def train_lstm_model(self, data: dict, sequence_length: int = 60, dataset_name=None):
        """Treina modelo LSTM (se TensorFlow disponível)"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow não disponível - pulando LSTM")
                return None, None
            logger.info("Treinando modelo LSTM...")
            X = data['X'].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = data['y_direction']
            X_sequences, y_sequences = self._create_sequences(X.values, y.values, sequence_length)
            split_index = int(len(X_sequences) * (1 - self.test_split))
            X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
            y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            
            # Garantir compilação do LSTM antes do treinamento
            if not hasattr(model, 'compiled') or not model.compiled:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("LSTM compilado antes do treinamento")
                print("LSTM compilado antes do treinamento")
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"LSTM Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            performance = {
                'accuracy': accuracy,
                'model_type': 'lstm_classifier',
                'test_samples': len(y_test),
                'sequence_length': sequence_length
            }
            model_path = f"{self.models_dir}lstm_{dataset_name}_model.h5" if dataset_name else f"{self.models_dir}lstm_model.h5"
            model.save(model_path)
            scaler_path = f"{self.models_dir}lstm_scaler_{dataset_name}.joblib" if dataset_name else f"{self.models_dir}lstm_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Modelo LSTM salvo: {model_path}")
            self.trained_models['lstm'] = model
            self.model_performance['lstm'] = performance
            return model, performance
        except Exception as e:
            logger.error(f"Erro treinando LSTM: {e}")
            return None, None
    
    def train_lightgbm_model(self, data: dict, task: str = "classification", target_type: str = 'binary', dataset_name=None):
        """Treina modelo LightGBM com balanceamento, tuning, walk-forward e early stopping"""
        try:
            import lightgbm as lgb
            logger.info("Treinando modelo LightGBM...")
            X = data['X']
            X = X.loc[:, ~X.columns.duplicated()]
            if target_type == 'binary':
                y = data['y_direction']
            elif target_type == 'multiclass':
                y = data['y_category']
            else:
                y = data['y_return']
            
            # Calcular scale_pos_weight para balanceamento dinâmico
            X_bal, y_bal = self.get_balanced_data(X, y)
            scale_pos_weight = sum(y_bal == 0) / sum(y_bal == 1) if sum(y_bal == 1) > 0 else 1.0
            
            param_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 6, 10, 15, 20, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'class_weight': ['balanced', None],
                'num_leaves': [15, 31, 63, 127, 255],
                'min_gain_to_split': [0, 0.1, 0.5, 1.0],
                'min_data_in_leaf': [10, 20, 50, 100],
                'scale_pos_weight': [scale_pos_weight]
            }
            
            # NÃO usar early stopping no tuning - apenas no fit final
            avg_score, best_params = self.walk_forward_validation(lgb.LGBMClassifier, param_dist, X, y, n_splits=5, task='classification')
            logger.info(f"LightGBM Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            print(f"LightGBM Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            
            # Fit final com early stopping
            model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
            model.fit(
                X_bal, y_bal,
                eval_set=[(X_bal, y_bal)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
            
            model_path = f"{self.models_dir}lightgbm_{task}_{target_type}_{dataset_name}_model.joblib" if dataset_name else f"{self.models_dir}lightgbm_{task}_{target_type}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Modelo salvo: {model_path}")
            self.trained_models[f'lightgbm_{task}_{target_type}'] = model
            self.model_performance[f'lightgbm_{task}_{target_type}'] = {'walkforward_score': avg_score, 'best_params': best_params}
            # Após SMOTE/padronização
            logger.info(f"Shape de X_bal após SMOTE: {X_bal.shape}")
            logger.info(f"Distribuição das classes após SMOTE: {pd.Series(y_bal).value_counts(normalize=True).to_dict()}")
            return model, {'walkforward_score': avg_score, 'best_params': best_params}
        except Exception as e:
            logger.error(f"Erro treinando LightGBM: {e}")
            print(f"Erro treinando LightGBM: {e}")
            return None, None

    def train_catboost_model(self, data: dict, task: str = "classification", target_type: str = 'binary', dataset_name=None):
        """Treina modelo CatBoost com balanceamento, tuning, walk-forward e early stopping (apenas no fit final)"""
        try:
            from catboost import CatBoostClassifier
            logger.info("Treinando modelo CatBoost...")
            X = data['X']
            X = X.loc[:, ~X.columns.duplicated()]
            if target_type == 'binary':
                y = data['y_direction']
            elif target_type == 'multiclass':
                y = data['y_category']
            else:
                y = data['y_return']
            param_dist = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            }
            # Não passar early_stopping_rounds no tuning
            avg_score, best_params = self.walk_forward_validation(CatBoostClassifier, param_dist, X, y, n_splits=5, task='classification', verbose=0)
            logger.info(f"CatBoost Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            print(f"CatBoost Walk-forward média: {avg_score:.4f}, melhores params: {best_params}")
            X_bal, y_bal = self.get_balanced_data(X, y)
            model = CatBoostClassifier(**best_params, random_state=42, verbose=0)
            # Aplicar early stopping apenas no fit final
            model.fit(X_bal, y_bal, early_stopping_rounds=10, eval_set=[(X_bal, y_bal)], verbose=False)
            model_path = f"{self.models_dir}catboost_{task}_{target_type}_{dataset_name}.cbm" if dataset_name else f"{self.models_dir}catboost_{task}_{target_type}.cbm"
            model.save_model(model_path)
            logger.info(f"Modelo salvo: {model_path}")
            self.trained_models[f'catboost_{task}_{target_type}'] = model
            self.model_performance[f'catboost_{task}_{target_type}'] = {'walkforward_score': avg_score, 'best_params': best_params}
            return model, {'walkforward_score': avg_score, 'best_params': best_params}
        except Exception as e:
            logger.error(f"Erro treinando CatBoost: {e}")
            print(f"Erro treinando CatBoost: {e}")
            return None, None
    
    def train_gru_model(self, data, sequence_length=60, dataset_name=None):
        """Treina modelo GRU (se TensorFlow disponível)"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow não disponível - pulando GRU")
                return None, None
            logger.info("Treinando modelo GRU...")
            X = data['X'].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = data['y_direction']
            X_sequences, y_sequences = self._create_sequences(X.values, y.values, sequence_length)
            split_index = int(len(X_sequences) * (1 - self.test_split))
            X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
            y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            from tensorflow.keras.layers import GRU
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            # Garantir compilação do GRU antes do treinamento
            if not hasattr(model, 'compiled') or not model.compiled:
                model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                logger.info("GRU compilado antes do treinamento")
                print("GRU compilado antes do treinamento")
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"GRU Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            performance = {'accuracy': accuracy, 'model_type': 'gru_classifier', 'test_samples': len(y_test), 'sequence_length': sequence_length}
            model_path = f"{self.models_dir}gru_{dataset_name}_model.h5" if dataset_name else f"{self.models_dir}gru_model.h5"
            model.save(model_path)
            scaler_path = f"{self.models_dir}gru_scaler_{dataset_name}.joblib" if dataset_name else f"{self.models_dir}gru_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Modelo GRU salvo: {model_path}")
            self.trained_models['gru'] = model
            self.model_performance['gru'] = performance
            return model, performance
        except Exception as e:
            logger.error(f"Erro treinando GRU: {e}")
            return None, None

    def train_cnn_model(self, data, sequence_length=60, dataset_name=None):
        """Treina modelo CNN (1D) para séries temporais"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow não disponível - pulando CNN")
                return None, None
            logger.info("Treinando modelo CNN...")
            X = data['X'].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = data['y_direction']
            X_sequences, y_sequences = self._create_sequences(X.values, y.values, sequence_length)
            split_index = int(len(X_sequences) * (1 - self.test_split))
            X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
            y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
            model = Sequential([
                Conv1D(32, kernel_size=3, activation='relu', input_shape=(sequence_length, X.shape[1])),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            # Garantir compilação do CNN antes do treinamento
            if not hasattr(model, 'compiled') or not model.compiled:
                model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                logger.info("CNN compilado antes do treinamento")
                print("CNN compilado antes do treinamento")
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"CNN Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            performance = {'accuracy': accuracy, 'model_type': 'cnn_classifier', 'test_samples': len(y_test), 'sequence_length': sequence_length}
            model_path = f"{self.models_dir}cnn_{dataset_name}_model.h5" if dataset_name else f"{self.models_dir}cnn_model.h5"
            model.save(model_path)
            scaler_path = f"{self.models_dir}cnn_scaler_{dataset_name}.joblib" if dataset_name else f"{self.models_dir}cnn_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Modelo CNN salvo: {model_path}")
            self.trained_models['cnn'] = model
            self.model_performance['cnn'] = performance
            return model, performance
        except Exception as e:
            logger.error(f"Erro treinando CNN: {e}")
            return None, None
    
    def _create_sequences(self, X, y, sequence_length):
        """Cria sequências para LSTM"""
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_ensemble_model(self, models: dict):
        """Cria modelo ensemble combinando múltiplos modelos"""
        try:
            logger.info("Criando modelo ensemble...")
            
            ensemble = EnsembleModel(models)
            
            # Salva ensemble
            ensemble_path = f"{self.models_dir}ensemble_model.joblib"
            joblib.dump(ensemble, ensemble_path)
            
            logger.info(f"Ensemble criado com {len(models)} modelos")
            logger.info(f"Ensemble salvo: {ensemble_path}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Erro criando ensemble: {e}")
            return None
    
    def evaluate_all_models(self, data: dict):
        """Avalia todos os modelos treinados"""
        try:
            logger.info("\n AVALIAÇÃO DE TODOS OS MODELOS")
            logger.info("="*50)
            print("\nAVALIAÇÃO DE TODOS OS MODELOS")
            print("="*50)
            best_model = None
            best_accuracy = 0
            for model_name, performance in self.model_performance.items():
                if 'accuracy' in performance:
                    accuracy = performance['accuracy']
                    logger.info(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
                elif 'mse' in performance:
                    mse = performance['mse']
                    logger.info(f"{model_name}: MSE={mse:.6f}")
                    print(f"{model_name}: MSE={mse:.6f}")
            logger.info(f"\n MELHOR MODELO: {best_model} ({best_accuracy:.4f})")
            print(f"\nMELHOR MODELO: {best_model} ({best_accuracy:.4f})")
            if best_accuracy >= self.target_accuracy:
                logger.info("META DE 80% ATINGIDA!")
                logger.info("Modelos prontos para Fase 2 - Testnet Intensivo")
                print("META DE 80% ATINGIDA!")
                print("Modelos prontos para Fase 2 - Testnet Intensivo")
            else:
                logger.info(f"Meta não atingida. Faltam {(self.target_accuracy - best_accuracy):.1%}")
                logger.info("Sugestões: Mais dados, feature engineering, hyperparameter tuning")
                print(f"Meta não atingida. Faltam {(self.target_accuracy - best_accuracy):.1%}")
                print("Sugestões: Mais dados, feature engineering, hyperparameter tuning")
            return best_model, best_accuracy
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            print(f"Erro na avaliação: {e}")
            return None, 0
    
    def save_training_report(self, dataset_name=None):
        """Salva relatório detalhado do treinamento"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{self.results_dir}training_report_{timestamp}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'target_accuracy': self.target_accuracy,
                'models_trained': list(self.trained_models.keys()),
                'performance': self.model_performance,
                'best_model': max(self.model_performance.items(), 
                                key=lambda x: x[1].get('accuracy', 0)) if self.model_performance else None
            }
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Relatório salvo: {report_path}")
            
        except Exception as e:
            logger.error(f"Erro salvando relatório: {e}")
    
    def train_stacking_meta_model(self, data, models_base, target_type='binary', dataset_name=None):
        from sklearn.linear_model import LogisticRegression
        logger.info("Treinando meta-modelo (stacking)...")
        X = data['X']
        if target_type == 'binary':
            y = data['y_direction']
        elif target_type == 'multiclass':
            y = data['y_category']
        else:
            y = data['y_return']
        # Gera previsões dos modelos base
        preds = []
        for name, model in models_base.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1] if target_type == 'binary' else model.predict(X)
            else:
                pred = model.predict(X)
            preds.append(pred)
        X_meta = np.column_stack(preds)
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_meta, y)
        y_pred = meta_model.predict(X_meta)
        if target_type == 'binary' or target_type == 'multiclass':
            acc = accuracy_score(y, y_pred)
            logger.info(f"Meta-modelo (stacking) Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"Meta-modelo (stacking) Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            performance = {'accuracy': acc, 'model_type': 'stacking_meta', 'test_samples': len(y)}
        else:
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y, y_pred)
            logger.info(f"Meta-modelo (stacking) MSE: {mse:.6f}")
            print(f"Meta-modelo (stacking) MSE: {mse:.6f}")
            performance = {'mse': mse, 'model_type': 'stacking_meta', 'test_samples': len(y)}
        self.trained_models['stacking_meta'] = meta_model
        self.model_performance['stacking_meta'] = performance
        return meta_model, performance

    def train_all_datasets(self, data_dir='data/historical'):
        """
        Treina todos os modelos em todos os datasets (.csv e .parquet) encontrados em data/historical.
        Salva modelos e relatórios com o nome do dataset.
        """
        print("Iniciando treinamento em todos os datasets...")
        dataset_files = glob.glob(os.path.join(data_dir, '*.csv')) + glob.glob(os.path.join(data_dir, '*.parquet'))
        for file_path in dataset_files:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\n--- Treinando modelos para dataset: {dataset_name} ---")
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    print(f"Ignorando arquivo não suportado: {file_path}")
                    continue
                # Garantir que o índice seja datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Remover colunas de timestamp, se existirem
                df = df.drop(columns=[col for col in df.columns if 'timestamp' in col.lower()], errors='ignore')
                # Rodar pipeline completo
                self.full_training_pipeline_sync(df, dataset_name=dataset_name)
            except Exception as e:
                print(f"Erro ao treinar com {dataset_name}: {e}")

    def _run_training_pipeline(self, data: dict, df: pd.DataFrame, dataset_name=None):
        """Executa o núcleo do pipeline de treinamento, dado features e DataFrame"""
        try:
            logger.info("TREINANDO MODELOS ML...")
            # Treina modelos base para todos os targets
            xgb_model, xgb_perf = self.train_xgboost_model(data, "classification", target_type='binary', dataset_name=dataset_name)
            rf_model, rf_perf = self.train_random_forest_model(data, "classification", target_type='binary', dataset_name=dataset_name)
            lgb_model, lgb_perf = self.train_lightgbm_model(data, "classification", target_type='binary', dataset_name=dataset_name)
            cat_model, cat_perf = self.train_catboost_model(data, "classification", target_type='binary', dataset_name=dataset_name)
            models_base = {
                'xgb': xgb_model,
                'rf': rf_model,
                'lgb': lgb_model,
                'cat': cat_model
            }
            if TENSORFLOW_AVAILABLE:
                lstm_model, lstm_perf = self.train_lstm_model(data, dataset_name=dataset_name)
                gru_model, gru_perf = self.train_gru_model(data, dataset_name=dataset_name)
                cnn_model, cnn_perf = self.train_cnn_model(data, dataset_name=dataset_name)
                models_base['lstm'] = lstm_model
                models_base['gru'] = gru_model
                models_base['cnn'] = cnn_model
            # Stacking/meta-modelo
            meta_model, meta_perf = self.train_stacking_meta_model(data, models_base, target_type='binary', dataset_name=dataset_name)
            # Treina e avalia para targets alternativos
            xgb_multi, _ = self.train_xgboost_model(data, "classification", target_type='multiclass', dataset_name=dataset_name)
            lgb_multi, _ = self.train_lightgbm_model(data, "classification", target_type='multiclass', dataset_name=dataset_name)
            cat_multi, _ = self.train_catboost_model(data, "classification", target_type='multiclass', dataset_name=dataset_name)
            xgb_reg, _ = self.train_xgboost_model(data, "regression", target_type='regression', dataset_name=dataset_name)
            lgb_reg, _ = self.train_lightgbm_model(data, "regression", target_type='regression', dataset_name=dataset_name)
            cat_reg, _ = self.train_catboost_model(data, "regression", target_type='regression', dataset_name=dataset_name)
            # Ensemble tradicional
            if len(self.trained_models) > 1:
                ensemble = self.create_ensemble_model(self.trained_models)
            best_model, best_accuracy = self.evaluate_all_models(data)
            self.save_training_report(dataset_name=dataset_name)
            logger.info(f"RESULTADO FINAL:")
            logger.info(f"    Melhor modelo: {best_model}")
            logger.info(f"    Acurácia: {best_accuracy}")
            print(f"RESULTADO FINAL: Melhor modelo: {best_model}, Acurácia: {best_accuracy}")
            return True
        except Exception as e:
            logger.error(f"Erro no pipeline: {e}")
            print(f"Erro no pipeline: {e}")
            return False
    
    async def full_training_pipeline(self, timeframe: str = "1h"):
        """Executa pipeline completo de treinamento"""
        try:
            logger.info("INICIANDO PIPELINE COMPLETO DE TREINAMENTO")
            logger.info("="*60)
            # 1. Carrega dados
            df = self.load_historical_data(timeframe)
            if df is None:
                logger.error("Falha ao carregar dados")
                return False
            # 2. Prepara features e targets
            data, df_processed = self.prepare_features_and_targets(df)
            if data is None:
                logger.error("Falha ao preparar features")
                return False
            return self._run_training_pipeline(data, df_processed)
        except Exception as e:
            logger.error(f"Erro no pipeline: {e}")
            return False

    def full_training_pipeline_sync(self, df: pd.DataFrame, dataset_name=None):
        """Executa pipeline completo de treinamento de forma síncrona, recebendo DataFrame já carregado"""
        try:
            logger.info("INICIANDO PIPELINE COMPLETO DE TREINAMENTO (SYNC)")
            logger.info("="*60)
            # 1. Prepara features e targets
            data, df_processed = self.prepare_features_and_targets(df)
            if data is None:
                logger.error("Falha ao preparar features")
                return False
            
            # Executar pipeline de treinamento
            result = self._run_training_pipeline(data, df_processed, dataset_name)
            
            # Executar testes de retreinamento se o treinamento foi bem-sucedido
            if result:
                self.test_retraining_scenarios(num_trades=100, verbose=False)
            
            return result
        except Exception as e:
            logger.error(f"Erro no pipeline sync: {e}")
            return False

    def test_retraining_scenarios(self, num_trades=500, verbose=False):
        """Simula trades e verifica o retreinamento"""
        if not verbose:
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
        
        logger.info(f"Simulando {num_trades} trades para teste de retreinamento...")
        print(f"Simulando {num_trades} trades para teste de retreinamento...")
        
        # Simular trades
        simulated_trades = []
        for i in range(num_trades):
            # Simular features baseadas nos modelos treinados
            if self.trained_models:
                # Usar o primeiro modelo como referência para dimensão das features
                first_model = list(self.trained_models.values())[0]
                if hasattr(first_model, 'feature_names_in_'):
                    n_features = len(first_model.feature_names_in_)
                else:
                    n_features = 20  # Fallback
            else:
                n_features = 20
            
            trade = {
                'features': np.random.rand(n_features),
                'outcome': np.random.choice([0, 1]),
                'price': 50000 + np.random.rand() * 1000,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=i)
            }
            simulated_trades.append(trade)
        
        # Simular processamento de trades
        for i, trade in enumerate(simulated_trades):
            X_trade = np.array(trade['features']).reshape(1, -1)
            
            for model_name, model in self.trained_models.items():
                if 'lstm' not in model_name and 'gru' not in model_name and 'cnn' not in model_name:  # Exclui modelos de sequência
                    try:
                        pred = model.predict(X_trade)
                        # Simular performance fictícia para trigger
                        perf = self.model_performance.get(model_name, {})
                        perf['simulated_accuracy'] = np.random.uniform(0.5, 0.9)
                        perf['trades_processed'] = perf.get('trades_processed', 0) + 1
                        self.model_performance[model_name] = perf
                        
                        # Verificar triggers de retreinamento
                        if i % 50 == 0:  # Verificar a cada 50 trades
                            self.check_retraining_triggers(model_name, perf)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Erro ao processar trade {i} com modelo {model_name}: {e}")
        
        logger.info(f"Teste de retreinamento concluído: {num_trades} trades simulados")
        print(f"Teste de retreinamento concluído: {num_trades} trades simulados")
        
        if not verbose:
            logging.getLogger().setLevel(original_level)

async def main():
    """Função principal"""
    try:
        trainer = RealModelTrainer()
        
        # Executa treinamento completo
        success = await trainer.full_training_pipeline(timeframe="1h")
        
        if success:
            logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
            logger.info("OK Modelos prontos para validação")
            return 0
        else:
            logger.info("Aviso Treinamento concluído, mas meta não atingida")
            logger.info("Ajustes necessários nos modelos")
            return 1
            
    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 