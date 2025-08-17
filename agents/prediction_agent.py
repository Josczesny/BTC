# agents/prediction_agent.py
"""
Agente de Previsão de Preços - REFATORADO PARA USAR SISTEMAS CENTRALIZADOS
Previsão de direção do preço de Bitcoin usando modelos de IA

Funcionalidades:
- Modelos LSTM para séries temporais
- XGBoost para features técnicas
- Prophet para tendências sazonais
- Ensemble otimizado para máximo lucro
- INTEGRADO COM CENTRAL FEATURE ENGINE
"""

import os
# 🚫 SUPRIME LOGS TENSORFLOW
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time

# Flags de disponibilidade
TENSORFLOW_AVAILABLE = True
XGBOOST_AVAILABLE = True
PROPHET_AVAILABLE = True

from utils.logger import setup_trading_logger
# INTEGRAÇÃO COM SISTEMAS CENTRALIZADOS
from models.central_feature_engine import CentralFeatureEngine
from models.central_feature_engine import enrich_with_advanced_indicators

logger = setup_trading_logger("prediction-agent")

class PredictionAgent:
    def __init__(self, config_path='config/agents.json', central_feature_engine=None):
        """
        Inicializa o agente de previsão - REFATORADO PARA USAR SISTEMAS CENTRALIZADOS
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.models_path = 'models/trained'
        self.min_data_points = 50
        self.sequence_length = 20  # Para LSTM
        self.lookback_periods = 20  # Para sequências
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5 minutos
        self.last_training_time = time.time()
        self.ensemble_weights = {'lstm': 0.4, 'xgboost': 0.3, 'rf': 0.2, 'prophet': 0.1}
        
        # 🚀 INTEGRAÇÃO COM SISTEMAS CENTRALIZADOS
        self.central_feature_engine = central_feature_engine or CentralFeatureEngine()
        
        # Inicializa scalers
        try:
            from sklearn.preprocessing import StandardScaler
            self.price_scaler = StandardScaler()
        except:
            self.price_scaler = None
        
        # Cria diretório de modelos se não existir
        os.makedirs(self.models_path, exist_ok=True)
        
        # Carrega modelos pré-treinados
        self._load_models()
        
        logger.info("[OK] PredictionAgent inicializado com Central Feature Engine")

    def _load_models(self):
        """
        Carrega modelos pré-treinados salvos ou cria novos
        """
        try:
            model_files = {
                'lstm': 'lstm_model.h5',
                'xgboost': 'xgboost_model.joblib', 
                'rf': 'rf_model.joblib',
                'prophet': 'prophet_model.joblib'
            }
            
            models_loaded = 0
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_path, filename)
                if os.path.exists(filepath):
                    try:
                        if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                            # Carrega modelo LSTM TensorFlow
                            custom_objects = {
                                'mse': tf.keras.losses.MeanSquaredError(),
                                'mae': tf.keras.losses.MeanAbsoluteError()
                            }
                            self.models[model_name] = tf.keras.models.load_model(
                                filepath, 
                                custom_objects=custom_objects
                            )
                            logger.info(f"[OK] Modelo {model_name} carregado")
                            models_loaded += 1
                        elif model_name in ['xgboost', 'rf']:
                            # Carrega modelos XGBoost e RandomForest
                            self.models[model_name] = joblib.load(filepath)
                            logger.info(f"[OK] Modelo {model_name} carregado")
                            models_loaded += 1
                        elif model_name == 'prophet' and PROPHET_AVAILABLE:
                            # Carrega modelo Prophet se existir
                            self.models[model_name] = joblib.load(filepath)
                            logger.info(f"[OK] Modelo {model_name} carregado")
                            models_loaded += 1
                            
                    except Exception as e:
                        logger.warning(f"[WARN] Erro ao carregar {model_name}: {e}")
                        # Remove modelo corrompido
                        try:
                            os.remove(filepath)
                            logger.info(f"[CLEAN] Modelo {model_name} corrompido removido")
                        except:
                            pass
                else:
                    logger.debug(f"[NOTE] Modelo {model_name} não encontrado")
            
            # Relatório final
            if models_loaded > 0:
                logger.info(f"[OK] PredictionAgent inicializado com {models_loaded} modelos")
                logger.debug(f"[MODELS] Disponíveis: {', '.join(self.models.keys())}")
            else:
                logger.warning("[WARN] Nenhum modelo carregado - usando fallbacks")
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao carregar modelos: {e}")
            self.models = {}

    def prepare_features(self, price_data):
        """Prepara features usando sistema centralizado e indicadores avançados"""
        try:
            if self.central_feature_engine is not None:
                features = self.central_feature_engine.get_all_features(price_data, 'basic')
            else:
                features = self._prepare_basic_features_fallback(price_data)
            # Enriquecer com indicadores avançados
            features = enrich_with_advanced_indicators(features)
            return features
        except Exception as e:
            print(f"⚠️ Erro na preparação de features: {e}")
            return pd.DataFrame()
    
    def _prepare_basic_features_fallback(self, price_data):
        """Fallback básico para features se sistema central não estiver disponível"""
        try:
            if price_data is None or len(price_data) < 20:
                return pd.DataFrame()
            
            # Usa apenas as últimas 50 linhas para evitar problemas de length
            df = price_data.tail(50).copy()
            
            # Calcula indicadores básicos com sistema centralizado
            rsi_values = self._calculate_rsi(df['close'].values)
            # Garante que RSI tenha o mesmo tamanho que o DataFrame
            if len(rsi_values) < len(df):
                # Preenche com valores padrão se necessário
                rsi_padded = np.full(len(df), 50.0)
                rsi_padded[-len(rsi_values):] = rsi_values
                df['rsi'] = rsi_padded
            else:
                df['rsi'] = pd.Series(rsi_values[-len(df):], index=df.index)
            
            df['volatility'] = df['close'].rolling(20).std().fillna(df['close'].std())
            df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'].mean())
            df['sma_50'] = df['close'].rolling(50).mean().fillna(df['close'].mean())
            
            # Garante que todas as colunas tenham o mesmo tamanho
            df = df.dropna()  # Remove linhas com NaN
            
            # Seleciona features básicas (8 features como esperado pelos modelos)
            basic_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_features = [col for col in basic_features if col in df.columns]
            
            # Garante que temos pelo menos algumas features
            if len(available_features) < 3:
                print("⚠️ Poucas features disponíveis, usando fallback")
                return pd.DataFrame()
            
            # Garante que o DataFrame tenha exatamente o tamanho esperado
            result_df = df[available_features].copy()
            
            # Se ainda houver problemas de tamanho, usa apenas a última linha
            if len(result_df) > 0:
                return result_df.iloc[-1:].copy()
            else:
                return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠️ Erro no fallback de features: {e}")
            return pd.DataFrame()
    
    def _prepare_basic_features(self, price_data):
        """Alias para compatibilidade - usa sistema centralizado"""
        return self.prepare_features(price_data)
    
    def _prepare_features(self, market_data):
        """Alias para compatibilidade - usa sistema centralizado"""
        return self.prepare_features(market_data)

    def _calculate_basic_indicators(self, df):
        """
        Calcula indicadores básicos se não existirem
        """
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Médias móveis
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Volatilidade
            df['volatility'] = df['close'].rolling(20).std()
            
            return df
        except Exception as e:
            logger.error(f"[ERROR] Erro no cálculo de indicadores básicos: {e}")
            return df

    def predict(self, market_data):
        """
        Prediz movimento do preço usando modelos treinados
        """
        try:
            current_price = market_data.get('current_price', 0)
            
            # Prepara features
            features = self._prepare_features(market_data)
            if features is None:
                return self._fallback_prediction(market_data)
            
            # Gera predições de todos os modelos disponíveis
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        # Usa features DataFrame diretamente para LSTM
                        lstm_result = self._lstm_prediction(features)
                        if lstm_result['confidence'] > 0:
                            predictions[model_name] = lstm_result['prediction']
                            confidences[model_name] = lstm_result['confidence']
                        
                    elif model_name in ['xgboost', 'rf']:
                        # Predição XGBoost/RandomForest
                        if model_name == 'xgboost':
                            xgb_result = self._xgboost_prediction(features)
                            if xgb_result['confidence'] > 0:
                                predictions[model_name] = xgb_result['prediction']
                                confidences[model_name] = xgb_result['confidence']
                        else:
                            rf_result = self._random_forest_prediction(features)
                            if rf_result['confidence'] > 0:
                                predictions[model_name] = rf_result['prediction']
                                confidences[model_name] = rf_result['confidence']
                        
                    elif model_name == 'prophet' and PROPHET_AVAILABLE:
                        # Predição Prophet
                        try:
                            future = model.make_future_dataframe(periods=1, freq='H')
                            forecast = model.predict(future)
                            pred = forecast['yhat'].iloc[-1]
                            predictions[model_name] = pred
                            confidences[model_name] = 0.70  # Confiança Prophet
                        except Exception as e:
                            logger.debug(f"[WARN] Prophet prediction failed: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"[WARN] Erro predição {model_name}: {e}")
                    continue
            
            # Se não há predições, usa fallback
            if not predictions:
                logger.debug("[FALLBACK] Usando predição técnica")
                return self._fallback_prediction(market_data)
                
            # Calcula ensemble das predições
            ensemble_pred = np.mean(list(predictions.values()))
            avg_confidence = np.mean(list(confidences.values()))
            
            # Determina direção (buy/sell/hold)
            price_change = (ensemble_pred - current_price) / current_price
            
            if price_change > 0.001:  # +0.1% ou mais
                direction = "buy"
            elif price_change < -0.001:  # -0.1% ou menos
                direction = "sell"
            else:
                direction = "hold"
                
            # Força do sinal baseada na mudança percentual
            signal_strength = min(abs(price_change) * 100, 1.0)
            
            return {
                'direction': direction,
                'confidence': avg_confidence,
                'signal_strength': signal_strength,
                'predicted_price': ensemble_pred,
                'current_price': current_price,
                'models_used': len(predictions),
                'predictions': predictions,
                'price_change_pct': price_change * 100,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na predição: {e}")
            return self._fallback_prediction(market_data)

    def _lstm_prediction(self, features_df):
        """
        Previsão usando modelo LSTM real - com 8 features corretas
        """
        try:
            # Se não tem modelo LSTM, usa fallback rápido em vez de treinar
            if 'lstm' not in self.models or not TENSORFLOW_AVAILABLE:
                logger.debug("[FALLBACK] LSTM não disponível - usando fallback")
                return {"prediction": 0.0, "confidence": 0.1, "model": "lstm_fallback"}
            
            model = self.models['lstm']
            
            # Usa as 8 features do Central Feature Engine
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_columns = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_columns) < 8:
                logger.warning(f"[WARN] LSTM: Esperadas 8 features, encontradas {len(available_columns)}")
                return {"prediction": 0.0, "confidence": 0.2, "model": "lstm"}
            
            # Remove targets das features
            model_features = features_df[available_columns].copy()
            
            # Remove linhas com NaN
            model_features = model_features.dropna()
            
            if len(model_features) < 20:
                return {"prediction": 0.0, "confidence": 0.2, "model": "lstm"}
            
            # Normaliza dados
            if self.price_scaler is not None:
                data_scaled = self.price_scaler.fit_transform(model_features.values)
            else:
                # Fallback: normalização manual simples
                data_scaled = (model_features.values - model_features.values.mean()) / model_features.values.std()
            
            # Cria sequências
            sequence_length = min(self.lookback_periods, len(data_scaled) - 1)
            if sequence_length < 10:
                return {"prediction": 0.0, "confidence": 0.2, "model": "lstm"}
            
            X = []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i])
            
            if len(X) == 0:
                return {"prediction": 0.0, "confidence": 0.2, "model": "lstm"}
            
            X = np.array(X)
            
            # Previsão (com timeout implícito - só uma chamada rápida)
            prepared = self._prepare_basic_features(features_df)
            # Garante exatamente 8 features
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'price_change_1']
            X_data = prepared[feature_cols].values if all(col in prepared.columns for col in feature_cols) else prepared.iloc[:, :8].values
            if len(X_data) < 20:
                return {"prediction": 0.0, "confidence": 0.2, "model": "lstm"}
            # Pega últimos 20 timesteps
            X_sequence = X_data[-20:].reshape(1, 20, 8)
            pred = model.predict(X_sequence, verbose=0)[0][0]
            
            # Converte para direção e confiança
            confidence = min(abs(pred) * 2, 0.9)  # Limita confiança
            direction_signal = np.tanh(pred * 5)  # Amplifica sinal
            
            logger.debug(f"[LSTM] Previsão: {direction_signal:.4f}, Confiança: {confidence:.4f}")
            
            return {
                "prediction": float(direction_signal),
                "confidence": float(confidence),
                "model": "lstm"
            }
            
        except Exception as e:
            logger.warning(f"[WARN] Erro LSTM: {e}")
            return {"prediction": 0.0, "confidence": 0.1, "model": "lstm_error"}

    def _xgboost_prediction(self, features_df):
        """
        Previsão usando XGBoost real - com 8 features corretas
        """
        try:
            # Força retreinamento se modelo existe mas tem features incompatíveis
            if 'xgboost' in self.models:
                try:
                    # Testa se o modelo funciona com as features atuais
                    test_features = features_df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']].iloc[-1:].fillna(0)
                    self.models['xgboost'].predict(test_features)
                    model = self.models['xgboost']
                except Exception as e:
                    logger.info("[BRAIN] Modelo XGBoost com features incompatíveis - retreinando...")
                    model = self._train_xgboost_model(features_df)
                    if model is None:
                        return {"prediction": 0.0, "confidence": 0.3, "model": "xgboost"}
            elif XGBOOST_AVAILABLE:
                # Treina modelo se não existe
                model = self._train_xgboost_model(features_df)
                if model is None:
                    return {"prediction": 0.0, "confidence": 0.3, "model": "xgboost"}
            else:
                return {"prediction": 0.0, "confidence": 0.3, "model": "xgboost"}
            
            # Usa as 8 features do Central Feature Engine
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_columns = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_columns) < 8:
                logger.warning(f"[WARN] XGBoost: Esperadas 8 features, encontradas {len(available_columns)}")
                return {"prediction": 0.0, "confidence": 0.2, "model": "xgboost"}
            
            # Remove targets das features
            model_features = features_df[available_columns].copy()
            
            # Remove linhas com NaN
            model_features = model_features.dropna()
            
            if len(model_features) < 5:
                return {"prediction": 0.0, "confidence": 0.2, "model": "xgboost"}
            
            # Últimas features
            latest_features = model_features.iloc[-1:].fillna(0)
            
            # Previsão
            prediction = model.predict(latest_features)[0]
            
            # Feature importance para confiança
            if hasattr(model, 'feature_importances_'):
                confidence = min(np.mean(model.feature_importances_) * 3, 0.9)
            else:
                confidence = 0.6
            
            # Converte para sinal de direção
            direction_signal = np.tanh(prediction * 10)
            
            logger.debug(f"[XGBOOST] Previsão: {direction_signal:.4f}, Confiança: {confidence:.4f}")
            
            return {
                "prediction": float(direction_signal),
                "confidence": float(confidence),
                "model": "xgboost"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro XGBoost: {e}")
            return {"prediction": 0.0, "confidence": 0.3, "model": "xgboost"}

    def _prophet_prediction(self, features_df):
        """
        Previsão usando Prophet real
        """
        try:
            if not PROPHET_AVAILABLE:
                return {"prediction": 0.0, "confidence": 0.2, "model": "prophet"}
            
            # Prepara dados para Prophet
            prophet_data = features_df.reset_index()
            
            if 'timestamp' not in prophet_data.columns:
                prophet_data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(prophet_data), freq='1H')
            
            prophet_df = pd.DataFrame({
                'ds': prophet_data['timestamp'],
                'y': prophet_data['close']
            })
            
            # Modelo Prophet
            if 'prophet' not in self.models:
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.1
                )
                model.fit(prophet_df)
                self.models['prophet'] = model
            else:
                model = self.models['prophet']
            
            # Previsão
            future = model.make_future_dataframe(periods=1, freq='H')
            forecast = model.predict(future)
            
            current_price = prophet_df['y'].iloc[-1]
            predicted_price = forecast['yhat'].iloc[-1]
            
            # Calcula direção
            price_change = (predicted_price - current_price) / current_price
            direction_signal = np.tanh(price_change * 20)
            
            # Confiança baseada no intervalo de predição
            uncertainty = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
            confidence = max(0.3, min(0.8, 1.0 - (uncertainty / predicted_price)))
            
            return {
                "prediction": float(direction_signal),
                "confidence": float(confidence),
                "model": "prophet"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro Prophet: {e}")
            return {"prediction": 0.0, "confidence": 0.2, "model": "prophet"}

    def _random_forest_prediction(self, features_df):
        """
        Previsão usando Random Forest como fallback - com 8 features corretas
        """
        try:
            # Força retreinamento se modelo existe mas tem features incompatíveis
            if 'rf' in self.models:
                try:
                    # Testa se o modelo funciona com as features atuais
                    test_features = features_df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']].iloc[-1:].fillna(0)
                    self.models['rf'].predict(test_features)
                    model = self.models['rf']
                except Exception as e:
                    logger.info("[BRAIN] Modelo Random Forest com features incompatíveis - retreinando...")
                    model = self._train_random_forest_model(features_df)
                    if model is None:
                        return {"prediction": 0.0, "confidence": 0.4, "model": "rf"}
            else:
                model = self._train_random_forest_model(features_df)
                if model is None:
                    return {"prediction": 0.0, "confidence": 0.4, "model": "rf"}
            
            # Usa as 8 features do Central Feature Engine
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_columns = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_columns) < 8:
                logger.warning(f"[WARN] RandomForest: Esperadas 8 features, encontradas {len(available_columns)}")
                return {"prediction": 0.0, "confidence": 0.3, "model": "rf"}
            
            # Remove targets das features
            model_features = features_df[available_columns].copy()
            
            # Remove linhas com NaN
            model_features = model_features.dropna()
            
            if len(model_features) < 3:
                return {"prediction": 0.0, "confidence": 0.3, "model": "rf"}
            
            # Últimas features
            latest_features = model_features.iloc[-1:].fillna(0)
            
            # Previsão
            prediction = model.predict(latest_features)[0]
            
            # Confiança baseada na variância das árvores
            if hasattr(model, 'estimators_'):
                tree_predictions = [tree.predict(latest_features)[0] for tree in model.estimators_[:10]]
                confidence = max(0.3, min(0.8, 1.0 - np.std(tree_predictions)))
            else:
                confidence = 0.5
            
            direction_signal = np.tanh(prediction * 8)
            
            logger.debug(f"[RF] Previsão: {direction_signal:.4f}, Confiança: {confidence:.4f}")
            
            return {
                "prediction": float(direction_signal),
                "confidence": float(confidence),
                "model": "rf"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro Random Forest: {e}")
            return {"prediction": 0.0, "confidence": 0.4, "model": "rf"}

    def _ensemble_predictions(self, predictions):
        """
        Combina previsões usando pesos otimizados para lucro
        """
        try:
            if not predictions:
                return {"prediction": 0.0, "confidence": 0.0, "direction": "neutral"}
            
            # Calcula ensemble ponderado
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                if model_name in self.ensemble_weights:
                    weight = self.ensemble_weights[model_name] * pred["confidence"]
                    weighted_prediction += pred["prediction"] * weight
                    weighted_confidence += pred["confidence"] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_prediction = 0.0
                final_confidence = 0.0
            
            # Determina direção
            if final_prediction > 0.1:
                direction = "bullish"
            elif final_prediction < -0.1:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Ajusta confiança baseado na concordância entre modelos
            model_predictions = [p["prediction"] for p in predictions.values()]
            agreement = 1.0 - np.std(model_predictions) if len(model_predictions) > 1 else 0.5
            final_confidence *= agreement
            
            return {
                "prediction": float(final_prediction),
                "confidence": float(min(final_confidence, 0.95)),
                "direction": direction,
                "model_count": len(predictions),
                "individual_predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no ensemble: {str(e)}")
            return {"prediction": 0.0, "confidence": 0.0, "direction": "neutral"}

    def _train_xgboost_model(self, features_df):
        """
        Treina modelo XGBoost em tempo real - com 8 features corretas
        """
        try:
            if not XGBOOST_AVAILABLE:
                return None
            
            logger.info("[BRAIN] Treinando modelo XGBoost...")
            
            # CRIA FEATURES FALTANTES AUTOMATICAMENTE
            df = features_df.copy()
            
            # Calcula indicadores básicos se não existem
            if 'rsi' not in df.columns:
                df = self._calculate_basic_indicators(df)
            
            if 'volatility' not in df.columns:
                df['volatility'] = df['close'].rolling(20).std().fillna(0)
            
            if 'sma_20' not in df.columns:
                df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
            
            if 'price_change_1' not in df.columns:
                df['price_change_1'] = df['close'].pct_change().fillna(0)
            
            # Usa as 8 features do Central Feature Engine
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) < 6:  # Reduzido para 6 features mínimas
                logger.error(f"[ERROR] Features insuficientes para treinar XGBoost: {available_columns}")
                return None
            
            # Cria target se não existe (para dados de teste)
            if 'target_return' not in df.columns:
                df['target_return'] = df['close'].pct_change().shift(-1).fillna(0)
            
            X = df[available_columns].fillna(0)
            y = df['target_return'].fillna(0)
            
            if len(X) < 20:  # Reduzido para 20 amostras mínimas
                logger.warning(f"[WARN] Dados insuficientes para treinar XGBoost: {len(X)} amostras")
                return None
            
            # Modelo XGBoost otimizado para trading
            model = xgb.XGBRegressor(
                n_estimators=50,  # Reduzido para treinamento mais rápido
                max_depth=4,      # Reduzido para evitar overfitting
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X, y)
            
            # Salva modelo
            joblib.dump(model, os.path.join(self.models_path, 'xgboost_model.joblib'))
            self.models['xgboost'] = model
            
            logger.info(f"[OK] Modelo XGBoost treinado com {len(available_columns)} features")
            return model
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no treinamento XGBoost: {e}")
            return None

    def _train_random_forest_model(self, features_df):
        """
        Treina modelo Random Forest como fallback - com 8 features corretas
        """
        try:
            logger.info("[BRAIN] Treinando modelo Random Forest...")
            
            # CRIA FEATURES FALTANTES AUTOMATICAMENTE
            df = features_df.copy()
            
            # Calcula indicadores básicos se não existem
            if 'rsi' not in df.columns:
                df = self._calculate_basic_indicators(df)
            
            if 'volatility' not in df.columns:
                df['volatility'] = df['close'].rolling(20).std().fillna(0)
            
            if 'sma_20' not in df.columns:
                df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
            
            if 'price_change_1' not in df.columns:
                df['price_change_1'] = df['close'].pct_change().fillna(0)
            
            # Usa as 8 features do Central Feature Engine
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility', 'sma_20']
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) < 6:  # Reduzido para 6 features mínimas
                logger.error(f"[ERROR] Features insuficientes para treinar Random Forest: {available_columns}")
                return None
            
            # Cria target se não existe (para dados de teste)
            if 'target_return' not in df.columns:
                df['target_return'] = df['close'].pct_change().shift(-1).fillna(0)
            
            X = df[available_columns].fillna(0)
            y = df['target_return'].fillna(0)
            
            if len(X) < 15:  # Reduzido para 15 amostras mínimas
                logger.warning(f"[WARN] Dados insuficientes para treinar Random Forest: {len(X)} amostras")
                return None
            
            # Modelo Random Forest otimizado
            model = RandomForestRegressor(
                n_estimators=25,  # Reduzido para treinamento mais rápido
                max_depth=6,      # Reduzido para evitar overfitting
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            # Salva modelo
            joblib.dump(model, os.path.join(self.models_path, 'rf_model.joblib'))
            self.models['rf'] = model
            
            logger.info(f"[OK] Modelo Random Forest treinado com {len(available_columns)} features")
            return model
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no treinamento Random Forest: {e}")
            return None

    def _is_cache_valid(self, cache_key):
        """
        Verifica se o cache é válido
        """
        if cache_key not in self.prediction_cache:
            return False
        return (datetime.now().timestamp() - self.prediction_cache[cache_key].get("timestamp", 0)) < self.cache_timeout

    def get_signal_strength(self, price_data):
        """
        Retorna força do sinal de trading baseado em previsões
        
        Returns:
            float: Força do sinal (-1 a 1)
        """
        prediction = self.predict_next_move(price_data)
        signal = prediction["prediction"] * prediction["confidence"]
        
        logger.info(f"[SIGNAL] Força do sinal de previsão: {signal:.3f}")
        return signal

    def get_model_performance(self):
        """
        Retorna performance dos modelos carregados
        """
        performance = {}
        for model_name in self.models.keys():
            performance[model_name] = {
                "loaded": True,
                "weight": self.ensemble_weights.get(model_name, 0),
                "available": True
            }
        
        performance["tensorflow_available"] = TENSORFLOW_AVAILABLE
        performance["xgboost_available"] = XGBOOST_AVAILABLE
        performance["prophet_available"] = PROPHET_AVAILABLE
        
        return performance

    def _prepare_features(self, market_data):
        """
        Prepara features dos dados de mercado para predição
        """
        try:
            # Extrai dados do market_data
            price_data = market_data.get('price_data', pd.DataFrame())
            
            if price_data.empty or len(price_data) < 10:
                logger.warning("[WARN] Dados insuficientes para features")
                return None
                
            # USA A FUNÇÃO CORRETA PARA 8 FEATURES
            features_df = self._prepare_basic_features(price_data)
            
            if features_df.empty:
                logger.warning("[WARN] Falha ao preparar features compatíveis")
                return None
                
            return features_df
            
        except Exception as e:
            logger.error(f"[ERROR] Erro preparando features: {e}")
            return None
    
    def _fallback_prediction(self, market_data):
        """
        Predição de fallback usando análise técnica simples
        """
        try:
            current_price = market_data.get('current_price', 0)
            price_data = market_data.get('price_data', pd.DataFrame())
            
            if price_data.empty:
                return {
                    'direction': 'hold',
                    'confidence': 0.1,
                    'signal_strength': 0.1,
                    'predicted_price': current_price,
                    'current_price': current_price,
                    'models_used': 0,
                    'predictions': {},
                    'price_change_pct': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Análise de tendência simples
            closes = price_data['close'].values
            if len(closes) >= 5:
                recent_avg = np.mean(closes[-5:])
                older_avg = np.mean(closes[-10:-5]) if len(closes) >= 10 else recent_avg
                
                trend = (recent_avg - older_avg) / older_avg
                
                if trend > 0.01:  # +1% de tendência
                    direction = "buy"
                elif trend < -0.01:  # -1% de tendência
                    direction = "sell"
                else:
                    direction = "hold"
                    
                return {
                    'direction': direction,
                    'confidence': 0.3,
                    'signal_strength': min(abs(trend) * 10, 1.0),
                    'predicted_price': current_price * (1 + trend),
                    'current_price': current_price,
                    'models_used': 0,
                    'predictions': {'technical': trend},
                    'price_change_pct': trend * 100,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'direction': 'hold',
                    'confidence': 0.1,
                    'signal_strength': 0.1,
                    'predicted_price': current_price,
                    'current_price': current_price,
                    'models_used': 0,
                    'predictions': {},
                    'price_change_pct': 0,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Erro fallback prediction: {e}")
            return {
                'direction': 'hold',
                'confidence': 0.1,
                'signal_strength': 0.1,
                'predicted_price': current_price,
                'current_price': current_price,
                'models_used': 0,
                'predictions': {},
                'price_change_pct': 0,
                'timestamp': datetime.now().isoformat()
            }

    def _load_config(self, config_path):
        """
        Carrega configurações do agente
        
        Args:
            config_path (str): Caminho para arquivo de configuração
            
        Returns:
            dict: Configurações carregadas
        """
        try:
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.debug(f"[CONFIG] Configurações carregadas de {config_path}")
                    return config.get('prediction_agent', {})
            else:
                logger.debug(f"[CONFIG] Arquivo {config_path} não encontrado, usando padrões")
                return {}
        except Exception as e:
            logger.warning(f"[WARN] Erro carregando config: {e}")
            return {}

    def predict_next_move(self, price_data):
        """
        Método de compatibilidade para predict - usado pelo DecisionAgent
        
        Args:
            price_data (pd.DataFrame): Dados históricos de preço
            
        Returns:
            dict: Previsão do próximo movimento
        """
        try:
            # Converte price_data para formato market_data
            if isinstance(price_data, pd.DataFrame) and not price_data.empty:
                current_price = price_data['close'].iloc[-1] if 'close' in price_data.columns else 0
                market_data = {
                    'current_price': current_price,
                    'price_data': price_data
                }
            else:
                # Fallback se dados são insuficientes
                market_data = {
                    'current_price': 0,
                    'price_data': pd.DataFrame()
                }
            
            # Usa o método predict existente
            prediction = self.predict(market_data)
            
            # Converte para formato esperado pelo DecisionAgent
            return {
                'prediction': prediction.get('signal_strength', 0),
                'confidence': prediction.get('confidence', 0),
                'direction': prediction.get('direction', 'hold'),
                'predicted_price': prediction.get('predicted_price', 0),
                'models_used': prediction.get('models_used', 0)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro em predict_next_move: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'direction': 'hold',
                'predicted_price': 0,
                'models_used': 0
            }

    def get_price_prediction(self, data):
        """
        Método legado - REFATORADO PARA USAR CENTRAL FEATURE ENGINE
        """
        if not self.models:
            return 0.5
        
        try:
            # 🚀 USA CENTRAL FEATURE ENGINE
            if isinstance(data, pd.DataFrame) and not data.empty:
                features_df = self.central_feature_engine.get_all_features(data, 'basic')
                
                if not features_df.empty:
                    # Obtém predições de todos os modelos
                    predictions = {}
                    
                    if 'lstm' in self.models:
                        try:
                            predictions['lstm'] = self._lstm_prediction(features_df)
                        except Exception as e:
                            logger.warning(f"[WARN] Erro no LSTM: {e}")
                    
                    if 'xgboost' in self.models:
                        try:
                            predictions['xgboost'] = self._xgboost_prediction(features_df)
                        except Exception as e:
                            logger.warning(f"[WARN] Erro no XGBoost: {e}")
                    
                    if 'rf' in self.models:
                        try:
                            predictions['rf'] = self._random_forest_prediction(features_df)
                        except Exception as e:
                            logger.warning(f"[WARN] Erro no Random Forest: {e}")
                    
                    # Combina predições usando ensemble
                    if predictions:
                        final_prediction = self._ensemble_predictions(predictions)
                        if isinstance(final_prediction, dict):
                            prediction_value = final_prediction.get('prediction', 0.0)
                        else:
                            prediction_value = final_prediction
                        logger.debug(f"[PRED] Predição via Central Engine: {prediction_value:.3f}")
                        return prediction_value
            
            # Fallback para método antigo
            if isinstance(data, pd.DataFrame):
                market_data = {
                    'current_price': data['close'].iloc[-1] if 'close' in data.columns else 0,
                    'price_data': data
                }
            else:
                market_data = {'price_data': pd.DataFrame()}
            
            result = self.predict(market_data)
            return result.get('signal_strength', 0.5)
            
        except Exception as e:
            logger.error(f"[ERROR] Erro em get_price_prediction via Central Engine: {str(e)}")
            return 0.5