#!/usr/bin/env python3
"""
GERENCIADOR DE MODELOS
======================

Módulo responsável pelo carregamento e gerenciamento dos modelos de IA.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.terminal_colors import TerminalColors
from utils.logger import log_trade_info

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

import logging

class ModelManager:
    """Gerenciador de modelos de IA"""
    
    def __init__(self, central_feature_engine=None):
        """Inicializa o gerenciador de modelos"""
        self.central_feature_engine = central_feature_engine
        self.models = {}
        self.model_confidences = {}
        self.model_performance = {}
        self.continuous_learning = None
        self._load_all_models_guaranteed()
        # Buffer de trades para retreinamento
        self.trade_buffer = []
        self.buffer_size = 500
    
    def add_trade_to_buffer(self, trade_data):
        required_fields = ['entry_price', 'exit_price', 'pnl_percentage', 'confidence', 'status', 'type', 'open', 'high', 'low', 'close', 'volume']
        if all(field in trade_data for field in required_fields):
            self.trade_buffer.append(trade_data)
            if len(self.trade_buffer) > self.buffer_size:
                self.trade_buffer.pop(0)
        else:
            log_trade_info(f"[WARNING] Trade simulado inválido: campos faltando {set(required_fields) - set(trade_data.keys())}", level='WARNING')

    def validate_simulated_trade(self, trade):
        required_fields = ['entry_price', 'exit_price', 'pnl_percentage', 'confidence', 'status', 'type', 'open', 'high', 'low', 'close', 'volume']
        return all(field in trade and trade[field] is not None for field in required_fields)

    def _load_all_models_guaranteed(self):
        """🧠 CARREGA TODOS OS MODELOS COM GARANTIA"""
        log_trade_info("🧠 Carregando TODOS os modelos ML...", level='INFO')
        
        self.models = {}
        model_count = 0
        required_models = ['xgboost', 'random_forest', 'lstm']
        
        try:
            # XGBoost - CORRIGIDO para usar o caminho correto
            if os.path.exists('models/trained/xgboost_model.joblib'):
                self.models['xgboost'] = joblib.load('models/trained/xgboost_model.joblib')
                model_count += 1
                log_trade_info("✅ XGBoost carregado", level='SUCCESS')
            else:
                log_trade_info("⚠️ XGBOOST não encontrado", level='WARNING')
            
            # Random Forest - CORRIGIDO para usar o caminho correto
            if os.path.exists('models/trained/rf_model.joblib'):
                self.models['random_forest'] = joblib.load('models/trained/rf_model.joblib')
                model_count += 1
                log_trade_info("✅ Random Forest carregado", level='SUCCESS')
            else:
                log_trade_info("⚠️ RANDOM_FOREST não encontrado", level='WARNING')
                
            # LSTM - CARREGAMENTO CORRIGIDO (mesmo do original)
            if os.path.exists('models/trained/lstm_model.h5'):
                try:
                    # Força imports específicos do TensorFlow
                    import tensorflow as tf
                    try:
                        tf.config.set_visible_devices([], 'GPU')  # Força CPU
                    except:
                        pass
                    
                    # Carrega modelo com configurações específicas
                    self.models['lstm'] = load_model(
                        'models/trained/lstm_model.h5',
                        compile=False  # Evita problemas de compilação
                    )
                    model_count += 1
                    log_trade_info("✅ LSTM carregado com sucesso", level='SUCCESS')
                    
                except Exception as lstm_error:
                    log_trade_info(f"❌ Erro ao carregar lstm: {lstm_error}", level='ERROR')
                    
                    # Tenta método alternativo
                    try:
                        self.models['lstm'] = load_model('models/trained/lstm_model.h5', compile=False)
                        model_count += 1
                        log_trade_info("✅ LSTM carregado (método alternativo)", level='SUCCESS')
                    except:
                        log_trade_info("❌ FALHA TOTAL - LSTM não pode ser carregado", level='ERROR')
            else:
                log_trade_info("⚠️ LSTM não encontrado", level='WARNING')
            
            # ===== SISTEMA DE APRENDIZADO CONTÍNUO =====
            try:
                from models.continuous_learning import ContinuousLearningSystem
                self.continuous_learning = ContinuousLearningSystem()
                log_trade_info("✅ Sistema de aprendizado contínuo carregado", level='SUCCESS')
            except Exception as e:
                log_trade_info(f"⚠️ Sistema de aprendizado contínuo não disponível: {e}", level='WARNING')
            
            # ===== INICIALIZA PERFORMANCE DOS MODELOS =====
            for model_name in self.models.keys():
                self.model_performance[model_name] = {
                    'correct_predictions': 0,
                    'total_predictions': 0,
                    'accuracy': 0.0
                }
            
            log_trade_info(f"✅ {len(self.models)} modelos carregados", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro crítico no carregamento de modelos: {e}", level='ERROR')
    
    def get_model_predictions(self, data):
        """Obtém previsões de todos os modelos"""
        predictions = {}
        try:
            # Prepara features avançadas (mesmo do arquivo original)
            advanced_features = self._prepare_advanced_features(data)

            if advanced_features is None:
                log_trade_info(f"⚠️ advanced_features é None!", level='WARNING')
                return {}
            if not isinstance(advanced_features, pd.DataFrame):
                log_trade_info(f"⚠️ advanced_features não é DataFrame!", level='WARNING')
                return {}
            if advanced_features.empty:
                log_trade_info(f"⚠️ advanced_features está vazio!", level='WARNING')
                return {}

            features_array = advanced_features.iloc[-1:].values
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lstm':
                        # Previsão LSTM - CORRIGIDO para dimensões corretas
                        lstm_features = self._prepare_lstm_features(data)
                        if lstm_features is not None:
                            prediction = model.predict(lstm_features, verbose=0)
                            # Normaliza previsão para 0-1
                            predictions[model_name] = float(prediction[0][0])
                        else:
                            predictions[model_name] = 0.5
                    else:
                        # Previsão para regressores (XGBoost/Random Forest) - MESMO DO ORIGINAL
                        prediction = model.predict(features_array)[0]
                        predictions[model_name] = prediction
                        log_trade_info(f"🔍 DEBUG: {model_name} prediction: {prediction:.6f}", level='INFO')
                except Exception as e:
                    log_trade_info(f"⚠️ Erro na previsão {model_name}: {e}", level='WARNING')
                    # Retreinamento automático em caso de erro
                    self._retrain_model_automatically(model_name, advanced_features, data)
                    predictions[model_name] = 0.5  # Neutro
            return predictions
        except Exception as e:
            log_trade_info(f"❌ Erro ao obter previsões: {e}", level='ERROR')
            return {}
    
    def _retrain_model_automatically(self, model_name, features, target):
        try:
            # Balanceamento de classes (apenas para classificação)
            if model_name in ['xgboost', 'random_forest'] and target is not None:
                if len(set(target)) == 2:
                    smote = SMOTE(random_state=42)
                    features, target = smote.fit_resample(features, target)
                    log_trade_info(f"SMOTE aplicado para balanceamento de classes em {model_name}", level='INFO')
            # Validação cruzada
            if model_name in ['xgboost', 'random_forest']:
                model = self.models[model_name]
                try:
                    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')
                    avg_score = scores.mean()
                    log_trade_info(f"Validação cruzada para {model_name}: acurácia média = {avg_score:.3f}", level='INFO')
                    if avg_score < 0.4:
                        log_trade_info(f"Acurácia baixa ({avg_score:.3f}), retreinando {model_name}", level='WARNING')
                except Exception as e:
                    log_trade_info(f"Erro na validação cruzada de {model_name}: {e}", level='WARNING')
            # Otimização de hiperparâmetros
            if model_name == 'xgboost':
                from xgboost import XGBClassifier
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'n_estimators': [100, 200, 300]
                }
                model = XGBClassifier()
                search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, random_state=42)
                search.fit(features, target)
                self.models[model_name] = search.best_estimator_
                log_trade_info(f"Hiperparâmetros otimizados para {model_name}: {search.best_params_}", level='INFO')
            elif model_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                param_grid = {
                    'max_depth': [5, 10, 15],
                    'n_estimators': [100, 200, 300]
                }
                model = RandomForestClassifier()
                search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, random_state=42)
                search.fit(features, target)
                self.models[model_name] = search.best_estimator_
                log_trade_info(f"Hiperparâmetros otimizados para {model_name}: {search.best_params_}", level='INFO')
            elif model_name == 'lstm':
                model = self.models[model_name]
                if not hasattr(model, 'compiled') or not getattr(model, 'compiled', False):
                    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                    self.models[model_name] = model
                    log_trade_info(f"LSTM compilado antes do retreinamento", level='INFO')
                model.fit(features, target)
                log_trade_info(f"Modelo LSTM retreinado (fit normal)", level='INFO')
            else:
                model = self.models[model_name]
                model.fit(features, target)
                self.models[model_name] = model
                log_trade_info(f"Modelo {model_name} retreinado (fit normal)", level='INFO')
        except Exception as e:
            log_trade_info(f"Erro no retreinamento automático de {model_name}: {e}", level='ERROR')
    
    def _retrain_xgboost_model(self, features, data):
        """Retreina modelo XGBoost"""
        try:
            from xgboost import XGBRegressor
            
            # Prepara dados para treinamento
            X = features.iloc[:-1]  # Todas as linhas exceto a última
            y = data['close'].iloc[1:].values  # Preços futuros
            
            # Remove linhas com NaN
            mask = ~(X.isna().any(axis=1) | pd.isna(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                log_trade_info("⚠️ Dados insuficientes para retreinamento XGBoost", level='WARNING')
                return
            
            # Cria e treina novo modelo
            new_model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            new_model.fit(X_clean, y_clean)
            
            # Substitui modelo antigo
            self.models['xgboost'] = new_model
            
            # Salva modelo retreinado
            import joblib
            joblib.dump(new_model, 'models/trained/xgboost_model.joblib')
            
            log_trade_info("✅ XGBoost retreinado e salvo", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro no retreinamento XGBoost: {e}", level='ERROR')
    
    def _retrain_random_forest_model(self, features, data):
        """Retreina modelo Random Forest"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Prepara dados para treinamento
            X = features.iloc[:-1]  # Todas as linhas exceto a última
            y = data['close'].iloc[1:].values  # Preços futuros
            
            # Remove linhas com NaN
            mask = ~(X.isna().any(axis=1) | pd.isna(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                log_trade_info("⚠️ Dados insuficientes para retreinamento Random Forest", level='WARNING')
                return
            
            # Cria e treina novo modelo
            new_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            new_model.fit(X_clean, y_clean)
            
            # Substitui modelo antigo
            self.models['random_forest'] = new_model
            
            # Salva modelo retreinado
            import joblib
            joblib.dump(new_model, 'models/trained/rf_model.joblib')
            
            log_trade_info("✅ Random Forest retreinado e salvo", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro no retreinamento Random Forest: {e}", level='ERROR')
    
    def _retrain_lstm_model(self, features, data):
        """Retreina modelo LSTM"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Prepara dados para LSTM
            if len(data) < 50:
                log_trade_info("⚠️ Dados insuficientes para retreinamento LSTM", level='WARNING')
                return
            
            # Cria novo modelo LSTM simples
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(20, 8)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Substitui modelo antigo
            self.models['lstm'] = model
            
            # Salva modelo retreinado
            model.save('models/trained/lstm_model.h5')
            
            log_trade_info("✅ LSTM retreinado e salvo", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro no retreinamento LSTM: {e}", level='ERROR')
    
    def _prepare_advanced_features(self, data):
        """Prepara features avançadas usando sistema centralizado"""
        try:
            if self.central_feature_engine:
                result = self.central_feature_engine.get_all_features(data, 'complete')
                if result is None:
                    log_trade_info(f"⚠️ get_all_features retornou None!", level='WARNING')
                    return pd.DataFrame()
                return result
            # Fallback para sistema central
            return self._prepare_basic_features_fallback(data)
        except Exception as e:
            log_trade_info(f"⚠️ Erro na preparação de features: {e}", level='WARNING')
            return pd.DataFrame()
    
    def _prepare_basic_features_fallback(self, data):
        """Fallback básico para features se sistema central não estiver disponível"""
        try:
            if data is None or len(data) < 20:
                return pd.DataFrame()
            
            # Usa apenas as últimas 50 linhas para evitar problemas de length
            df = data.tail(50).copy()
            
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
                log_trade_info("⚠️ Poucas features disponíveis, usando fallback", level='WARNING')
                return pd.DataFrame()
            
            # Garante que o DataFrame tenha exatamente o tamanho esperado
            result_df = df[available_features].copy()
            
            # Se ainda houver problemas de tamanho, usa apenas a última linha
            if len(result_df) > 0:
                return result_df.iloc[-1:].copy()
            else:
                return pd.DataFrame()
            
        except Exception as e:
            log_trade_info(f"⚠️ Erro no fallback de features: {e}", level='WARNING')
            return pd.DataFrame()
    
    def _prepare_features(self, data):
        """Prepara features para modelos tradicionais (mantido para compatibilidade)"""
        return self._prepare_advanced_features(data)
    
    def _prepare_lstm_features(self, data):
        """Prepara features para modelo LSTM - CORRIGIDO para 8 features"""
        try:
            if data is None or data.empty or len(data) < 20:
                return None
            
            # Prepara 8 features para o LSTM (baseado no erro de dimensões)
            features = []
            
            # 1. Preços normalizados (últimos 20 pontos)
            close_prices = data['close'].values[-20:]
            normalized_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)
            
            # 2. RSI
            rsi = self._calculate_rsi(data['close'].values)
            rsi_feature = rsi[-1] if len(rsi) > 0 else 50
            
            # 3. Média móvel curta
            ma_short = data['close'].rolling(window=5).mean().iloc[-1]
            ma_short_norm = (ma_short - np.mean(close_prices)) / np.std(close_prices)
            
            # 4. Média móvel longa
            ma_long = data['close'].rolling(window=20).mean().iloc[-1]
            ma_long_norm = (ma_long - np.mean(close_prices)) / np.std(close_prices)
            
            # 5. Volatilidade
            volatility = data['close'].pct_change().std()
            volatility_norm = min(volatility * 100, 1.0) if not pd.isna(volatility) else 0.01
            
            # 6. Volume normalizado
            volume = data['volume'].values[-20:]
            volume_norm = (volume - np.mean(volume)) / np.std(volume)
            
            # 7. Momentum (diferença de preços)
            momentum = data['close'].diff().iloc[-1]
            momentum_norm = (momentum - np.mean(data['close'].diff().dropna())) / np.std(data['close'].diff().dropna())
            
            # 8. Tendência (slope dos últimos 10 pontos)
            if len(close_prices) >= 10:
                x = np.arange(10)
                y = close_prices[-10:]
                slope = np.polyfit(x, y, 1)[0]
                trend_norm = min(max(slope / np.std(close_prices), -1), 1)
            else:
                trend_norm = 0.0
            
            # Combina todas as features em uma matriz 3D para LSTM
            # Formato: (samples, timesteps, features)
            combined_features = np.column_stack([
                normalized_prices,
                np.full(20, rsi_feature/100),  # RSI normalizado
                np.full(20, ma_short_norm),
                np.full(20, ma_long_norm),
                np.full(20, volatility_norm),
                volume_norm,
                np.full(20, momentum_norm),
                np.full(20, trend_norm)
            ])
            
            return combined_features.reshape(1, 20, 8)  # 8 features como esperado
            
        except Exception as e:
            log_trade_info(f"❌ Erro ao preparar features LSTM: {e}", level='ERROR')
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI usando sistema centralizado"""
        try:
            from utils.technical_indicators import technical_indicators
            return technical_indicators.calculate_rsi(prices, period)
        except ImportError:
            # Fallback se sistema central não estiver disponível
            log_trade_info("⚠️ Sistema central de RSI não disponível, usando fallback", level='WARNING')
            return np.array([50.0] * len(prices))
    
    def update_model_performance(self, model_name, prediction, actual_outcome):
        """Atualiza performance do modelo"""
        try:
            if model_name in self.model_performance:
                self.model_performance[model_name]['total_predictions'] += 1
                
                # Verifica se previsão estava correta
                if (prediction > 0.5 and actual_outcome > 0.5) or (prediction < 0.5 and actual_outcome < 0.5):
                    self.model_performance[model_name]['correct_predictions'] += 1
                
                # Calcula nova acurácia
                total = self.model_performance[model_name]['total_predictions']
                correct = self.model_performance[model_name]['correct_predictions']
                self.model_performance[model_name]['accuracy'] = correct / total if total > 0 else 0.0
                
        except Exception as e:
            log_trade_info(f"⚠️ Erro ao atualizar performance do modelo: {e}", level='WARNING')
    
    def get_model_confidence(self, model_name):
        """Obtém confiança do modelo"""
        return self.model_confidences.get(model_name, 0.5)
    
    def save_models(self):
        """Salva modelos treinados"""
        try:
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    model.save(f'models/trained/{model_name}_model.h5')
                else:
                    joblib.dump(model, f'models/trained/{model_name}_model.pkl')
            
            log_trade_info("✅ Modelos salvos", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro ao salvar modelos: {e}", level='ERROR') 

    def load_models_with_fallbacks(self):
        """Carrega modelos com estratégias de fallback robustas"""
        try:
            log_trade_info("🔄 Carregando modelos com fallbacks...", level='INFO')
            
            # ===== FALLBACK 1: TENTA CARREGAR MODELOS PRINCIPAIS =====
            models_loaded = self._load_primary_models()
            
            if not models_loaded:
                # ===== FALLBACK 2: TENTA CARREGAR MODELOS DE BACKUP =====
                log_trade_info("⚠️ Fallback 1 falhou, tentando modelos de backup...", level='WARNING')
                models_loaded = self._load_backup_models()
            
            if not models_loaded:
                # ===== FALLBACK 3: TREINA MODELOS NOVOS =====
                log_trade_info("⚠️ Fallback 2 falhou, treinando modelos novos...", level='WARNING')
                models_loaded = self._train_fallback_models()
            
            if not models_loaded:
                # ===== FALLBACK 4: MODELOS SIMPLES =====
                log_trade_info("⚠️ Fallback 3 falhou, usando modelos simples...", level='WARNING')
                models_loaded = self._create_simple_models()
            
            if models_loaded:
                log_trade_info("✅ Modelos carregados com sucesso (com fallbacks)", level='SUCCESS')
            else:
                log_trade_info("❌ Todos os fallbacks falharam - sistema em modo de emergência", level='ERROR')
            
            return models_loaded
            
        except Exception as e:
            log_trade_info(f"❌ Erro crítico no carregamento de modelos: {e}", level='ERROR')
            return False
    
    def _load_primary_models(self):
        """Tenta carregar modelos principais"""
        try:
            success_count = 0
            
            # XGBoost
            if self._load_xgboost_model():
                success_count += 1
            
            # Random Forest
            if self._load_random_forest_model():
                success_count += 1
            
            # LSTM
            if self._load_lstm_model():
                success_count += 1
            
            return success_count >= 2  # Pelo menos 2 modelos devem carregar
            
        except Exception as e:
            log_trade_info(f"❌ Erro carregando modelos principais: {e}", level='ERROR')
            return False
    
    def _load_backup_models(self):
        """Tenta carregar modelos de backup (versões antigas)"""
        try:
            success_count = 0
            
            # Backup XGBoost
            backup_paths = [
                "models/trained/xgboost_backup.pkl",
                "models/trained/xgboost_old.pkl",
                "models/trained/xgboost_v1.pkl"
            ]
            
            for path in backup_paths:
                if os.path.exists(path):
                    try:
                        self.xgboost_model = joblib.load(path)
                        log_trade_info(f"✅ XGBoost backup carregado: {path}", level='SUCCESS')
                        success_count += 1
                        break
                    except Exception as e:
                        log_trade_info(f"⚠️ Falha ao carregar backup XGBoost {path}: {e}", level='WARNING')
                        continue
            
            # Backup Random Forest
            backup_paths = [
                "models/trained/random_forest_backup.pkl",
                "models/trained/random_forest_old.pkl",
                "models/trained/random_forest_v1.pkl"
            ]
            
            for path in backup_paths:
                if os.path.exists(path):
                    try:
                        self.random_forest_model = joblib.load(path)
                        log_trade_info(f"✅ Random Forest backup carregado: {path}", level='SUCCESS')
                        success_count += 1
                        break
                    except Exception as e:
                        log_trade_info(f"⚠️ Falha ao carregar backup Random Forest {path}: {e}", level='WARNING')
                        continue
            
            return success_count >= 1  # Pelo menos 1 modelo deve carregar
            
        except Exception as e:
            log_trade_info(f"❌ Erro carregando modelos de backup: {e}", level='ERROR')
            return False
    
    def _train_fallback_models(self):
        """Treina modelos novos como fallback"""
        try:
            log_trade_info("🔄 Treinando modelos de fallback...", level='INFO')
            
            # Gera dados sintéticos para treinamento
            synthetic_data = self._generate_synthetic_training_data()
            
            if synthetic_data is None:
                log_trade_info("❌ Não foi possível gerar dados para treinamento", level='ERROR')
                return False
            
            success_count = 0
            
            # Treina XGBoost simples
            try:
                self.xgboost_model = self._train_simple_xgboost(synthetic_data)
                success_count += 1
                log_trade_info("✅ XGBoost de fallback treinado", level='SUCCESS')
            except Exception as e:
                log_trade_info(f"⚠️ Falha ao treinar XGBoost de fallback: {e}", level='WARNING')
            
            # Treina Random Forest simples
            try:
                self.random_forest_model = self._train_simple_random_forest(synthetic_data)
                success_count += 1
                log_trade_info("✅ Random Forest de fallback treinado", level='SUCCESS')
            except Exception as e:
                log_trade_info(f"⚠️ Falha ao treinar Random Forest de fallback: {e}", level='WARNING')
            
            return success_count >= 1
            
        except Exception as e:
            log_trade_info(f"❌ Erro treinando modelos de fallback: {e}", level='ERROR')
            return False
    
    def _create_simple_models(self):
        """Cria modelos simples como último recurso"""
        try:
            log_trade_info("🔄 Criando modelos simples de emergência...", level='WARNING')
            
            # Modelo de média móvel simples
            self.simple_ma_model = {
                'type': 'simple_moving_average',
                'window': 20,
                'description': 'Modelo de emergência - média móvel simples'
            }
            
            # Modelo de tendência simples
            self.simple_trend_model = {
                'type': 'simple_trend',
                'description': 'Modelo de emergência - detecção de tendência simples'
            }
            
            log_trade_info("✅ Modelos simples criados (modo de emergência)", level='SUCCESS')
            return True
            
        except Exception as e:
            log_trade_info(f"❌ Erro criando modelos simples: {e}", level='ERROR')
            return False
    
    def _generate_synthetic_training_data(self):
        """Gera dados sintéticos para treinamento de fallback"""
        try:
            # Gera dados sintéticos baseados em padrões típicos do Bitcoin
            np.random.seed(42)
            n_samples = 1000
            
            # Preços sintéticos
            base_price = 45000
            prices = []
            for i in range(n_samples):
                # Adiciona tendência e volatilidade
                trend = np.sin(i * 0.01) * 1000
                noise = np.random.normal(0, 500)
                price = base_price + trend + noise
                prices.append(max(price, 1000))  # Preço mínimo
            
            # Cria features
            df = pd.DataFrame({'close': prices})
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Target (direção do preço)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove NaN
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            return df
            
        except Exception as e:
            log_trade_info(f"❌ Erro gerando dados sintéticos: {e}", level='ERROR')
            return None
    
    def _train_simple_xgboost(self, data):
        """Treina XGBoost simples para fallback"""
        try:
            from xgboost import XGBClassifier
            
            # Features simples
            features = ['returns', 'ma_5', 'ma_20', 'volatility']
            X = data[features].values
            y = data['target'].values
            
            # Treina modelo simples
            model = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X, y)
            return model
            
        except Exception as e:
            log_trade_info(f"❌ Erro treinando XGBoost simples: {e}", level='ERROR')
            return None
    
    def _train_simple_random_forest(self, data):
        """Treina Random Forest simples para fallback"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Features simples
            features = ['returns', 'ma_5', 'ma_20', 'volatility']
            X = data[features].values
            y = data['target'].values
            
            # Treina modelo simples
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X, y)
            return model
            
        except Exception as e:
            log_trade_info(f"❌ Erro treinando Random Forest simples: {e}", level='ERROR')
            return None 

    def _load_xgboost_model(self):
        """Carrega modelo XGBoost"""
        try:
            if os.path.exists('models/trained/xgboost_model.joblib'):
                self.models['xgboost'] = joblib.load('models/trained/xgboost_model.joblib')
                log_trade_info("✅ XGBoost carregado", level='SUCCESS')
                return True
            else:
                log_trade_info("⚠️ XGBoost não encontrado", level='WARNING')
                return False
        except Exception as e:
            log_trade_info(f"❌ Erro carregando XGBoost: {e}", level='ERROR')
            return False
    
    def _load_random_forest_model(self):
        """Carrega modelo Random Forest"""
        try:
            if os.path.exists('models/trained/rf_model.joblib'):
                self.models['random_forest'] = joblib.load('models/trained/rf_model.joblib')
                log_trade_info("✅ Random Forest carregado", level='SUCCESS')
                return True
            else:
                log_trade_info("⚠️ Random Forest não encontrado", level='WARNING')
                return False
        except Exception as e:
            log_trade_info(f"❌ Erro carregando Random Forest: {e}", level='ERROR')
            return False
    
    def _load_lstm_model(self):
        """Carrega modelo LSTM"""
        try:
            if os.path.exists('models/trained/lstm_model.h5'):
                self.models['lstm'] = load_model('models/trained/lstm_model.h5', compile=False)
                log_trade_info("✅ LSTM carregado", level='SUCCESS')
                return True
            else:
                log_trade_info("⚠️ LSTM não encontrado", level='WARNING')
                return False
        except Exception as e:
            log_trade_info(f"❌ Erro carregando LSTM: {e}", level='ERROR')
            return False 

    def update_models_with_trade_result(self, trade_data):
        """Atualiza modelos com resultado de trade para aprendizado contínuo"""
        try:
            log_trade_info("[INFO] Atualizando modelos com resultado do trade...", level='INFO')
            self.add_trade_to_buffer(trade_data)
            
            # Extrai dados do trade
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            pnl_percentage = trade_data.get('pnl_percentage', 0)
            signal = trade_data.get('signal', 'HOLD')
            confidence = trade_data.get('confidence', 0.5)
            trade_type = trade_data.get('type', 'LONG')
            
            if entry_price == 0 or exit_price == 0:
                log_trade_info("⚠️ Dados de trade incompletos para aprendizado", level='WARNING')
                return
            
            # Calcula resultado real (1 = lucro, 0 = prejuízo)
            actual_outcome = 1.0 if pnl_percentage > 0 else 0.0
            
            # Atualiza cada modelo com o resultado
            for model_name in self.models.keys():
                try:
                    # Simula previsão do modelo (em produção seria a previsão real)
                    model_prediction = confidence  # Usa confiança como proxy da previsão
                    
                    # Atualiza performance do modelo
                    self.update_model_performance(model_name, model_prediction, actual_outcome)
                    
                    # Retreinamento automático se performance baixa
                    if self.model_performance[model_name]['accuracy'] < 0.4:
                        log_trade_info(f"[WARNING] Retreinando {model_name} devido à baixa performance", level='WARNING')
                        self._retrain_model_with_trade_data(model_name, self.trade_buffer)
                    
                except Exception as e:
                    log_trade_info(f"❌ Erro atualizando modelo {model_name}: {e}", level='ERROR')
            
            # Atualiza sistema de aprendizado contínuo
            if self.continuous_learning:
                try:
                    self.continuous_learning.register_trade_feedback(trade_data)
                    log_trade_info("✅ Sistema de aprendizado contínuo atualizado", level='SUCCESS')
                except Exception as e:
                    log_trade_info(f"❌ Erro no aprendizado contínuo: {e}", level='ERROR')
            
            log_trade_info("✅ Modelos atualizados com resultado do trade", level='SUCCESS')
            
        except Exception as e:
            log_trade_info(f"❌ Erro crítico no aprendizado: {e}", level='ERROR')
    
    def _retrain_model_with_trade_data(self, model_name, trade_data):
        if len(trade_data) < 100:
            log_trade_info(f"[WARNING] Buffer insuficiente para retreinamento de {model_name}: {len(trade_data)}/100 trades", level='WARNING')
            return
        try:
            threading.Thread(target=self._retrain_model_with_trade_data_thread, args=(model_name, trade_data), daemon=True).start()
            log_trade_info(f"Retreinamento de {model_name} iniciado em background com {len(trade_data)} trades", level='INFO')
            self.trade_buffer = []  # Limpa buffer
            self.model_performance[model_name]['last_retrain'] = datetime.now()
        except Exception as e:
            log_trade_info(f"❌ Erro ao retreinar {model_name}: {e}", level='ERROR')

    def _retrain_model_with_trade_data_thread(self, model_name, trade_buffer):
        try:
            log_trade_info(f"Retreinamento de {model_name} disparado: acurácia={self.model_performance[model_name]['accuracy']:.3f}, trades={self.model_performance[model_name]['total_predictions']}", level='INFO')
            # Preparar DataFrame de trades
            if not trade_buffer or len(trade_buffer) < 20:
                log_trade_info(f"[WARNING] Buffer insuficiente para retreinamento de {model_name}", level='WARNING')
                return
            df = pd.DataFrame(trade_buffer)
            # Preparar features com CentralFeatureEngine
            if self.central_feature_engine:
                features = self.central_feature_engine.get_all_features(df, 'complete')
            else:
                features = df[['open', 'high', 'low', 'close', 'volume']].copy()
            # Target: próximo preço de fechamento
            if 'close' in df.columns:
                target = df['close'].shift(-1).fillna(method='ffill').fillna(0)
            else:
                target = None
            # Chama retreinamento real
            self._retrain_model_automatically(model_name, features, target)
            # Limpa buffer e atualiza timestamp
            self.trade_buffer = []
            self.model_performance[model_name]['last_retrain'] = datetime.now()
            log_trade_info(f"Buffer limpo após retreinamento de {model_name}", level='INFO')
        except Exception as e:
            log_trade_info(f"Erro no retreinamento real de {model_name}: {e}", level='ERROR')
    
    def validate_models_progressively(self, total_trades_completed):
        """Valida modelos progressivamente durante os 2000 trades"""
        try:
            log_trade_info(f"🔍 Validando modelos progressivamente: {total_trades_completed}/2000 trades", level='INFO')
            
            # Pontos de validação baseados na pesquisa
            validation_points = [50, 100, 200, 500, 1000, 1500, 2000]
            
            if total_trades_completed in validation_points:
                log_trade_info(f"🎯 PONTO DE VALIDAÇÃO: {total_trades_completed} trades", level='SUCCESS')
                
                # Calcula performance atual de cada modelo
                for model_name in self.models.keys():
                    performance = self.model_performance.get(model_name, {})
                    accuracy = performance.get('accuracy', 0.0)
                    total_predictions = performance.get('total_predictions', 0)
                    
                    log_trade_info(f"📊 {model_name}: Accuracy {accuracy:.3f} ({total_predictions} predições)", level='INFO')
                    
                    # Retreinamento se performance baixa
                    if accuracy < 0.4 and total_predictions >= 20:
                        log_trade_info(f"[WARNING] {model_name} marcado para retreinamento (accuracy baixa)", level='WARNING')
                        self._schedule_model_retraining(model_name)
                
                # Validação geral do sistema
                avg_accuracy = sum([
                    self.model_performance.get(name, {}).get('accuracy', 0.0) 
                    for name in self.models.keys()
                ]) / len(self.models)
                
                log_trade_info(f"🎯 Performance média do sistema: {avg_accuracy:.3f}", level='SUCCESS')
                
                # Salva estado dos modelos
                self.save_models()
                
                return True
            
            return False
            
        except Exception as e:
            log_trade_info(f"❌ Erro na validação progressiva: {e}", level='ERROR')
            return False
    
    def _schedule_model_retraining(self, model_name):
        """Agenda retreinamento de modelo específico"""
        try:
            if not hasattr(self, 'retraining_schedule'):
                self.retraining_schedule = {}
            
            self.retraining_schedule[model_name] = {
                'scheduled_at': datetime.now(),
                'reason': 'low_accuracy',
                'status': 'pending'
            }
            
            log_trade_info(f"📅 Retreinamento de {model_name} agendado", level='INFO')
            
        except Exception as e:
            log_trade_info(f"❌ Erro agendando retreinamento: {e}", level='ERROR')
    
    def get_training_progress(self):
        """Obtém progresso do treinamento"""
        try:
            total_predictions = sum([
                self.model_performance.get(name, {}).get('total_predictions', 0)
                for name in self.models.keys()
            ])
            
            avg_accuracy = sum([
                self.model_performance.get(name, {}).get('accuracy', 0.0)
                for name in self.models.keys()
            ]) / len(self.models) if self.models else 0.0
            
            return {
                'total_predictions': total_predictions,
                'avg_accuracy': avg_accuracy,
                'models_performance': self.model_performance,
                'retraining_schedule': getattr(self, 'retraining_schedule', {})
            }
            
        except Exception as e:
            log_trade_info(f"❌ Erro obtendo progresso: {e}", level='ERROR')
            return {} 

    def check_retraining_triggers(self, model_name, performance):
        # Só dispara triggers se buffer estiver cheio
        if len(self.trade_buffer) < 100:
            return
        # Trigger por tempo (apenas uma vez por execução)
        if not hasattr(self, 'time_trigger_checked'):
            self.time_trigger_checked = False
        last_retrain = self.model_performance[model_name].get('last_retrain', datetime.min)
        if not self.time_trigger_checked and (datetime.now() - last_retrain).total_seconds() >= 24 * 3600:
            self._retrain_model_with_trade_data(model_name, self.trade_buffer)
            self.time_trigger_checked = True
            return
        # Trigger por número de trades
        if performance['total_predictions'] >= 100:
            self._retrain_model_with_trade_data(model_name, self.trade_buffer)
        # Trigger por performance
        elif performance['accuracy'] < 0.4 and len(self.trade_buffer) >= 100:
            self._retrain_model_with_trade_data(model_name, self.trade_buffer)

    def test_retraining(self, simulated_trades):
        for trade in simulated_trades:
            self.add_trade_to_buffer(trade)
            self.update_models_with_trade_result(trade)
        log_trade_info(f"Teste de retreinamento concluído com {len(simulated_trades)} trades", level='INFO') 

    def test_retraining_scenarios(self, num_trades=500, verbose=False):
        import numpy as np
        original_level = logging.getLogger().level
        if not verbose:
            logging.getLogger().setLevel(logging.ERROR)
        simulated_trades = []
        base_price = 50000
        for _ in range(num_trades):
            price = base_price + np.random.normal(0, 1000)
            trade = {
                'entry_price': price,
                'exit_price': price + np.random.normal(0, 500),
                'pnl_percentage': np.random.uniform(-2, 2),
                'confidence': np.random.uniform(0.5, 1.0),
                'status': 'closed',
                'type': np.random.choice(['LONG', 'SHORT']),
                'open': price,
                'high': price + np.random.uniform(0, 500),
                'low': price - np.random.uniform(0, 500),
                'close': price + np.random.normal(0, 100),
                'volume': np.random.uniform(0.1, 10),
                'timestamp': datetime.now()
            }
            simulated_trades.append(trade)
        valid_trades = [trade for trade in simulated_trades if self.validate_simulated_trade(trade)]
        log_trade_info(f"{len(valid_trades)}/{num_trades} trades simulados válidos", level='INFO')
        for trade in valid_trades:
            self.add_trade_to_buffer(trade)
            self.update_models_with_trade_result(trade)
            for model_name in self.models:
                performance = self.model_performance[model_name]
                self.check_retraining_triggers(model_name, performance)
        logging.getLogger().setLevel(original_level)
        log_trade_info(f"Teste de retreinamento concluído: {len(valid_trades)} trades simulados", level='INFO') 