"""
CONFIGURAÇÃO DE FEATURES DOS MODELOS
===================================

Este arquivo define exatamente as features que foram usadas no treinamento
dos modelos ML para garantir compatibilidade.

IMPORTANTE: Não altere sem retreinar os modelos!
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("model-features")

class ModelFeatures:
    """
    Classe responsável por preparar features compatíveis com os modelos treinados
    """
    
    def __init__(self):
        """
        Define as 8 features originais usadas no treinamento
        """
        # AS 8 FEATURES ORIGINAIS DO TREINAMENTO
        self.original_features = [
            'open',           # Preço de abertura  
            'high',           # Preço máximo
            'low',            # Preço mínimo
            'close',          # Preço de fechamento
            'volume',         # Volume de negociação
            'rsi',            # Relative Strength Index
            'volatility',     # Volatilidade (desvio padrão)
            'price_change_1'  # Mudança de preço do período anterior
        ]
        
        logger.info(f"[OK] ModelFeatures inicializado com {len(self.original_features)} features")
    
    def prepare_features_for_models(self, price_data):
        """
        Prepara exatamente as 8 features originais para compatibilidade com modelos
        
        Args:
            price_data (pd.DataFrame): Dados OHLCV com indicadores
            
        Returns:
            pd.DataFrame: DataFrame com exatamente as 8 features originais
        """
        try:
            if price_data is None or price_data.empty:
                logger.warning("[WARN] Dados de preço vazios")
                return pd.DataFrame()
            
            df = price_data.copy()
            
            # Garante que temos os dados básicos OHLCV
            required_basic = ['open', 'high', 'low', 'close', 'volume']
            missing_basic = [col for col in required_basic if col not in df.columns]
            
            if missing_basic:
                logger.error(f"[ERROR] Colunas básicas faltando: {missing_basic}")
                return pd.DataFrame()
            
            # Calcula RSI se não existir
            if 'rsi' not in df.columns:
                df['rsi'] = self._calculate_rsi(df['close'])
            
            # Calcula volatilidade se não existir  
            if 'volatility' not in df.columns:
                df['volatility'] = df['close'].rolling(window=20).std()
            
            # Calcula mudança de preço se não existir
            if 'price_change_1' not in df.columns:
                df['price_change_1'] = df['close'].pct_change()
            
            # Seleciona APENAS as 8 features originais
            features_df = df[self.original_features].copy()
            
            # Remove linhas com NaN
            features_df = features_df.dropna()
            
            if features_df.empty:
                logger.warning("[WARN] Todas as linhas foram removidas devido a NaN")
                return pd.DataFrame()
            
            logger.debug(f"[OK] Preparadas {len(features_df)} amostras com {len(self.original_features)} features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao preparar features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calcula RSI (Relative Strength Index)
        
        Args:
            prices (pd.Series): Série de preços
            period (int): Período para cálculo
            
        Returns:
            pd.Series: Valores RSI
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"[ERROR] Erro no cálculo RSI: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def get_feature_info(self):
        """
        Retorna informações sobre as features
        
        Returns:
            dict: Informações das features
        """
        return {
            'count': len(self.original_features),
            'features': self.original_features,
            'description': {
                'open': 'Preço de abertura do período',
                'high': 'Preço máximo do período', 
                'low': 'Preço mínimo do período',
                'close': 'Preço de fechamento do período',
                'volume': 'Volume de negociação',
                'rsi': 'Relative Strength Index (14 períodos)',
                'volatility': 'Volatilidade (desvio padrão 20 períodos)',
                'price_change_1': 'Mudança percentual do preço'
            }
        }
    
    def validate_features(self, features_df):
        """
        Valida se as features estão corretas
        
        Args:
            features_df (pd.DataFrame): DataFrame com features
            
        Returns:
            dict: Resultado da validação
        """
        try:
            if features_df.empty:
                return {'valid': False, 'reason': 'DataFrame vazio'}
            
            # Verifica se tem todas as features
            missing_features = [f for f in self.original_features if f not in features_df.columns]
            if missing_features:
                return {'valid': False, 'reason': f'Features faltando: {missing_features}'}
            
            # Verifica se tem exatamente 8 features
            if len(features_df.columns) != len(self.original_features):
                return {'valid': False, 'reason': f'Esperadas {len(self.original_features)} features, encontradas {len(features_df.columns)}'}
            
            # Verifica se todas as features são numéricas
            non_numeric = features_df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                return {'valid': False, 'reason': f'Features não numéricas: {list(non_numeric)}'}
            
            # Verifica se tem dados suficientes
            if len(features_df) < 10:
                return {'valid': False, 'reason': f'Dados insuficientes: {len(features_df)} linhas'}
            
            # Verifica se há valores infinitos ou muito grandes
            if features_df.isin([np.inf, -np.inf]).any().any():
                return {'valid': False, 'reason': 'Valores infinitos encontrados'}
            
            # Verifica ranges razoáveis
            if (features_df < 0).any().any():
                negative_cols = features_df.columns[(features_df < 0).any()].tolist()
                # Apenas price_change_1 pode ser negativo
                invalid_negative = [col for col in negative_cols if col != 'price_change_1']
                if invalid_negative:
                    return {'valid': False, 'reason': f'Valores negativos inválidos em: {invalid_negative}'}
            
            return {
                'valid': True, 
                'shape': features_df.shape,
                'features': list(features_df.columns),
                'data_range': {
                    'min': features_df.min().to_dict(),
                    'max': features_df.max().to_dict(),
                    'mean': features_df.mean().to_dict()
                }
            }
            
        except Exception as e:
            return {'valid': False, 'reason': f'Erro na validação: {e}'}


# Instância global para uso em outros módulos
model_features = ModelFeatures()


def prepare_model_features(price_data):
    """
    Função de conveniência para preparar features
    
    Args:
        price_data (pd.DataFrame): Dados de preço
        
    Returns:
        pd.DataFrame: Features preparadas
    """
    return model_features.prepare_features_for_models(price_data)


def validate_model_features(features_df):
    """
    Função de conveniência para validar features
    
    Args:
        features_df (pd.DataFrame): Features a validar
        
    Returns:
        dict: Resultado da validação
    """
    return model_features.validate_features(features_df)


if __name__ == "__main__":
    # Teste das features
    print("🔧 TESTANDO CONFIGURAÇÃO DE FEATURES")
    print("="*50)
    
    features = ModelFeatures()
    info = features.get_feature_info()
    
    print(f"📊 Total de features: {info['count']}")
    print(f"📋 Features: {info['features']}")
    print("\n📝 Descrições:")
    for feature, desc in info['description'].items():
        print(f"   • {feature}: {desc}")
    
    print("\n✅ Configuração das features carregada com sucesso!") 