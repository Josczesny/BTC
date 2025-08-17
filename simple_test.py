#!/usr/bin/env python3
"""
TESTE SIMPLES DOS FEATURES
==========================

Teste direto para verificar se os features estão sendo calculados.
"""

import sys
import os
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.central_feature_engine import CentralFeatureEngine

def simple_test():
    """Teste simples dos features"""
    print("🧪 TESTE SIMPLES DOS FEATURES")
    print("="*40)
    
    # Cria dados simulados
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # Inicializa o feature engine
    feature_engine = CentralFeatureEngine()
    
    # Testa enriquecimento
    enriched_data = feature_engine.enrich_with_advanced_indicators(market_data)
    
    print(f"📊 Colunas após enriquecimento: {list(enriched_data.columns)}")
    
    # Verifica features específicos
    required_features = [
        'atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14',
        'roc_10', 'mom_10', 'trix_15', 'ultosc', 'mfi_14',
        'stoch_k_14', 'stoch_d_14', 'bb_upper_20', 'bb_lower_20',
        'bb_middle_20', 'bb_width_20', 'bb_position_20'
    ]
    
    print(f"\n🔍 VERIFICAÇÃO:")
    missing_features = []
    for feature in required_features:
        if feature in enriched_data.columns:
            print(f"✅ {feature}: PRESENTE")
        else:
            print(f"❌ {feature}: FALTANDO")
            missing_features.append(feature)
    
    print(f"\n📊 RESULTADO:")
    print(f"✅ Features presentes: {len(required_features) - len(missing_features)}")
    print(f"❌ Features faltando: {len(missing_features)}")
    
    if missing_features:
        print(f"🔍 Features faltando: {missing_features}")
        return False
    else:
        print("🎉 TODOS OS FEATURES ESTÃO PRESENTES!")
        return True

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1) 