#!/usr/bin/env python3
"""
DEBUG DOS FEATURES AVANÇADOS
============================

Script para debugar especificamente os features que estão faltando.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.central_feature_engine import CentralFeatureEngine

def debug_features():
    """Debug específico dos features"""
    print("🔍 DEBUG DOS FEATURES AVANÇADOS")
    print("="*50)
    
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
    
    print(f"📊 Dados simulados criados: {len(market_data)} linhas")
    print(f"📋 Colunas originais: {list(market_data.columns)}")
    
    # Inicializa o feature engine
    feature_engine = CentralFeatureEngine()
    
    # Testa enriquecimento
    enriched_data = feature_engine.enrich_with_advanced_indicators(market_data)
    
    print(f"\n📈 Dados enriquecidos: {len(enriched_data)} linhas")
    print(f"📋 Colunas após enriquecimento: {list(enriched_data.columns)}")
    
    # Verifica features específicos
    required_features = [
        'atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14',
        'roc_10', 'mom_10', 'trix_15', 'ultosc', 'mfi_14',
        'stoch_k_14', 'stoch_d_14', 'bb_upper_20', 'bb_lower_20',
        'bb_middle_20', 'bb_width_20', 'bb_position_20'
    ]
    
    print(f"\n🔍 VERIFICAÇÃO DOS FEATURES:")
    print("-" * 30)
    
    missing_features = []
    for feature in required_features:
        if feature in enriched_data.columns:
            # Verifica se tem valores válidos
            valid_values = enriched_data[feature].notna().sum()
            total_values = len(enriched_data)
            print(f"✅ {feature}: {valid_values}/{total_values} valores válidos")
            
            # Mostra alguns valores de exemplo
            sample_values = enriched_data[feature].dropna().head(3).tolist()
            print(f"   Exemplo: {sample_values}")
        else:
            print(f"❌ {feature}: NÃO ENCONTRADO")
            missing_features.append(feature)
    
    print(f"\n📊 RESUMO:")
    print(f"✅ Features presentes: {len(required_features) - len(missing_features)}")
    print(f"❌ Features faltando: {len(missing_features)}")
    
    if missing_features:
        print(f"🔍 Features faltando: {missing_features}")
        
        # Tenta calcular manualmente os features faltando
        print(f"\n🔧 TENTANDO CALCULAR MANUALMENTE:")
        
        # TRIX
        if 'trix_15' in missing_features:
            print("Calculando TRIX manualmente...")
            try:
                trix = feature_engine._calculate_trix(enriched_data['close'], 15)
                enriched_data['trix_15'] = trix
                print(f"✅ TRIX calculado: {trix.notna().sum()} valores válidos")
            except Exception as e:
                print(f"❌ Erro calculando TRIX: {e}")
        
        # Bollinger Bands
        if 'bb_width_20' in missing_features or 'bb_position_20' in missing_features:
            print("Calculando Bollinger Bands manualmente...")
            try:
                bb_upper, bb_middle, bb_lower = feature_engine._calculate_bollinger_bands(enriched_data['close'])
                enriched_data['bb_upper_20'] = bb_upper
                enriched_data['bb_lower_20'] = bb_lower
                enriched_data['bb_middle_20'] = bb_middle
                
                # Calcula width e position
                enriched_data['bb_width_20'] = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0)
                enriched_data['bb_position_20'] = np.where((bb_upper - bb_lower) != 0, (enriched_data['close'] - bb_lower) / (bb_upper - bb_lower), 0.5)
                
                print(f"✅ Bollinger Bands calculados")
                print(f"   bb_width_20: {enriched_data['bb_width_20'].notna().sum()} valores válidos")
                print(f"   bb_position_20: {enriched_data['bb_position_20'].notna().sum()} valores válidos")
            except Exception as e:
                print(f"❌ Erro calculando Bollinger Bands: {e}")
    
    # Verifica novamente após correções
    print(f"\n🔍 VERIFICAÇÃO FINAL:")
    final_missing = []
    for feature in required_features:
        if feature in enriched_data.columns:
            print(f"✅ {feature}: OK")
        else:
            print(f"❌ {feature}: AINDA FALTANDO")
            final_missing.append(feature)
    
    if not final_missing:
        print(f"\n🎉 TODOS OS FEATURES ESTÃO PRESENTES!")
    else:
        print(f"\n⚠️ Ainda faltam: {final_missing}")

if __name__ == "__main__":
    debug_features() 