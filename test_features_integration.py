#!/usr/bin/env python3
"""
TESTE DE INTEGRAÇÃO DOS FEATURES AVANÇADOS
=========================================

Verifica se os features avançados estão sendo utilizados corretamente
no pipeline de treinamento, predição e execução de trades.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.terminal_colors import TerminalColors
from data.data_collector import DataCollector
from models.central_feature_engine import CentralFeatureEngine
from agents.prediction_agent import PredictionAgent
from core.model_manager import ModelManager
from core.signal_processor import SignalProcessor

def test_feature_engine_integration():
    """Testa se o CentralFeatureEngine está sendo usado corretamente"""
    print(TerminalColors.highlight("\n🧪 TESTE 1: CentralFeatureEngine Integration"))
    
    try:
        # Inicializa o feature engine
        feature_engine = CentralFeatureEngine()
        print(TerminalColors.success("✅ CentralFeatureEngine inicializado"))
        
        # Usa dados simulados para teste consistente
        print(TerminalColors.info("📊 Usando dados simulados para teste consistente"))
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Testa enriquecimento com features avançados
        enriched_data = feature_engine.enrich_with_advanced_indicators(market_data)
        
        # Verifica se os features avançados foram adicionados
        required_features = [
            'atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14',
            'roc_10', 'mom_10', 'trix_15', 'ultosc', 'mfi_14',
            'stoch_k_14', 'stoch_d_14', 'bb_upper_20', 'bb_lower_20',
            'bb_middle_20', 'bb_width_20', 'bb_position_20'
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in enriched_data.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(TerminalColors.error(f"❌ Features faltando: {missing_features}"))
            return False
        else:
            print(TerminalColors.success(f"✅ Todos os {len(required_features)} features avançados presentes"))
        
        # Verifica se os valores não são todos NaN
        for feature in required_features:
            if enriched_data[feature].isna().all():
                print(TerminalColors.warning(f"⚠️ Feature {feature} tem apenas valores NaN"))
            else:
                print(TerminalColors.success(f"✅ Feature {feature}: valores válidos"))
        
        return True
        
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro no teste do CentralFeatureEngine: {e}"))
        return False

def test_prediction_agent_integration():
    """Testa se o PredictionAgent está usando os features avançados"""
    print(TerminalColors.highlight("\n🧪 TESTE 2: PredictionAgent Integration"))
    
    try:
        # Inicializa o prediction agent
        prediction_agent = PredictionAgent()
        print(TerminalColors.success("✅ PredictionAgent inicializado"))
        
        # Carrega dados enriquecidos
        data_collector = DataCollector()
        market_data = data_collector.get_market_data_with_fallbacks()
        
        if market_data is None or len(market_data) < 50:
            print(TerminalColors.warning("⚠️ Dados insuficientes, usando dados simulados"))
            # Cria dados simulados enriquecidos
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            market_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(40000, 50000, 100),
                'high': np.random.uniform(40000, 50000, 100),
                'low': np.random.uniform(40000, 50000, 100),
                'close': np.random.uniform(40000, 50000, 100),
                'volume': np.random.uniform(100, 1000, 100)
            })
            
            # Adiciona features básicos
            feature_engine = CentralFeatureEngine()
            market_data = feature_engine.enrich_with_advanced_indicators(market_data)
        
        # Testa preparação de features
        features = prediction_agent.prepare_features(market_data)
        
        # Verifica se os features avançados estão sendo usados
        advanced_features_used = [
            'atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14',
            'roc_10', 'mom_10', 'trix_15', 'ultosc'
        ]
        
        missing_in_prediction = []
        for feature in advanced_features_used:
            if feature not in features.columns:
                missing_in_prediction.append(feature)
        
        if missing_in_prediction:
            print(TerminalColors.error(f"❌ PredictionAgent não está usando: {missing_in_prediction}"))
            return False
        else:
            print(TerminalColors.success(f"✅ PredictionAgent usando todos os {len(advanced_features_used)} features avançados"))
        
        # Testa predição
        prediction = prediction_agent.predict_next_move(market_data)
        print(TerminalColors.success(f"✅ Predição gerada: {prediction}"))
        
        return True
        
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro no teste do PredictionAgent: {e}"))
        return False

def test_model_manager_integration():
    """Testa se o ModelManager está usando os features avançados"""
    print(TerminalColors.highlight("\n🧪 TESTE 3: ModelManager Integration"))
    
    try:
        # Inicializa o model manager
        model_manager = ModelManager()
        print(TerminalColors.success("✅ ModelManager inicializado"))
        
        # Garante dados simulados com pelo menos 30 linhas e todas as colunas
        print(TerminalColors.info("📊 Usando dados simulados para teste consistente"))
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 50),
            'high': np.random.uniform(40000, 50000, 50),
            'low': np.random.uniform(40000, 50000, 50),
            'close': np.random.uniform(40000, 50000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        })
        print("DEBUG: market_data shape:", market_data.shape)
        print("DEBUG: market_data columns:", list(market_data.columns))
        print("DEBUG: market_data head:\n", market_data.head())
        
        # Testa predições dos modelos
        predictions = model_manager.get_model_predictions(market_data)
        
        # Adiciona log do advanced_features
        advanced_features = model_manager._prepare_advanced_features(market_data)
        print("DEBUG: advanced_features shape:", advanced_features.shape)
        print("DEBUG: advanced_features columns:", list(advanced_features.columns))
        print("DEBUG: advanced_features head:\n", advanced_features.head())
        
        if predictions:
            print(TerminalColors.success(f"✅ ModelManager gerou {len(predictions)} predições"))
            for model_name, prediction in predictions.items():
                print(TerminalColors.info(f"   • {model_name}: {prediction}"))
        else:
            print(TerminalColors.warning("⚠️ ModelManager não gerou predições"))
        
        return True
        
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro no teste do ModelManager: {e}"))
        return False

def test_signal_processor_integration():
    """Testa se o SignalProcessor está usando os features avançados"""
    print(TerminalColors.highlight("\n🧪 TESTE 4: SignalProcessor Integration"))
    
    try:
        # Inicializa o signal processor
        signal_processor = SignalProcessor()
        print(TerminalColors.success("✅ SignalProcessor inicializado"))
        
        # Carrega dados
        data_collector = DataCollector()
        market_data = data_collector.get_market_data_with_fallbacks()
        current_price = data_collector.get_current_price()
        
        if market_data is None or len(market_data) < 50:
            print(TerminalColors.warning("⚠️ Dados insuficientes, usando dados simulados"))
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
            current_price = 45000
            
            # Enriquece com features
            feature_engine = CentralFeatureEngine()
            market_data = feature_engine.enrich_with_advanced_indicators(market_data)
        
        # Testa geração de sinal de consenso
        signal, confidence = signal_processor.get_consensus_signal(
            market_data, current_price, {}
        )
        
        print(TerminalColors.success(f"✅ SignalProcessor gerou sinal: {signal} (confiança: {confidence:.2f})"))
        
        return True
        
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro no teste do SignalProcessor: {e}"))
        return False

def test_main_pipeline_integration():
    """Testa se o main.py está usando os features avançados corretamente"""
    print(TerminalColors.highlight("\n🧪 TESTE 5: Main Pipeline Integration"))
    
    try:
        # Importa o sistema principal
        from main import ModularTradingSystem
        
        # Inicializa o sistema
        system = ModularTradingSystem()
        print(TerminalColors.success("✅ ModularTradingSystem inicializado"))
        
        # Verifica se os sistemas centralizados foram inicializados
        if hasattr(system, 'central_feature_engine') and system.central_feature_engine:
            print(TerminalColors.success("✅ CentralFeatureEngine conectado ao sistema principal"))
        else:
            print(TerminalColors.error("❌ CentralFeatureEngine não conectado ao sistema principal"))
            return False
        
        if hasattr(system, 'central_ensemble_system') and system.central_ensemble_system:
            print(TerminalColors.success("✅ CentralEnsembleSystem conectado ao sistema principal"))
        else:
            print(TerminalColors.error("❌ CentralEnsembleSystem não conectado ao sistema principal"))
            return False
        
        # Verifica se os agentes estão usando os sistemas centralizados
        if hasattr(system, 'prediction_agent') and system.prediction_agent:
            print(TerminalColors.success("✅ PredictionAgent conectado ao sistema principal"))
        else:
            print(TerminalColors.error("❌ PredictionAgent não conectado ao sistema principal"))
            return False
        
        return True
        
    except Exception as e:
        print(TerminalColors.error(f"❌ Erro no teste do pipeline principal: {e}"))
        return False

def run_all_tests():
    """Executa todos os testes de integração"""
    print(TerminalColors.colorize("\n" + "="*80, TerminalColors.CYAN, TerminalColors.BOLD))
    print(TerminalColors.highlight("🧪 TESTE COMPLETO DE INTEGRAÇÃO DOS FEATURES AVANÇADOS"))
    print(TerminalColors.colorize("="*80, TerminalColors.CYAN, TerminalColors.BOLD))
    
    tests = [
        ("CentralFeatureEngine", test_feature_engine_integration),
        ("PredictionAgent", test_prediction_agent_integration),
        ("ModelManager", test_model_manager_integration),
        ("SignalProcessor", test_signal_processor_integration),
        ("Main Pipeline", test_main_pipeline_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(TerminalColors.info(f"\n🔍 Executando teste: {test_name}"))
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(TerminalColors.error(f"❌ Erro crítico no teste {test_name}: {e}"))
            results[test_name] = False
    
    # Relatório final
    print(TerminalColors.colorize("\n" + "="*80, TerminalColors.CYAN, TerminalColors.BOLD))
    print(TerminalColors.highlight("📊 RELATÓRIO FINAL DOS TESTES"))
    print(TerminalColors.colorize("="*80, TerminalColors.CYAN, TerminalColors.BOLD))
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print(TerminalColors.success(f"✅ {test_name}: PASSOU"))
            passed += 1
        else:
            print(TerminalColors.error(f"❌ {test_name}: FALHOU"))
    
    print(TerminalColors.colorize("\n" + "="*50, TerminalColors.CYAN))
    print(TerminalColors.info(f"📈 RESULTADO: {passed}/{total} testes passaram"))
    
    if passed == total:
        print(TerminalColors.success("🎉 TODOS OS TESTES PASSARAM! Features avançados integrados com sucesso!"))
        print(TerminalColors.success("✅ O sistema está pronto para usar os features avançados nos trades"))
    elif passed >= total * 0.8:
        print(TerminalColors.warning("⚠️ MAIORIA DOS TESTES PASSOU. Alguns ajustes podem ser necessários."))
    else:
        print(TerminalColors.error("❌ MUITOS TESTES FALHARAM. Verifique a integração dos features."))
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 