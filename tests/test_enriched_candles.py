import sys
import os
sys.path.append(os.path.abspath('.'))
from data.data_collector import load_enriched_candles
from models.central_feature_engine import enrich_with_advanced_indicators
from agents.prediction_agent import PredictionAgent

if __name__ == "__main__":
    df = load_enriched_candles("data/historical/raw/BTCUSDT-1h.csv")
    df = enrich_with_advanced_indicators(df)
    print(df.tail())
    # Testa colunas de open interest/funding rate
    assert 'open_interest' in df.columns, "Coluna 'open_interest' não encontrada!"
    assert 'open_interest_value' in df.columns, "Coluna 'open_interest_value' não encontrada!"
    assert 'funding_rate' in df.columns, "Coluna 'funding_rate' não encontrada!"
    # Testa colunas de indicadores técnicos avançados
    for col in ['atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14', 'roc_10', 'mom_10', 'trix_15', 'ultosc']:
        assert col in df.columns, f"Coluna '{col}' não encontrada!"
    print("Teste de integração dos dados enriquecidos e indicadores avançados: OK!")

    # Testa PredictionAgent usando os features avançados
    agent = PredictionAgent()
    features = agent.prepare_features(df)
    print("Features usados pelo PredictionAgent:", features.columns.tolist())
    for col in ['atr_14', 'adx_14', 'cci_20', 'obv', 'williams_r_14', 'roc_10', 'mom_10', 'trix_15', 'ultosc']:
        assert col in features.columns, f"PredictionAgent não está usando o feature '{col}'!"
    print("PredictionAgent está usando todos os features avançados: OK!") 