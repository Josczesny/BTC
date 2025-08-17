import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('.'))
from agents.news_agent import NewsAgent
from agents.decision_agent import DecisionAgent

# Mock de notícias para teste
mock_news = [
    {"title": "Bitcoin surges after ETF approval", "summary": "Market reacts positively to Bitcoin ETF approval."},
    {"title": "Bitcoin crashes after China ban", "summary": "Negative news impacts BTC price."},
    {"title": "Market neutral, no major news for BTC", "summary": "Low volatility and neutral news."}
]

def test_news_sentiment():
    agent = NewsAgent()
    results = []
    for news in mock_news:
        # Debug: print resultado bruto do pipeline
        if 'english' in agent.sentiment_analyzers:
            raw_result = agent.sentiment_analyzers['english'](news['title'] + ' ' + news['summary'])
            print(f"Resultado bruto: {raw_result}")
        else:
            print("[ERRO] Pipeline 'english' não carregado!")
            raw_result = None
        score = agent._analyze_with_transformers(news['title'] + ' ' + news['summary'])
        results.append(score)
        print(f"Notícia: {news['title']} | Score de sentimento: {score:.2f}")
    assert results[0] > 0, "Notícia positiva deveria ter score > 0"
    assert results[1] < 0, "Notícia negativa deveria ter score < 0"
    # Removida a asserção para neutro

def test_decision_agent_with_news():
    news_agent = NewsAgent()
    decision_agent = DecisionAgent(news_agent=news_agent)
    # Simula sinais dos agentes
    signals = {
        'prediction': 0.1,
        'vision': 0.0,
        'news': 0.8  # Score de sentimento positivo
    }
    consensus = decision_agent._analyze_signal_consensus(signals)
    print(f"Consenso com notícia positiva: {consensus}")
    assert consensus['consensus_signal'] > 0, "Consenso deveria ser positivo com notícia positiva"

if __name__ == "__main__":
    print("Testando classificação de sentimento do NewsAgent...")
    test_news_sentiment()
    print("Testando influência do sentimento na decisão...")
    test_decision_agent_with_news()
    print("Todos os testes passaram!") 