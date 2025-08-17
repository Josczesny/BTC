# agents/__init__.py
"""
Módulo de Agentes Inteligentes para Trading Automatizado

Este módulo contém todos os agentes especializados:
- NewsAgent: Análise de sentimento de notícias
- VisionAgent: Análise visual de gráficos
- PredictionAgent: Previsão de preços
- DecisionAgent: Tomada de decisões integradas
- OrderAgent: Gestão de ordens
"""

import threading
import time
from agents.news_agent import NewsAgent

# Orquestrador de agentes para execução simultânea
class AgentOrchestrator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.results = {}
        self.threads = []
        self.running = True

    def run_news(self):
        while self.running:
            news = self.news_agent.fetch_news()
            self.results['news'] = news
            print('[ORCHESTRATOR] Notícias atualizadas')
            time.sleep(60)  # Atualiza a cada 1 min

    def run_coingecko(self):
        while self.running:
            cg = self.news_agent.fetch_coingecko_price_sentiment()
            self.results['coingecko'] = cg
            print('[ORCHESTRATOR] CoinGecko atualizado')
            time.sleep(60)

    def run_fear_greed(self):
        while self.running:
            fg = self.news_agent.fetch_fear_greed_index()
            self.results['fear_greed'] = fg
            print('[ORCHESTRATOR] Fear & Greed atualizado')
            time.sleep(3600)  # Fear & Greed atualiza a cada hora

    def start(self):
        self.threads = [
            threading.Thread(target=self.run_news, daemon=True),
            threading.Thread(target=self.run_coingecko, daemon=True),
            threading.Thread(target=self.run_fear_greed, daemon=True)
        ]
        for t in self.threads:
            t.start()
        print('[ORCHESTRATOR] Todos os agentes iniciados em paralelo.')

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=2)
        print('[ORCHESTRATOR] Orquestrador parado.')

# Exemplo de uso
if __name__ == '__main__':
    orchestrator = AgentOrchestrator()
    orchestrator.start()
    try:
        while True:
            time.sleep(10)
            print('[ORCHESTRATOR] Resultados parciais:', orchestrator.results)
    except KeyboardInterrupt:
        orchestrator.stop() 