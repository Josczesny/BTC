# agents/news_agent.py
"""
Agente de Análise de Notícias
Coleta e analisa notícias para prever sentimento do mercado de Bitcoin

Funcionalidades:
- Coleta de notícias de múltiplas fontes (RSS, Twitter, CoinTelegraph)
- Análise de sentimento usando NLP (LangChain, transformers)
- Scoring de impacto no preço do Bitcoin
- Cache inteligente para evitar reprocessamento
"""

import os
# 🚫 SUPRIME LOGS TENSORFLOW/TRANSFORMERS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import requests
import feedparser
from datetime import datetime, timedelta
import re
import json
import time
import hashlib
from collections import Counter
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline as hf_pipeline
from textblob import TextBlob
from urllib.parse import urlparse

from utils.logger import setup_trading_logger

logger = setup_trading_logger("news-agent")

CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')
CRYPTOPANIC_API_URL = 'https://cryptopanic.com/api/v1/posts/'
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'
FEAR_GREED_API_URL = os.getenv('FEAR_GREED_API_URL', 'https://api.alternative.me/fng/')

class NewsAgent:
    def __init__(self, central_feature_engine=None):
        """
        Inicializa o agente de notícias
        """
        self.central_feature_engine = central_feature_engine
        logger.info("[NEWS]  Inicializando NewsAgent")
        
        # URLs de feeds RSS específicos para Bitcoin
        self.rss_feeds = [
            "https://cointelegraph.com/rss/tag/bitcoin",
            "https://coindesk.com/arc/outboundfeeds/rss/",
            "https://news.bitcoin.com/feed/",
            # "https://bitcoinmagazine.com/.rss/full/",  # REMOVIDO - erro 403
            "https://cryptonews.com/news/bitcoin-news/feed/",
            "https://www.crypto-news-flash.com/feed/"
        ]
        
        # Inicializa analisadores de sentimento
        self.sentiment_analyzers = {}
        self._initialize_sentiment_models()
        
        # Cache de notícias processadas (evita reprocessamento)
        self.processed_news = {}
        self.cache_duration = 1800  # 30 minutos
        
        # Palavras-chave específicas para Bitcoin com pesos
        self.bitcoin_keywords = {
            # Bullish keywords
            'bullish': {
                'adoption': 1.0, 'bull': 1.0, 'rally': 1.0, 'surge': 1.0, 'moon': 1.0, 'pump': 1.0, 'breakout': 1.0, 
                'institutional': 1.0, 'etf': 1.0, 'approval': 1.0, 'upgrade': 1.0, 'halving': 1.0, 'scarcity': 1.0,
                'buy': 1.0, 'hodl': 1.0, 'investment': 1.0, 'store of value': 1.0, 'digital gold': 1.0,
                'rise': 0.7, 'growth': 0.7, 'increase': 0.7, 'profit': 0.7, 'gains': 0.7, 'up': 0.7,
                'positive': 0.5, 'good': 0.5, 'great': 0.5, 'excellent': 0.5, 'amazing': 0.5
            },
            # Bearish keywords  
            'bearish': {
                'crash': 1.0, 'dump': 1.0, 'bear': 1.0, 'decline': 1.0, 'fall': 1.0, 'drop': 1.0, 'correction': 1.0,
                'regulation': 1.0, 'ban': 1.0, 'restriction': 1.0, 'concern': 1.0, 'fear': 1.0, 'panic': 1.0,
                'sell': 1.0, 'liquidation': 1.0, 'volatility': 1.0, 'risk': 1.0,
                'down': 0.7, 'loss': 0.7, 'losses': 0.7, 'decrease': 0.7, 'red': 0.7,
                'negative': 0.5, 'bad': 0.5, 'terrible': 0.5, 'awful': 0.5, 'concerning': 0.5
            },
            # High impact keywords
            'high_impact': {
                'sec': 2.0, 'etf': 2.0, 'institutional': 2.0, 'tesla': 2.0, 'microstrategy': 2.0, 'saylor': 2.0,
                'fed': 2.0, 'federal reserve': 2.0, 'interest rate': 2.0, 'inflation': 2.0, 'regulation': 2.0,
                'china': 2.0, 'el salvador': 2.0, 'mining': 2.0, 'hashrate': 2.0, 'halving': 2.0
            }
        }
        
        # Fontes confiáveis com pesos diferentes
        self.source_weights = {
            'cointelegraph.com': 1.0,
            'coindesk.com': 1.2,  # CoinDesk é mais confiável
            # 'bitcoinmagazine.com': 1.1,  # REMOVIDO - erro 403
            'news.bitcoin.com': 0.9,
            'cryptonews.com': 0.8,
            'crypto-news-flash.com': 0.7
        }
        
        self.last_cryptopanic_request = None
        self.cryptopanic_rate_limit = 120  # 120 req/hora (2/min)
        self.cryptopanic_min_interval = 31  # segundos entre requests
        
        logger.info("[OK] NewsAgent inicializado com sucesso")

    def _initialize_sentiment_models(self):
        """
        Inicializa modelos de análise de sentimento
        """
        try:
            # Inicializa pipeline em inglês
            self.sentiment_analyzers['english'] = hf_pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            logger.info("[OK] Pipeline 'english' carregado com sucesso!")
        except Exception as e:
            logger.error(f"[ERROR] Erro ao carregar pipeline em inglês: {e}")
            print(f"[DEBUG] Falha ao carregar pipeline 'english': {e}")

    def fetch_cryptopanic_news(self, currency='BTC', filter_='hot', kind='news'):
        """Busca notícias do CryptoPanic para o Bitcoin"""
        now = datetime.utcnow()
        if self.last_cryptopanic_request and (now - self.last_cryptopanic_request).total_seconds() < self.cryptopanic_min_interval:
            logger.warning('[NEWS] Limite de requisição CryptoPanic atingido, aguardando...')
            return []
        params = {
            'auth_token': CRYPTOPANIC_API_KEY,
            'currencies': currency,
            'filter': filter_,
            'kind': kind,
            'public': 'true'
        }
        try:
            response = requests.get(CRYPTOPANIC_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                self.last_cryptopanic_request = now
                data = response.json()
                news = data.get('results', [])
                logger.info(f'[NEWS] {len(news)} notícias recebidas do CryptoPanic')
                return news
            elif response.status_code == 429:
                logger.warning('[NEWS] Limite de requisições CryptoPanic excedido (429)')
                return []
            else:
                logger.error(f'[NEWS] Erro CryptoPanic: {response.status_code}')
                return []
        except Exception as e:
            logger.error(f'[NEWS] Erro ao requisitar CryptoPanic: {e}')
            return []

    def fetch_coingecko_price_sentiment(self, coin_id='bitcoin'):
        """Busca preço e sentimento de mercado do CoinGecko"""
        url = f"{COINGECKO_API_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': coin_id,
            'x_cg_pro_api_key': COINGECKO_API_KEY
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    price = data[0].get('current_price')
                    sentiment = data[0].get('sentiment_votes_up_percentage', None)
                    logger.info(f"[COINGECKO] Preço: {price}, Sentimento: {sentiment}")
                    return {'price': price, 'sentiment': sentiment}
            elif response.status_code == 429:
                logger.warning('[COINGECKO] Limite de requisições excedido (429)')
                return None
            else:
                logger.error(f'[COINGECKO] Erro: {response.status_code}')
                return None
        except Exception as e:
            logger.error(f'[COINGECKO] Erro ao requisitar CoinGecko: {e}')
            return None

    def fetch_fear_greed_index(self, limit=1):
        """Busca o Fear & Greed Index da Alternative.me"""
        params = {'limit': limit}
        try:
            response = requests.get(FEAR_GREED_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    logger.info(f"[FEAR&GREED] Valor: {data['data'][0]['value']}, Classificação: {data['data'][0]['value_classification']}")
                    return data['data']
            else:
                logger.error(f'[FEAR&GREED] Erro: {response.status_code}')
                return None
        except Exception as e:
            logger.error(f'[FEAR&GREED] Erro ao requisitar índice: {e}')
            return None

    def fetch_news(self, hours_back=6):
        """Busca notícias de múltiplas fontes, priorizando CryptoPanic se disponível"""
        news = self.fetch_cryptopanic_news()
        if news:
            return news
        # Fallback para RSS se CryptoPanic não retornar nada
        return self._fetch_from_rss_all(hours_back)

    def _fetch_from_rss_all(self, hours_back=6):
        """Busca notícias de todos os feeds RSS (original)"""
        logger.info(f"[NEWS2] Coletando notícias das últimas {hours_back} horas")
        
        all_news = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # TIMEOUT GERAL: máximo 15 segundos para toda coleta
        start_time = time.time()
        max_collection_time = 15  # 15 segundos máximo
        
        for i, feed_url in enumerate(self.rss_feeds):
            # Verifica timeout geral
            if time.time() - start_time > max_collection_time:
                logger.warning(f"[TIMEOUT] Coleta interrompida após {max_collection_time}s (processou {i}/{len(self.rss_feeds)} feeds)")
                break
                
            try:
                feed_news = self._fetch_from_rss(feed_url, cutoff_time)
                all_news.extend(feed_news)
                time.sleep(0.5)  # Rate limiting reduzido
                
            except Exception as e:
                logger.error(f"[ERROR] Erro ao coletar de {feed_url}: {e}")
                continue
        
        # Remove duplicatas baseado no hash do conteúdo
        unique_news = self._remove_duplicates(all_news)
        
        elapsed_time = time.time() - start_time
        logger.info(f"[DATA] Coletadas {len(unique_news)} notícias únicas de {len(all_news)} total em {elapsed_time:.1f}s")
        
        return unique_news

    def _fetch_from_rss(self, feed_url, cutoff_time):
        """
        Coleta notícias de um feed RSS específico
        
        Args:
            feed_url (str): URL do feed RSS
            cutoff_time (datetime): Tempo limite para notícias antigas
            
        Returns:
            list: Notícias do feed
        """
        try:
            logger.debug(f"[SIGNAL] Coletando de: {feed_url}")
            
            # Headers para evitar blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Fetch RSS feed com timeout reduzido
            response = requests.get(feed_url, headers=headers, timeout=5)  # 5 segundos máximo
            response.raise_for_status()
            
            # Parse RSS
            feed = feedparser.parse(response.content)
            
            if not hasattr(feed, 'entries') or not feed.entries:
                logger.warning(f"[WARN] Feed vazio: {feed_url}")
                return []
            
            news_items = []
            
            for entry in feed.entries:
                try:
                    # Parse data
                    pub_date = self._parse_date(entry.get('published', ''))
                    
                    # Filtra por data
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Extrai conteúdo
                    title = entry.get('title', '').strip()
                    description = entry.get('description', '').strip()
                    link = entry.get('link', '')
                    
                    # Filtra por relevância Bitcoin
                    if not self._is_bitcoin_relevant(title + ' ' + description):
                        continue
                    
                    # Limpa texto
                    content = self._clean_text(title + '. ' + description)
                    
                    if len(content) < 10:  # Muito curto
                        continue
                    
                    # Determina fonte
                    source = self._extract_source(feed_url)
                    
                    news_item = {
                        'title': title,
                        'content': content,
                        'description': description,
                        'link': link,
                        'published': pub_date or datetime.now(),
                        'source': source,
                        'source_weight': self.source_weights.get(source, 0.5),
                        'content_hash': hashlib.md5(content.encode()).hexdigest()
                    }
                    
                    news_items.append(news_item)
                    
                    # Limita a 5 notícias por feed para evitar travamento
                    if len(news_items) >= 5:
                        break
                    
                except Exception as e:
                    logger.debug(f"[WARN] Erro ao processar entrada: {e}")
                    continue
            
            logger.debug(f"[OK] {len(news_items)} notícias relevantes de {feed_url}")
            return news_items
            
        except requests.exceptions.Timeout:
            logger.warning(f"[TIMEOUT] Feed {feed_url} demorou muito (>5s)")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"[ERROR] Erro de rede em {feed_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"[ERROR] Erro ao coletar RSS {feed_url}: {e}")
            return []

    def _parse_date(self, date_str):
        """
        Parse data de publicação
        """
        try:
            if not date_str:
                return None
            
            # Remove timezone info para simplificar
            date_str = re.sub(r'\s*[+-]\d{4}$', '', date_str)
            date_str = re.sub(r'\s*GMT$', '', date_str)
            date_str = re.sub(r'\s*UTC$', '', date_str)
            
            # Formatos comuns
            formats = [
                '%a, %d %b %Y %H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%d %b %Y %H:%M:%S',
                '%a %b %d %H:%M:%S %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None

    def _is_bitcoin_relevant(self, text):
        """
        Verifica se o texto é relevante para Bitcoin
        """
        text_lower = text.lower()
        
        # Palavras-chave obrigatórias
        bitcoin_terms = ['bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain']
        
        return any(term in text_lower for term in bitcoin_terms)

    def _clean_text(self, text):
        """
        Limpa e normaliza texto
        """
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # Remove caracteres especiais em excesso
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
            
            # Normaliza espaços
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception:
            return text

    def _extract_source(self, feed_url):
        """
        Extrai nome da fonte do URL
        """
        try:
            domain = urlparse(feed_url).netloc
            return domain.replace('www.', '')
        except:
            return 'unknown'

    def _remove_duplicates(self, news_list):
        """
        Remove notícias duplicadas baseado no hash do conteúdo
        """
        seen_hashes = set()
        unique_news = []
        
        for news in news_list:
            content_hash = news.get('content_hash')
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_news.append(news)
        
        return unique_news

    def analyze_sentiment(self, text):
        """
        Analisa o sentimento de um texto usando múltiplos modelos
        
        Args:
            text (str): Texto para análise
            
        Returns:
            dict: Score de sentimento (-1 a 1) e confiança
        """
        if not text or len(text.strip()) < 5:
            return {"sentiment": 0.0, "confidence": 0.0}
        
        try:
            sentiments = []
            
            # Análise com modelo financeiro (FinBERT)
            if 'finbert' in self.sentiment_analyzers:
                finbert_result = self._analyze_with_finbert(text)
                sentiments.append(finbert_result)
            
            # Análise com modelo de Twitter
            elif 'twitter' in self.sentiment_analyzers: # Changed from 'cardiffnlp' to 'twitter'
                twitter_result = self._analyze_with_twitter_model(text)
                sentiments.append(twitter_result)
            
            # Análise com TextBlob
            if 'textblob' in self.sentiment_analyzers:
                textblob_result = self._analyze_with_textblob(text)
                sentiments.append(textblob_result)
            
            # Análise por palavras-chave específicas
            keyword_result = self._analyze_keywords(text)
            sentiments.append(keyword_result)
            
            # Combina resultados
            if sentiments:
                avg_sentiment = np.mean([s["sentiment"] for s in sentiments])
                avg_confidence = np.mean([s["confidence"] for s in sentiments])
                
                # Bonus por concordância entre modelos
                sentiment_values = [s["sentiment"] for s in sentiments]
                agreement = 1.0 - np.std(sentiment_values) if len(sentiment_values) > 1 else 0.5
                final_confidence = avg_confidence * agreement
                
                return {
                    "sentiment": float(avg_sentiment),
                    "confidence": float(min(final_confidence, 0.95)),
                    "individual_results": sentiments
                }
            
            return {"sentiment": 0.0, "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de sentimento: {e}")
            return {"sentiment": 0.0, "confidence": 0.0}

    def _analyze_with_finbert(self, text):
        """
        Análise com FinBERT (modelo financeiro)
        """
        try:
            analyzer = self.sentiment_analyzers['finbert']
            result = analyzer(text[:512])  # Limite do modelo
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Converte para escala -1 a 1
            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:  # neutral
                sentiment = 0.0
            
            return {
                "sentiment": sentiment,
                "confidence": score,
                "model": "finbert"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro FinBERT: {e}")
            return {"sentiment": 0.0, "confidence": 0.0, "model": "finbert"}

    def _analyze_with_twitter_model(self, text):
        """
        Análise com modelo Twitter RoBERTa
        """
        try:
            analyzer = self.sentiment_analyzers['twitter'] # Changed from 'cardiffnlp' to 'twitter'
            result = analyzer(text[:512])
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Converte labels do Twitter RoBERTa
            if 'positive' in label:
                sentiment = score
            elif 'negative' in label:
                sentiment = -score
            else:
                sentiment = 0.0
            
            return {
                "sentiment": sentiment,
                "confidence": score,
                "model": "twitter_roberta"
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro Twitter model: {e}")
            return {"sentiment": 0.0, "confidence": 0.0, "model": "twitter_roberta"}

    def _analyze_with_transformers(self, text):
        try:
            if 'english' in self.sentiment_analyzers:
                result = self.sentiment_analyzers['english'](text)
                # O resultado é uma lista de dicts [{'label': 'POSITIVE'/'NEGATIVE', 'score': ...}]
                label = result[0]['label']
                score = result[0]['score']
                if label == 'POSITIVE':
                    return score
                elif label == 'NEGATIVE':
                    return -score
                else:
                    return 0.0
            else:
                return self._analyze_with_textblob(text)
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de sentimento com transformers: {e}")
            return self._analyze_with_textblob(text)

    def _analyze_with_textblob(self, text):
        try:
            blob = TextBlob(text)
            if blob.detect_language() != 'en':
                text_en = str(blob.translate(to='en'))
                blob = TextBlob(text_en)
            return blob.sentiment.polarity
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de sentimento com TextBlob: {e}")
            return 0.0

    def _analyze_keywords(self, text):
        """
        Análise baseada em palavras-chave específicas de Bitcoin
        """
        try:
            text_lower = text.lower()
            
            bullish_score = 0.0
            bearish_score = 0.0
            high_impact_multiplier = 1.0
            
            # Conta palavras bullish
            for word, weight in self.bitcoin_keywords['bullish'].items():
                count = text_lower.count(word)
                bullish_score += count * weight
            
            # Conta palavras bearish
            for word, weight in self.bitcoin_keywords['bearish'].items():
                count = text_lower.count(word)
                bearish_score += count * weight
            
            # Multiplica por impacto de palavras de alto impacto
            for word, multiplier in self.bitcoin_keywords['high_impact'].items():
                if word in text_lower:
                    high_impact_multiplier *= multiplier
            
            # Calcula sentimento final
            total_score = (bullish_score - bearish_score) * high_impact_multiplier
            
            # Normaliza para escala -1 a 1
            max_possible_score = 10.0  # Valor empírico
            sentiment = np.tanh(total_score / max_possible_score)
            
            # Confiança baseada na quantidade de palavras-chave encontradas
            total_keywords = bullish_score + bearish_score
            confidence = min(0.8, total_keywords / 5.0)  # Máximo 80% de confiança
            
            return {
                "sentiment": float(sentiment),
                "confidence": float(confidence),
                "model": "keywords",
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "high_impact": high_impact_multiplier > 1.0
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Erro análise de keywords: {e}")
            return {"sentiment": 0.0, "confidence": 0.0, "model": "keywords"}

    def get_market_sentiment_score(self):
        """
        Calcula score geral de sentimento do mercado
        
        Returns:
            dict: Score agregado e metadados
        """
        logger.info("[UP] Calculando sentimento geral do mercado")
        
        # Coleta notícias recentes
        recent_news = self.fetch_news(hours_back=6)
        
        if not recent_news:
            logger.warning("[WARN]  Nenhuma notícia encontrada")
            return {
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "news_count": 0,
                "positive_news": 0,
                "negative_news": 0,
                "source_breakdown": {}
            }
        
        # Analisa sentimento de cada notícia
        sentiments = []
        positive_count = 0
        negative_count = 0
        source_sentiments = {}
        
        for news_item in recent_news:
            # Combina título e conteúdo para análise
            text = news_item.get("title", "") + ". " + news_item.get("content", "")
            
            sentiment_result = self.analyze_sentiment(text)
            
            # Aplica peso da fonte
            source_weight = news_item.get("source_weight", 1.0)
            weighted_sentiment = sentiment_result["sentiment"] * source_weight
            weighted_confidence = sentiment_result["confidence"] * source_weight
            
            sentiment_result["weighted_sentiment"] = weighted_sentiment
            sentiment_result["weighted_confidence"] = weighted_confidence
            sentiment_result["source"] = news_item.get("source", "unknown")
            
            sentiments.append(sentiment_result)
            
            # Contagem por direção
            if weighted_sentiment > 0.1:
                positive_count += 1
            elif weighted_sentiment < -0.1:
                negative_count += 1
            
            # Breakdown por fonte
            source = news_item.get("source", "unknown")
            if source not in source_sentiments:
                source_sentiments[source] = []
            source_sentiments[source].append(weighted_sentiment)
        
        # Calcula score agregado ponderado
        if sentiments:
            # Peso por confiança e fonte
            weighted_scores = [s["weighted_sentiment"] * s["weighted_confidence"] for s in sentiments]
            total_confidence = sum(s["weighted_confidence"] for s in sentiments)
            
            overall_sentiment = sum(weighted_scores) / total_confidence if total_confidence > 0 else 0.0
            avg_confidence = total_confidence / len(sentiments)
            
            # Aplica decaimento temporal (notícias mais recentes têm mais peso)
            time_weights = [1.0 / (1.0 + i * 0.1) for i in range(len(sentiments))]
            time_weighted_sentiment = np.average([s["weighted_sentiment"] for s in sentiments], weights=time_weights)
            
            # Media entre agregação simples e temporal
            final_sentiment = (overall_sentiment + time_weighted_sentiment) / 2.0
            
        else:
            final_sentiment = 0.0
            avg_confidence = 0.0
        
        # Breakdown por fonte
        source_breakdown = {}
        for source, sent_list in source_sentiments.items():
            source_breakdown[source] = {
                "avg_sentiment": np.mean(sent_list),
                "news_count": len(sent_list)
            }
        
        result = {
            "overall_sentiment": float(final_sentiment),
            "confidence": float(avg_confidence),
            "news_count": len(recent_news),
            "positive_news": positive_count,
            "negative_news": negative_count,
            "neutral_news": len(recent_news) - positive_count - negative_count,
            "source_breakdown": source_breakdown,
            "individual_sentiments": sentiments,
            "timestamp": datetime.now()
        }
        
        logger.info(f"[DATA] Sentimento: {final_sentiment:.3f} ({positive_count}+ / {negative_count}- / {len(recent_news)} total)")
        
        return result

    def get_market_sentiment(self):
        """
        Método de compatibilidade para get_market_sentiment_score
        
        Returns:
            dict: Score de sentimento do mercado
        """
        return self.get_market_sentiment_score()
    
    def analyze_single_text(self, text):
        """
        Analisa sentimento de um texto único
        
        Args:
            text (str): Texto para análise
            
        Returns:
            dict: Resultado da análise
        """
        try:
            # Usa FinBERT se disponível
            if hasattr(self, 'finbert_pipeline'):
                result = self.finbert_pipeline(text)
                sentiment = result[0]['label'].lower()
                confidence = result[0]['score']
                
                # Converte para score numérico
                if sentiment == 'positive':
                    score = 0.5 + (confidence * 0.5)
                elif sentiment == 'negative':
                    score = 0.5 - (confidence * 0.5)
                else:
                    score = 0.5
                    
                return {
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': confidence,
                    'method': 'finbert'
                }
            else:
                # Fallback para TextBlob
                from textblob import TextBlob
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                # Normaliza para 0-1
                score = (sentiment_score + 1) / 2
                
                return {
                    'sentiment': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral',
                    'score': score,
                    'confidence': abs(sentiment_score),
                    'method': 'textblob'
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Erro na análise de texto: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'method': 'error'
            }

    def get_signal_strength(self):
        """
        Retorna força do sinal de trading baseado em notícias
        
        Returns:
            float: Força do sinal (-1 a 1)
        """
        sentiment_data = self.get_market_sentiment_score()
        
        sentiment = sentiment_data["overall_sentiment"]
        confidence = sentiment_data["confidence"]
        news_count = sentiment_data["news_count"]
        
        # Ajusta força baseado na quantidade e qualidade das notícias
        volume_factor = min(news_count / 10.0, 1.0)  # Normaliza até 10 notícias
        quality_factor = confidence  # Qualidade baseada na confiança
        
        # Formula otimizada para trading
        signal_strength = sentiment * quality_factor * volume_factor
        
        # Aplicar multiplicador se há consenso forte
        positive_ratio = sentiment_data["positive_news"] / max(news_count, 1)
        negative_ratio = sentiment_data["negative_news"] / max(news_count, 1)
        
        if positive_ratio > 0.7:  # 70% das notícias positivas
            signal_strength *= 1.2
        elif negative_ratio > 0.7:  # 70% das notícias negativas
            signal_strength *= 1.2
        
        # Limita entre -1 e 1
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        logger.info(f"[SIGNAL] Força do sinal de notícias: {signal_strength:.3f}")
        
        return signal_strength

    def get_trending_topics(self):
        """
        Identifica tópicos em tendência nas notícias
        
        Returns:
            dict: Tópicos mais mencionados
        """
        try:
            recent_news = self.fetch_news(hours_back=12)
            
            if not recent_news:
                return {}
            
            # Extrai palavras-chave de todas as notícias
            all_text = " ".join([
                news.get("title", "") + " " + news.get("content", "")
                for news in recent_news
            ]).lower()
            
            # Remove stop words simples
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'this', 'that', 'these', 'those', 'bitcoin', 'btc', 'crypto'
            }
            
            # Extrai palavras
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            words = [word for word in words if word not in stop_words]
            
            # Conta frequência
            word_counts = Counter(words)
            
            # Top 10 palavras mais frequentes
            trending = dict(word_counts.most_common(10))
            
            return trending
            
        except Exception as e:
            logger.error(f"[ERROR] Erro ao identificar trending topics: {e}")
            return {}

    def force_update_news(self):
        """
        Força atualização de notícias limpando cache e garantindo dados frescos
        
        Returns:
            bool: True se atualização foi bem-sucedida, False caso contrário
        """
        try:
            logger.info("[FORCE] Forçando atualização de notícias...")
            
            # Limpa cache de notícias processadas
            self.processed_news.clear()
            logger.debug("[CACHE] Cache de notícias limpo")
            
            # Força coleta de notícias frescas (últimas 2 horas)
            fresh_news = self.fetch_news(hours_back=2)
            
            if fresh_news and len(fresh_news) > 0:
                # Processa sentimento das notícias frescas
                sentiment_score = self.get_market_sentiment_score()
                
                # Atualiza cache com dados frescos
                for news in fresh_news:
                    news_hash = hashlib.md5(news['title'].encode()).hexdigest()
                    self.processed_news[news_hash] = {
                        'news': news,
                        'timestamp': time.time(),
                        'sentiment': sentiment_score
                    }
                
                logger.info("[SUCCESS] Noticias atualizadas")
                return True
            else:
                logger.warning("[FAIL] Noticias nao atualizadas")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Erro na atualização forçada de notícias: {e}")
            logger.warning("[FAIL] Noticias nao atualizadas")
            return False