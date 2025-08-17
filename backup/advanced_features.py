"""
Sistema Avançado de Feature Engineering
Cria features inteligentes para maximizar precisão da IA
"""

import pandas as pd
import numpy as np
import talib
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Engenheiro de features avançado com seleção automática
    e detecção de regime de mercado
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.regime_detector = MarketRegimeDetector()
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria conjunto completo de features avançadas
        """
        print("[FIX] Criando features avançadas...")
        
        # Features básicas
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        
        # Features técnicas avançadas
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_trend_features(df)
        df = self._add_pattern_features(df)
        
        # Features de microestrutura
        df = self._add_microstructure_features(df)
        
        # Features de regime de mercado
        df = self._add_regime_features(df)
        
        # Features estatísticas
        df = self._add_statistical_features(df)
        
        # Features de machine learning
        df = self._add_ml_features(df)
        
        return df.fillna(method='ffill').fillna(0)
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas em preço"""
        
        # Retornos em múltiplos timeframes
        for period in [1, 5, 15, 30, 60]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Gaps de preço
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = (df['close'] > df['open'].shift(1)).astype(int)
        
        # Razões de preço
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['open'] - df['close']) / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Preços normalizados
        for period in [20, 50, 100]:
            df[f'price_position_{period}'] = (df['close'] - df['close'].rolling(period).min()) / \
                                           (df['close'].rolling(period).max() - df['close'].rolling(period).min())
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas em volume"""
        
        # Volume normalizado
        for period in [20, 50]:
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # On Balance Volume
        df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        
        # Volume Weighted Average Price
        for period in [20, 50]:
            df[f'vwap_{period}'] = (df['close'] * df['volume']).rolling(period).sum() / \
                                 df['volume'].rolling(period).sum()
            df[f'vwap_distance_{period}'] = (df['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de momentum"""
        
        # RSI em múltiplos períodos
        for period in [14, 21, 30]:
            df[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'].values)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = hist
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        
        # Williams %R
        for period in [14, 21]:
            df[f'williams_r_{period}'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        # Commodity Channel Index
        for period in [14, 20]:
            df[f'cci_{period}'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        # Rate of Change
        for period in [10, 20, 30]:
            df[f'roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade"""
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
        
        # Average True Range
        for period in [14, 21]:
            df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Volatilidade realizada
        for period in [20, 50]:
            df[f'realized_vol_{period}'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # Keltner Channels
        for period in [20, 50]:
            ema = df['close'].ewm(span=period).mean()
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            df[f'keltner_upper_{period}'] = ema + (2 * atr)
            df[f'keltner_lower_{period}'] = ema - (2 * atr)
            df[f'keltner_position_{period}'] = (df['close'] - df[f'keltner_lower_{period}']) / \
                                             (df[f'keltner_upper_{period}'] - df[f'keltner_lower_{period}'])
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de tendência"""
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'sma_distance_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'ema_distance_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # Crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_cross_12_26'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'].values, df['low'].values)
        df['sar_trend'] = (df['close'] > df['sar']).astype(int)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de padrões de candlestick"""
        
        # Padrões de candlestick do TA-Lib
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern in patterns:
            try:
                pattern_func = getattr(talib, pattern)
                df[pattern.lower()] = pattern_func(df['open'].values, df['high'].values, 
                                                 df['low'].values, df['close'].values)
            except:
                continue
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de microestrutura de mercado"""
        
        # Spread simulado (high-low como proxy)
        df['spread'] = df['high'] - df['low']
        df['spread_ratio'] = df['spread'] / df['close']
        
        # Imbalance de volume (simulado)
        df['volume_imbalance'] = df['volume'] * np.sign(df['close'] - df['open'])
        
        # Tick direction
        df['tick_direction'] = np.sign(df['close'].diff())
        
        # Volume at Price levels (simulado)
        for period in [10, 20]:
            df[f'volume_at_high_{period}'] = df['volume'].rolling(period).apply(
                lambda x: x[df['high'].rolling(period).idxmax()]
            )
            df[f'volume_at_low_{period}'] = df['volume'].rolling(period).apply(
                lambda x: x[df['low'].rolling(period).idxmin()]
            )
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de regime de mercado"""
        
        # Detectar regime usando volatilidade
        vol_short = df['close'].pct_change().rolling(20).std()
        vol_long = df['close'].pct_change().rolling(100).std()
        df['volatility_regime'] = (vol_short > vol_long * 1.5).astype(int)
        
        # Detectar regime usando tendência
        sma_20 = df['close'].rolling(20).mean()
        sma_100 = df['close'].rolling(100).mean()
        df['trend_regime'] = (sma_20 > sma_100).astype(int)
        
        # Regime combinado
        df['market_regime'] = df['volatility_regime'] + df['trend_regime'] * 2
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features estatísticas"""
        
        # Skewness e Kurtosis
        for period in [20, 50]:
            returns = df['close'].pct_change()
            df[f'skewness_{period}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(period).kurt()
        
        # Z-Score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / std
        
        # Percentile Rank
        for period in [20, 50]:
            df[f'percentile_rank_{period}'] = df['close'].rolling(period).rank(pct=True)
        
        return df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features baseadas em machine learning"""
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)
        
        # Rolling statistics
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            df[f'return_mean_{period}'] = returns.rolling(period).mean()
            df[f'return_std_{period}'] = returns.rolling(period).std()
            df[f'return_min_{period}'] = returns.rolling(period).min()
            df[f'return_max_{period}'] = returns.rolling(period).max()
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, 
                           method: str = 'mutual_info', k: int = 50) -> pd.DataFrame:
        """
        Seleciona as melhores features automaticamente
        """
        print(f"[TARGET] Selecionando {k} melhores features usando {method}...")
        
        # Preparar dados
        feature_cols = [col for col in df.columns if col != target_col and not col.startswith('target')]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remover features com variância zero
        non_zero_var = X.var() > 0
        X = X.loc[:, non_zero_var]
        
        # Seleção de features
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        elif method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            selected_features = feature_importance.nlargest(k).index.tolist()
            self.selected_features = selected_features
            return df[[target_col] + selected_features]
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Armazenar features selecionadas
        self.selected_features = selected_features
        self.feature_importance = dict(zip(selected_features, selector.scores_[selector.get_support()]))
        
        return df[[target_col] + selected_features]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features selecionadas"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(list(self.feature_importance.items()), 
                                   columns=['feature', 'importance'])
        return importance_df.sort_values('importance', ascending=False)

class MarketRegimeDetector:
    """
    Detector de regime de mercado usando HMM e clustering
    """
    
    def __init__(self):
        self.regimes = None
        
    def detect_regimes(self, returns: pd.Series, n_regimes: int = 3) -> pd.Series:
        """
        Detecta regimes de mercado baseado em retornos
        """
        try:
            # Preparar dados
            X = returns.values.reshape(-1, 1)
            
            # Ajustar modelo
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            regimes = gmm.fit_predict(X)
            
            # Ordenar regimes por volatilidade
            regime_stats = []
            for i in range(n_regimes):
                mask = regimes == i
                vol = returns[mask].std()
                regime_stats.append((i, vol))
            
            regime_stats.sort(key=lambda x: x[1])  # Ordenar por volatilidade
            regime_mapping = {old: new for new, (old, _) in enumerate(regime_stats)}
            
            # Mapear regimes
            regimes_mapped = pd.Series([regime_mapping[r] for r in regimes], index=returns.index)
            
            return regimes_mapped
            
        except ImportError:
            # Fallback simples baseado em volatilidade
            vol = returns.rolling(20).std()
            low_vol = vol.quantile(0.33)
            high_vol = vol.quantile(0.67)
            
            regimes = pd.Series(1, index=returns.index)  # Regime normal
            regimes[vol <= low_vol] = 0  # Regime baixa volatilidade
            regimes[vol >= high_vol] = 2  # Regime alta volatilidade
            
            return regimes

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50000,
        'low': np.random.randn(1000).cumsum() + 50000,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Criar target (próximo retorno)
    df['target'] = df['close'].pct_change().shift(-1)
    
    # Inicializar feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    print("[FIX] Criando features avançadas...")
    df_features = feature_engineer.create_advanced_features(df)
    
    print(f"[DATA] Features criadas: {len(df_features.columns)}")
    
    # Selecionar melhores features
    df_selected = feature_engineer.select_best_features(df_features, 'target', k=20)
    
    print(f"[TARGET] Features selecionadas: {len(df_selected.columns) - 1}")
    
    # Mostrar importância
    importance = feature_engineer.get_feature_importance()
    print("\n[UP] Top 10 Features mais importantes:")
    print(importance.head(10)) 