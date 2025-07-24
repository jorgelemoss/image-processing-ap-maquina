import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint, uniform
import joblib

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer customizado para pré-processamento de dados."""
    def __init__(self, date_columns=None, categorical_threshold=20):
        self.date_columns = date_columns
        self.categorical_threshold = categorical_threshold
        self.categorical_features = None
        self.numeric_features = None

    def fit(self, X, y=None):
        # Identifica features categóricas e numéricas
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        return self

    def transform(self, X):
        print("Realizando pré-processamento de dados...")
        X_processed = X.copy()

        # Lida com colunas de data se especificadas
        if self.date_columns:
            for col in self.date_columns:
                if col in X_processed.columns:
                    try:
                        # Converte para objetos datetime, forçando erros
                        X_processed[col] = pd.to_datetime(X_processed[col], errors='coerce')
                        # Exemplo de extração de feature: ano
                        X_processed[col + '_year'] = X_processed[col].dt.year
                        # Remove a coluna de data original
                        X_processed = X_processed.drop(columns=[col])
                        print(f"Coluna de data processada: {col}")
                    except Exception as e:
                        print(f"Não foi possível processar a coluna de data {col}: {e}")

        # Lida com features categóricas (exemplo: codificação simples de rótulo ou one-hot baseada no limiar)
        if self.categorical_features:
             for col in self.categorical_features:
                 if col in X_processed.columns:
                    if X_processed[col].nunique() > self.categorical_threshold:
                        # Muitos valores únicos, talvez remover ou aplicar uma estratégia diferente
                        print(f"Aviso: A coluna categórica '{col}' tem muitos valores únicos ({X_processed[col].nunique()}). Considere tratamento alternativo.")
                        # Exemplo: Remover coluna ou manter como está por enquanto, dependendo da estratégia
                        # X_processed = X_processed.drop(columns=[col]) # Exemplo: removendo alta cardinalidade
                    else:
                         # Para demonstração, vamos mantê-las como estão antes do OneHotEncoding no pipeline do modelo
                         pass # OneHotEncoding é tratado no pipeline do modelo

        print("Pré-processamento concluído.")
        return X_processed
