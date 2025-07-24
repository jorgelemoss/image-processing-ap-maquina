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

def create_diagnostic_plots(df):
    """Gera visualizações diagnósticas dos dados"""
    if df is None or 'obesity_rate' not in df.columns:
        print("Dados inválidos ou coluna 'obesity_rate' não encontrada para plotagem")
        return

    try:
        df['obesity_rate'] = pd.to_numeric(df['obesity_rate'], errors='coerce')  # Corrigido typo aqui
        year_col = None
        if 'yearstart_year' in df.columns:
            year_col = 'yearstart_year'
        elif 'yearend_year' in df.columns:
            year_col = 'yearend_year'
        elif 'yearstart' in df.columns:
            year_col = 'yearstart'

        plt.figure(figsize=(18, 14))
        plt.subplot(2, 2, 1)
        sns.histplot(df['obesity_rate'].dropna(), kde=True, bins=20)
        plt.title('Distribuição da Taxa de Obesidade')
        plt.xlabel('Taxa de Obesidade (%)')
        plt.ylabel('Frequência')

        plt.subplot(2, 2, 2)
        if year_col and df[year_col].nunique() > 1:
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            yearly_data = df.groupby(year_col)['obesity_rate'].agg(['mean', 'median', 'std']).dropna()
            plt.plot(yearly_data.index, yearly_data['median'], marker='o')
            plt.fill_between(yearly_data.index,
                             yearly_data['median'] - yearly_data['std'],
                             yearly_data['median'] + yearly_data['std'],
                             alpha=0.2)
            plt.title('Evolução Anual da Obesidade')
            plt.xlabel('Ano')
            plt.ylabel('Taxa de Obesidade (Mediana)')
        else:
            plt.text(0.5, 0.5, 'Dados insuficientes para análise temporal', ha='center')
            plt.title('Evolução Anual da Obesidade')

        # CORREÇÃO PRINCIPAL: Identação correta deste bloco
        plt.subplot(2, 2, 3)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_col = None
        for col in categorical_cols:
            if col != 'locationdesc' and 1 < df[col].nunique() <= 20:
                cat_col = col
                break

        # Identação corrigida abaixo
        if cat_col is None and len(categorical_cols) > 0:
            for col in categorical_cols:
                if 1 < df[col].nunique() <= 30:
                    cat_col = col
                    break

        if cat_col:
            if df[cat_col].nunique() > 10:
                top_cats = df[cat_col].value_counts().nlargest(10).index
                filtered_df = df[df[cat_col].isin(top_cats)]
            else:
                filtered_df = df

            if not filtered_df.empty and filtered_df['obesity_rate'].notnull().any():
                sns.boxplot(
                    x=cat_col,
                    y='obesity_rate',
                    data=filtered_df.dropna(subset=['obesity_rate', cat_col])
                )
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Obesidade por {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel('Taxa de Obesidade')
            else:
                plt.text(0.5, 0.5, 'Dados insuficientes para plotar', ha='center')
                plt.title(f'Obesidade por {cat_col}')
        else:
            plt.text(0.5, 0.5, 'Nenhuma coluna categórica disponível', ha='center')
            plt.title('Obesidade por Categoria')

        plt.subplot(2, 2, 4)
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                        vmin=-1, vmax=1, annot_kws={"size":8})
            plt.title('Matriz de Correlação')
        else:
            plt.text(0.5, 0.5, 'Dados numéricos insuficientes para matriz de correlação', ha='center')
            plt.title('Matriz de Correlação')

        plt.tight_layout()
        plt.savefig('analise_obesidade_diagnostico.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Visualizações salvas como 'analise_obesidade_diagnostico.png'")
    except Exception as e:
        print(f"❌ Erro ao gerar visualizações: {e}")
        traceback.print_exc()
