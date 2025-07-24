from .cleaning import basic_data_cleaning
from .preprocessing import DataPreprocessor
from .engineering import feature_engineering
from .modeling import build_model_pipeline, compare_models, compare_optimized_models
from .optimization import optimize_hyperparameters
from .visualization import create_diagnostic_plots

import pandas as pd
import traceback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import joblib


def main(df):
    """Fluxo principal de execução otimizado para velocidade"""
    try:
        # 1. Limpeza básica de dados
        df_cleaned = basic_data_cleaning(df)

        # 2. Identificar colunas de data automaticamente
        date_cols = [col for col in df_cleaned.columns
                     if 'date' in col.lower() or 'year' in col.lower() or 'time' in col.lower()]
        print(f"Colunas de data identificadas: {date_cols}")

        # 3. Pré-processamento
        preprocessor = DataPreprocessor(date_columns=date_cols, categorical_threshold=20)
        X = df_cleaned.drop(columns=['obesity_rate'], errors='ignore')
        y = df_cleaned['obesity_rate']

        # Converter para numérico se necessário
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')

        # 4. Engenharia de features
        X_preprocessed = preprocessor.fit_transform(X)
        X_engineered = feature_engineering(X_preprocessed)

        # Verificar vazamento de dados
        if 'obesity_rate' in X_engineered.columns:
            print("⚠️ ATENÇÃO: Variável alvo encontrada nas features. Removendo...")
            X_engineered = X_engineered.drop(columns=['obesity_rate'])

        # 5. Dividir dados (usando amostra se conjunto for muito grande)
        if len(X_engineered) > 10000:
            sample_size = min(10000, len(X_engineered))
            X_sample = X_engineered.sample(sample_size, random_state=42)
            y_sample = y.loc[X_sample.index]
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_sample, test_size=0.2, random_state=42)
            print(f"Usando amostra de {sample_size} registros para modelagem")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_engineered, y, test_size=0.2, random_state=42)

        # 6. Construir pipeline de modelagem
        model_pipeline, numeric_features, categorical_features = build_model_pipeline(X_train)
        preprocessor_pipe = model_pipeline.named_steps['preprocessor']

        # 7. Comparar modelos com amostra
        results_df = compare_models(X_train, y_train, X_test, y_test, preprocessor_pipe)

        # 8. Otimizar apenas os 2 melhores modelos
        top_models = results_df.head(2)['Modelo'].tolist()
        print(f"\nOtimizando os melhores modelos: {top_models}")

        # Grades de parâmetros simplificadas
        param_grids = {
            'Random Forest': {
                'regressor__n_estimators': randint(50, 200),
                'regressor__max_depth': [None, 5, 10],
                'regressor__min_samples_split': randint(2, 10)
            },
            'Gradient Boosting': {
                'regressor__n_estimators': randint(50, 150),
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': randint(3, 8)
            },
            'Linear Regression': {}  # Sem otimização
        }

        best_models = {}
        for model_name in top_models:
            if model_name == 'Random Forest':
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'Gradient Boosting':
                base_model = GradientBoostingRegressor(random_state=42)
            elif model_name == 'Linear Regression':
                base_model = LinearRegression(n_jobs=-1)
            else:
                continue

            best_model, best_params = optimize_hyperparameters(
                base_model,
                param_grids.get(model_name, {}),
                preprocessor_pipe,
                X_train,
                y_train
            )
            best_models[model_name] = best_model

        # 9. Comparar modelos otimizados
        optimized_results = compare_optimized_models(best_models, X_test, y_test)

        # 10. Salvar o melhor modelo
        best_model_name = optimized_results.iloc[0]['Modelo']
        best_model = best_models[best_model_name]
        joblib.dump(best_model, 'best_model.pkl')
        print(f"Melhor modelo ({best_model_name}) salvo como 'best_model.pkl'")

        # 11. Visualizações diagnósticas
        create_diagnostic_plots(df_cleaned)

        print("\n" + "="*80)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERRO NO PROCESSO PRINCIPAL")
        print("="*80)
        print(f"Erro: {str(e)}")
        traceback.print_exc()
# if __name__ == "__main__":
#     import pandas as pd
#     import joblib
#     from scipy.stats import randint, uniform
#     import traceback
#     import re

#     # Carregar dados
#     try:
#         # Substitua com seu carregamento real de dados
#         # df = pd.read_csv('obesity_data.csv')

#         # Dados de exemplo otimizados para testes rápidos
#         print("Usando dados de exemplo otimizados...")
#         data = {
#             'YearStart': list(range(2010, 2020)) * 10,
#             'YearEnd': list(range(2010, 2020)) * 10,
#             'LocationDesc': ['Alabama', 'Alaska', 'Arizona'] * 33 + ['California'],
#             'StratificationCategory1': ['Total', 'Gender', 'Education'] * 33 + ['Income'],
#             'Data_Value': np.random.uniform(20, 40, 100),
#             'Sample_Size': np.random.randint(100, 1000, 100)
#         }
#         df = pd.DataFrame(data)
#         df.rename(columns={'Data_Value': 'obesity_rate'}, inplace=True)

#         # Executar o fluxo principal
#         main(df)

#     except Exception as e:
#         print(f"Erro ao carregar dados: {str(e)}")
#         traceback.print_exc()