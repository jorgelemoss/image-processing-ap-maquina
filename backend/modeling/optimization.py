import numpy as np
import pandas as pd
import traceback
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def compare_optimized_models(best_models, X_test, y_test):
    """Compara a performance de modelos otimizados."""
    print("\nComparando modelos otimizados...")
    results = []

    for name, model in best_models.items():
        try:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results.append({'Modelo': name, 'RMSE': rmse, 'R²': r2, 'MAE': mae})
            print(f"  - {name}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

        except Exception as e:
            print(f"  - Erro ao avaliar o modelo otimizado {name}: {str(e)}")
            results.append({'Modelo': name, 'RMSE': np.nan, 'R²': np.nan, 'MAE': np.nan})


    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\nResultados da comparação de modelos otimizados:")
    display(results_df)
    return results_df

def optimize_hyperparameters(base_model, param_grid, preprocessor_pipe, X_train, y_train):
    """Otimiza hiperparâmetros para um dado modelo usando RandomizedSearchCV."""
    print(f"Otimizando hiperparâmetros para {type(base_model).__name__}...")

    # Cria um pipeline com o preprocessor e o modelo base
    pipeline = Pipeline(steps=[('preprocessor', preprocessor_pipe),
                               ('regressor', base_model)])

    if not param_grid:
        print("Nenhuma grade de parâmetros fornecida para otimização. Usando o modelo base.")
        return pipeline.fit(X_train, y_train), {}

    try:
        # Usa RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20, # Número de configurações de parâmetros a serem amostradas
            cv=KFold(n_splits=5, shuffle=True, random_state=42), # Usando validação cruzada KFold
            scoring='neg_root_mean_squared_error', # Otimiza para RMSE
            random_state=42,
            n_jobs=-1, # Usa todos os núcleos disponíveis
            verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"Melhores parâmetros encontrados: {random_search.best_params_}")
        print(f"Melhor RMSE de validação cruzada: {-random_search.best_score_:.4f}") # Observação: é o RMSE negativo

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        print(f"Erro durante a otimização de hiperparâmetros: {str(e)}")
        traceback.print_exc()
        return pipeline.fit(X_train, y_train), {} # Retorna o pipeline do modelo base se a otimização falhar