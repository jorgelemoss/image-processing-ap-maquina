import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def compare_models(X_train, y_train, X_test, y_test, preprocessor_pipe):
    """Compara a performance de diferentes modelos de regressão."""
    print("\nComparando diferentes modelos...")
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Support Vector Machine': SVR(kernel='rbf'),
        'ElasticNet': ElasticNet(random_state=42),
        'KNeighbors Regressor': KNeighborsRegressor()

    }

    results = []

    for name, model in models.items():
        try:
            # Constrói um pipeline para cada modelo usando o preprocessor fornecido
            pipeline = Pipeline(steps=[('preprocessor', preprocessor_pipe),
                                       ('regressor', model)])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results.append({'Modelo': name, 'RMSE': rmse, 'R²': r2, 'MAE': mae})
            print(f"  - {name}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

        except Exception as e:
            print(f"  - Erro ao treinar/avaliar {name}: {str(e)}")
            results.append({'Modelo': name, 'RMSE': np.nan, 'R²': np.nan, 'MAE': np.nan})


    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\nResultados da comparação de modelos:")
    display(results_df)
    return results_df


def build_model_pipeline(X_train):
    """Constrói pipeline de modelagem dinâmico baseado nos dados"""
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline, numeric_features, categorical_features


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
