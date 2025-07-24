import pandas as pd

def feature_engineering(df):
    """Realiza engenharia de features."""
    print("Realizando engenharia de features...")
    df_engineered = df.copy()

    # Exemplo de engenharia de features: diferença entre ano de início e fim
    if 'yearstart_year' in df_engineered.columns and 'yearend_year' in df_engineered.columns:
        df_engineered['year_difference'] = df_engineered['yearend_year'] - df_engineered['yearstart_year']
        print("Criada feature 'year_difference'.")

    # Exemplo: Termo de interação (se colunas relevantes existirem e forem numéricas)
    # Verifica se as colunas 'Data_Value' e 'Sample_Size' são numéricas antes de multiplicar
    if all(col in df_engineered.columns and pd.api.types.is_numeric_dtype(df_engineered[col]) for col in ['Data_Value', 'Sample_Size']):
        df_engineered['value_sample_interaction'] = df_engineered['Data_Value'] * df_engineered['Sample_Size']
        print("Criada feature 'value_sample_interaction'.")
    else:
        # Lida com potenciais tipos não numéricos, se necessário, por exemplo, convertendo ou pulando a interação
         if 'Data_Value' in df_engineered.columns and 'Sample_Size' in df_engineered.columns:
            print("Pulando a criação da feature 'value_sample_interaction' devido a tipos não numéricos ou colunas ausentes.")


    print("Engenharia de features concluída.")
    return df_engineered
