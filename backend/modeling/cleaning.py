def basic_data_cleaning(df):
    """Realiza passos básicos de limpeza de dados."""
    print("Realizando limpeza básica de dados...")
    # Exemplo de limpeza: remover colunas com muitos valores ausentes
    threshold = len(df) * 0.5
    df_cleaned = df.dropna(axis=1, thresh=threshold)
    print(f"Colunas originais: {df.shape[1]}, Colunas após limpeza: {df_cleaned.shape[1]}")
    return df_cleaned