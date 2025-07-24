from fastapi import FastAPI, UploadFile, File
import pandas as pd
from modeling.pipeline import main

app = FastAPI()

@app.post("/processar/")
async def processar_arquivo(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    try:
        main(df)
        return {"mensagem": "Processamento concluído com sucesso"}
    except Exception as e:
        return {"erro": str(e)}
