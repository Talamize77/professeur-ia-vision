import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

app = FastAPI()

# Autorise la connexion depuis Systeme.io
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration via les variables d'environnement Render
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...), phrases_cible: str = Form("هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ")):
    try:
        image_data = await file.read()
        
        # 1. Azure lit l'image
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_extrait = " ".join([line.content for page in result.pages for line in page.lines])

        # 2. OpenAI compare et corrige
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un professeur d'arabe. Compare le texte lu avec la liste cible. Réponds de manière encourageante."},
                {"role": "user", "content": f"Texte extrait : {texte_extrait}. Liste cible : {phrases_cible}"}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
