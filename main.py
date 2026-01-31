import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utilisation de tes variables Render déjà configurées
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...), phrases_cible: str = Form(...)):
    try:
        image_data = await file.read()
        
        # Lecture par Azure
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_extrait = " ".join([line.content for page in result.pages for line in page.lines])

        # Correction STRICTE par OpenAI
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """Tu es un correcteur d'arabe strict. 
                1. Analyse le texte extrait de la photo par rapport à la liste cible.
                2. Si des mots manquent ou sont mal écrits (fautes de harakats ou de lettres), liste précisément les erreurs.
                3. Ne dis JAMAIS 'Bravo' s'il y a des erreurs.
                4. Sois court et précis. Si tout est parfait, réponds juste 'Excellent, tout est correct'."""},
                {"role": "user", "content": f"Texte lu sur la photo : {texte_extrait}. Liste officielle à vérifier : {phrases_cible}"}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
