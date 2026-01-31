import os
from fastapi import FastAPI, UploadFile, File
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

# Utilisation de tes variables Render
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

# LA LISTE OFFICIELLE REVIENT ICI POUR PLUS DE SÉCURITÉ
LISTE_OFFICIELLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # 1. Azure lit la photo
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_lu = " ".join([l.content for p in result.pages for l in p.lines])

        # 2. OpenAI compare avec la liste fixe
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Tu es un prof d'arabe. Voici les mots corrects : {LISTE_OFFICIELLE}. Compare avec ce que l'élève a écrit : {texte_lu}. Sois précis sur les erreurs de lettres ou de harakats."},
                {"role": "user", "content": "Analyse ma photo."}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
