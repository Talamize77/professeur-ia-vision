import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

app = FastAPI()

# Autorisation pour que Systeme.io puisse parler au serveur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation avec tes variables Render
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

# LA VÉRITÉ EST ICI : L'IA comparera la photo à ces mots uniquement
LISTE_OFFICIELLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # 1. Azure scanne les traits sur la photo
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_lu = " ".join([l.content for p in result.pages for l in p.lines])

        # 2. OpenAI compare le texte lu avec la LISTE_OFFICIELLE
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Tu es un professeur d'arabe. On vient de scanner une photo d'exercice. Voici les mots que l'élève devait écrire : {LISTE_OFFICIELLE}. Voici ce que j'ai lu sur sa photo : '{texte_lu}'. Liste précisément les erreurs (lettres manquantes ou harakats faux). Sois encourageant mais très précis sur la correction."},
                {"role": "user", "content": "Corrige mon écriture s'il te plaît."}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
