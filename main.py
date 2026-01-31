import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Connexion aux services (Clés configurées sur Render)
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # 1. ANALYSE AZURE (Lecture chirurgicale de l'arabe)
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_extrait = " ".join([line.content for page in result.pages for line in page.lines])

        # 2. DECISION OPENAI (Comparaison intelligente)
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un professeur d'arabe. Valide si les 10 phrases sont présentes. Sois indulgent sur le manuscrit mais strict sur les voyelles finales (tanwin)."},
                {"role": "user", "content": f"Texte lu par l'OCR : {texte_extrait}. Liste cible : هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ."}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
