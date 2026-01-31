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

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"), 
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

LISTE_OFFICIELLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # 1. Extraction du texte par Azure
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_lu = " ".join([l.content for p in result.pages for l in p.lines])

        # 2. OpenAI compare et affiche ce qu'il a lu
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""Tu es un correcteur strict. 
                LISTE DE RÉFÉRENCE : {LISTE_OFFICIELLE}
                
                MISSION :
                1. Commence ta réponse par : 'TEXTE DÉTECTÉ SUR LA PHOTO : [insère ici le texte lu par Azure]'.
                2. Ensuite, compare ce texte avec la LISTE DE RÉFÉRENCE.
                3. Pour chaque mot, indique s'il est correct ou s'il y a une erreur (lettre, voyelle ou mot manquant).
                4. Ne sois PAS complaisant. Si le texte détecté est différent de la référence, signale l'erreur."""},
                {"role": "user", "content": f"Voici le texte brut extrait de la photo : '{texte_lu}'. Corrige-le par rapport à la liste officielle."}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
