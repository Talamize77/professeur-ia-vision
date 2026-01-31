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
        
        # Azure extrait le texte brut
        poller = document_client.begin_analyze_document("prebuilt-read", image_data)
        result = poller.result()
        texte_lu = " ".join([l.content for p in result.pages for l in p.lines])

        # OpenAI en mode "Inspection de police"
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""Tu es un correcteur d'arabe extrêmement rigoureux. 
                Voici la liste parfaite : {LISTE_OFFICIELLE}. 
                Voici ce que tu dois corriger (texte extrait de la photo) : '{texte_lu}'.
                
                CONSIGNES STRICTES :
                1. Compare chaque mot extrait avec le mot correspondant dans la liste.
                2. Si un 'Tanwin' (double voyelle) manque, c'est une erreur.
                3. Si une lettre est mal lue ou absente, signale-le.
                4. Ne dis PAS 'parfait' ou 'excellent' si le texte lu est approximatif. 
                5. Si le texte extrait est très court ou illisible, demande une photo plus nette.
                6. Présente tes corrections sous forme de liste numérotée (1 à 10)."""},
                {"role": "user", "content": "Analyse précisément mon écriture."}
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
