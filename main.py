import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LISTE_CIBLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""Tu es un professeur d'arabe. 
                    L'élève doit écrire ces phrases : {LISTE_CIBLE}.
                    
                    INSTRUCTIONS SIMPLES :
                    1. Analyse l'image et compare chaque mot écrit avec la liste.
                    2. Pour chaque phrase, donne un verdict court : 'Correct', 'Erreur de lettre' ou 'Harakat manquante'.
                    3. Sois précis : si une voyelle manque, ce n'est pas correct.
                    4. Réponds avec une liste numérotée de 1 à 10."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Voici mon exercice, corrige-le simplement phrase par phrase."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
