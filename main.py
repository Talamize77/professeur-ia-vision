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

# On utilise uniquement OpenAI pour la vision et la correction
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LISTE_OFFICIELLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        # Lecture et encodage de l'image pour l'IA
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # L'IA regarde l'image et corrige en même temps
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""Tu es un professeur d'arabe expert. 
                    L'élève doit écrire ces phrases : {LISTE_OFFICIELLE}.
                    Regarde l'image envoyée et compare son écriture manuscrite avec la liste.
                    
                    CONSIGNES :
                    1. Liste les 10 phrases.
                    2. Pour chaque phrase, dis si elle est correcte ou s'il y a une erreur de lettre ou de voyelle (harakat).
                    3. Sois très strict sur les harakats.
                    4. Réponds en français avec les mots arabes cités."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Voici mon exercice d'écriture, peux-tu le corriger ?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
