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
                    "content": f"""Tu es un expert en analyse de manuscrits arabes. Ton but est de vérifier la présence RÉELLE des voyelles (harakats).
                    
                    RÈGLES D'ANALYSE :
                    1. Pour chaque mot, cherche d'abord les petits traits des voyelles sur l'image.
                    2. Si tu ne vois pas DISTINCTEMENT un trait (comme le Tanwin sur la dernière lettre), tu DOIS considérer qu'il est absent.
                    3. Ne complète PAS les mots dans ton esprit. Si le papier est nu, le mot est incomplet.
                    4. Structure ta réponse ainsi :
                       - Mot attendu : [Mot de la liste {LISTE_CIBLE}]
                       - Analyse visuelle : (Décris ce que tu vois : 'Je vois les lettres mais aucun trait au-dessus', etc.)
                       - Verdict : (Correct / Incomplet / Erreur)"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse très précisément les traits de voyelles sur ce scan."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
