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

# Utilisation de l'IA OpenAI en direct pour la vision
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LISTE_CIBLE = "هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ"

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        # Lecture de l'image
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Appel à l'IA avec le réglage de détail "HIGH"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""Tu es un professeur d'arabe extrêmement rigoureux. 
                    Ton travail est de vérifier si l'élève a écrit exactement cette liste : {LISTE_CIBLE}.
                    
                    ATTENTION : 
                    1. Tu dois vérifier chaque trait, surtout les voyelles (harakats) et le tanwin à la fin des mots.
                    2. Si le mot 'هَذَا' n'a pas son alif khanjariya (petite dague au-dessus) ou ses voyelles, c'est FAUX.
                    3. Si tu ne vois pas de voyelles sur un mot, ne les invente pas. Déclare le mot 'Incomplet'.
                    4. Réponds avec une liste numérotée de 1 à 10."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse mon écriture en zoomant sur chaque voyelle. Sois très sévère."},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # Force l'IA à voir les plus petits détails
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
