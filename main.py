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
                    "content": f"""Tu es un professeur d'arabe extrêmement pointilleux. 
                    L'élève doit écrire : {LISTE_CIBLE}.
                    
                    RÈGLES D'INSPECTION :
                    1. Analyse chaque phrase en deux temps : d'abord le mot 'هَذَا', puis le nom qui suit.
                    2. Si 'هَذَا' n'a pas sa petite dague (alif khanjariya) ou ses harakats sur l'image, c'est une ERREUR.
                    3. Si le Tanwin (double voyelle finale) manque sur le nom, c'est une ERREUR.
                    4. Ne sois pas indulgent. Si un seul trait manque sur la photo, le verdict est 'Incomplet'.
                    5. Réponds par une liste numérotée claire."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Vérifie chaque trait, surtout sur le mot 'هذا' à chaque ligne."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
