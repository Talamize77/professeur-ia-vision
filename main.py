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

# Configuration pour OpenAI (plus stable pour la vision)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # Remplace cette partie dans ton main.py
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Tu es un correcteur expert en langue arabe. 
                            Ta mission : Compter et vérifier précisément 10 phrases manuscrites.
                            
                            LES 10 PHRASES ATTENDUES :
                            1. هَذَا مَسْجِدٌ  2. هَذَا كِتَابٌ  3. هَذَا قَلَمٌ  4. هَذَا مِفْتَاحٌ  5. هَذَا مَكْتَبٌ 
                            6. هَذَا سَرِيرٌ  7. هَذَا كُرْسِيٌّ  8. هَذَا بَيْتٌ  9. هَذَا بَابٌ  10. هَذَا وَلَدٌ

                            INSTRUCTIONS CRITIQUES :
                            1. Analyse l'image ligne par ligne, de droite à gauche.
                            2. Identifie chaque mot et chaque voyelle (tanwin final ٌ ).
                            3. Ne conclus pas qu'il manque des phrases avant d'avoir scanné TOUTE l'image.
                            
                            RÉPONSE (JSON UNIQUEMENT) :
                            - Si les 10 sont là : {"status": "SUCCESS", "message": "Parfait ! Les 10 phrases sont bien écrites et complètes."}
                            - Si une phrase manque : {"status": "FAIL", "message": "Il semble que tu as oublié la phrase : [Nom du mot]."}
                            - Si une voyelle est fausse : {"status": "FAIL", "message": "Vérifie la voyelle finale du mot [Nom du mot]."}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ],
                }
            ],
            temperature=0.1 # Rend l'IA très factuelle et moins distraite
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
