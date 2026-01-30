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

# Configuration DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # Structure corrigée pour DeepSeek Vision
        response = client.chat.completions.create(
            model="deepseek-reasoner", # On utilise le modèle le plus performant
            messages=[
                {
                    "role": "user",
                    "content": f"Analyse cette photo d'écriture arabe. L'élève doit avoir écrit : هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ. Réponds en JSON : {{'status': 'SUCCESS' ou 'FAIL', 'message': 'ton commentaire'}}. Voici l'image en base64 : data:image/jpeg;base64,{base64_image}"
                }
            ]
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
