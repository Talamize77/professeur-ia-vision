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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tu es un professeur d'arabe. Analyse l'écriture sur cette photo. L'élève doit avoir écrit 10 phrases avec 'Haza'. Vérifie les voyelles. Réponds en JSON : {'status': 'SUCCESS' ou 'FAIL', 'message': 'ton commentaire'}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
