import os
import base64
import json
import re
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

# Utilise ta clé API (elle est compatible OpenAI/DeepSeek selon ton choix)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tu es un professeur d'arabe. Analyse cette image. L'élève doit avoir écrit 10 phrases avec 'Haza'. Vérifie les voyelles finales. Réponds UNIQUEMENT au format JSON strict : {'status': 'SUCCESS', 'message': 'ton compliment'} ou {'status': 'FAIL', 'message': 'ton conseil'}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        
        # Sécurité pour extraire le JSON proprement
        texte_brut = response.choices[0].message.content
        match = re.search(r'\{.*\}', texte_brut, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"status": "FAIL", "message": "L'IA n'a pas pu formater sa réponse. Réessaie."}

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
