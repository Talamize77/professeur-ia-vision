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

# On utilise l'interface OpenAI qui est compatible avec DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # CHANGEMENT ICI : On utilise le modèle chat qui supporte la vision chez DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tu es un expert en calligraphie arabe. Analyse cette photo. L'élève doit avoir écrit 10 phrases commençant par 'Haza'. Vérifie l'orthographe et les voyelles. Réponds UNIQUEMENT en JSON : {'status': 'SUCCESS' ou 'FAIL', 'message': 'ton commentaire court'}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return {"status": "ERROR", "message": "Erreur technique : " + str(e)}
