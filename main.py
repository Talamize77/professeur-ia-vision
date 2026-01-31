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
                    "content": f"""Tu es un inspecteur de calligraphie arabe. 
                    L'élève doit écrire exactement ceci : {LISTE_CIBLE}.
                    
                    CONSIGNES DE CORRECTION ULTRA-STRICTES :
                    1. Regarde chaque trait sur l'image.
                    2. Si une voyelle (Fatha, Damma, Kasra, Tanwin) manque sur le papier, tu DOIS la signaler comme une erreur.
                    3. Ne sois pas gentil. Si le mot n'est pas écrit à 100% comme dans la liste (harakats inclus), dis 'ERREUR'.
                    4. Pour chaque mot, écris : 'Mot X : Ce que je vois sur le papier VS ce qui est attendu'.
                    5. Si tu vois 'هذا مسجد' sans les deux damma sur le dal, c'est une faute de harakat."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspecte mon écriture avec sévérité."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
