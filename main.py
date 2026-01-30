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
                        {
                            "type": "text", 
                            "text": """Tu es un expert en lecture d'arabe manuscrit. 
                            TACHE : Vérifie si l'élève a écrit ces 10 phrases : هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ.

                            CONSIGNES DE RIGUEUR :
                            1. Sois indulgent avec les formes manuscrites : si le tanwin (ٌ) ou les points sont présents, même stylisés, VALIDE le mot.
                            2. NE DIS PAS qu'il manque un tanwin si on voit un petit gribouillage au-dessus de la dernière lettre.
                            3. Si le mot 'كُرْسِيٌّ' est écrit avec ses points et une marque au-dessus, il est CORRECT.
                            
                            REPONDS UNIQUEMENT EN JSON :
                            {"status": "SUCCESS", "message": "Feedback positif"} ou {"status": "FAIL", "message": "Feedback précis"}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" 
                            }
                        }
                    ],
                }
            ],
            temperature=0  # Obligatoire pour éviter les hallucinations
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
