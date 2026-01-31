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

LISTE_CIBLE = "1. هَذَا مَسْجِدٌ, 2. هَذَا كِتَابٌ, 3. هَذَا قَلَمٌ, 4. هَذَا مِفْتَاحٌ, 5. هَذَا مَكْتَبٌ, 6. هَذَا سَرِيرٌ, 7. هَذَا كُرْسِيٌّ, 8. هَذَا بَيْتٌ, 9. هَذَا بَابٌ, 10. هَذَا وَلَدٌ"

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
                    "content": f"""Tu es un professeur d'arabe qui corrige un examen.
                    L'élève doit avoir écrit : {LISTE_CIBLE}.
                    
                    MÉTHODE DE CORRECTION :
                    1. Pour chaque ligne, regarde d'abord le mot 'هَذَا'. Est-ce que la petite dague (alif khanjariya) est présente ?
                    2. Regarde ensuite le nom. Est-ce que le double damma (tanwin) est présent à la fin ?
                    3. Si un trait manque sur ta photo en haute définition, c'est FAUX.
                    
                    FORMAT DE RÉPONSE :
                    Ligne X : [Verdict : CORRECT ou INCOMPLET]
                    - Détail : (Explique précisément quel trait de voyelle manque)."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Regarde bien les voyelles sur chaque mot de cette photo."},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" 
                            }
                        }
                    ]
                }
            ]
        )
        return {"status": "SUCCESS", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
