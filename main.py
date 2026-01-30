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
                        {
                            "type": "text", 
                            "text": """Tu es un professeur de calligraphie arabe très strict. 
                            Vérifie ce manuscrit par rapport à cette liste exacte de 10 mots :
                            1. هَذَا مَسْجِدٌ  2. هَذَا كِتَابٌ  3. هَذَا قَلَمٌ  4. هَذَا مِفْتَاحٌ  5. هَذَا مَكْتَبٌ 
                            6. هَذَا سَرِيرٌ  7. هَذَا كُرْسِيٌّ  8. هَذَا بَيْتٌ  9. هَذَا بَابٌ  10. هَذَا وَلَدٌ

                            MÉTHODE DE CORRECTION :
                            - Regarde chaque mot individuellement.
                            - Ne critique pas un mot si les points et les voyelles (shadda, tanwin) sont présents, même s'ils sont écrits de façon manuscrite.
                            - Si un mot est écrit correctement, NE DIS PAS qu'il est faux.
                            
                            RÉPONSE JSON :
                            {"status": "SUCCESS" ou "FAIL", "message": "ton feedback en français"}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                        }
                    ],
                }
            ],
            temperature=0, # Force l'IA à être 100% factuelle (pas de créativité)
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
