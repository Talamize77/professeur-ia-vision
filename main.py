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

# Configuration pour OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # La correction est ici : tout ce bloc doit être aligné sous 'contents'
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Tu es un professeur d'arabe strict. 
                            Vérifie ce manuscrit par rapport à cette liste exacte :
                            هَذَا مَسْجِدٌ, هَذَا كِتَابٌ, هَذَا قَلَمٌ, هَذَا مِفْتَاحٌ, هَذَا مَكْتَبٌ, 
                            هَذَا سَرِيرٌ, هَذَا كُرْسِيٌّ, هَذَا بَيْتٌ, هَذَا بَابٌ, هَذَا وَلَدٌ.

                            RÈGLES :
                            1. Si le mot est lisible et que les points/voyelles sont là, valide-le impérativement.
                            2. NE signale PAS d'erreur sur 'كُرْسِيٌّ' si la shadda et le tanwin sont visibles.
                            3. Réponds UNIQUEMENT en JSON : {"status": "SUCCESS", "message": "Bravo !"} ou {"status": "FAIL", "message": "Erreur précise..."}"""
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
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
