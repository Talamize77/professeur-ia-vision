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

# Liste exacte attendue
LISTE_PARFAITE = [
    "هَذَا مَسْجِدٌ", "هَذَا كِتَابٌ", "هَذَا قَلَمٌ", "هَذَا مِفْتَاحٌ", "هَذَا مَكْتَبٌ",
    "هَذَا سَرِيرٌ", "هَذَا كُرْسِيٌّ", "هَذَا بَيْتٌ", "هَذَا بَابٌ", "هَذَا وَلَدٌ"
]

@app.post("/analyser-ecriture")
async def analyser_ecriture(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # ÉTAPE 1 : On demande à l'IA de recopier bêtement ce qu'elle voit
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "Tu es un scanner de texte. Recopie exactement le texte arabe que tu vois sur l'image, ligne par ligne. N'ajoute AUCUNE voyelle que tu ne vois pas clairement. Si un mot n'a pas de voyelle, écris-le sans voyelle."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                    ]
                }
            ]
        )
        
        texte_extrait = response.choices[0].message.content.split('\n')
        
        # ÉTAPE 2 : Comparaison stricte par le code
        resultats = []
        for i, phrase_attendue in enumerate(LISTE_PARFAITE):
            phrase_eleve = texte_extrait[i] if i < len(texte_extrait) else "Manquant"
            if phrase_eleve.strip() == phrase_attendue.strip():
                resultats.append(f"{i+1}. ✅ Parfait")
            else:
                resultats.append(f"{i+1}. ❌ Erreur : Tu as écrit '{phrase_eleve}' au lieu de '{phrase_attendue}'")

        return {"status": "SUCCESS", "message": "\n".join(resultats)}
        
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
