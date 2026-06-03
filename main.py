from fastapi import FastAPI, File, UploadFile
import whisper
import tempfile
import os

model = whisper.load_model("small")  

app = FastAPI()

@app.post("/transcrever")
async def transcrever(file: UploadFile = File(...)):
    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    # Processa o áudio
    result = model.transcribe(
        tmp_path, 
        language="pt",  # Força português
        fp16=False      # Necessário para CPU
    )
    
    # Limpa o arquivo temporário
    os.unlink(tmp_path)
    
    return {"texto": result["text"]}

@app.get("/")
async def root():
    return {"mensagem": "Whisper API no ar! Use POST /transcrever para enviar áudio."}