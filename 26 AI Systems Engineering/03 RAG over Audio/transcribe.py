import whisper
 
def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]