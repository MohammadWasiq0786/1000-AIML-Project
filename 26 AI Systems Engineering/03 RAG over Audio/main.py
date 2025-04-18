from transcribe import transcribe_audio
from ingest import ingest_text_to_chroma
from query import query_audio_knowledge
 
audio_path = "audio/sample.wav"
 
# Step 1: Transcribe
print("Transcribing...")
transcript = transcribe_audio(audio_path)
 
# Step 2: Ingest into Chroma
print("Indexing transcript...")
ingest_text_to_chroma(transcript)
 
# Step 3: Ask a question
print("Ask a question about the audio:")
user_query = input(">> ")
answer = query_audio_knowledge(user_query)
print("\nAnswer:", answer)