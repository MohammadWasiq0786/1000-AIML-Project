# Project 3: RAG over Audio

## Overview

You'll build a system where:

Audio is transcribed to text (using Whisper)

Text is chunked and embedded into a vector store (ChromaDB)

User queries are semantically matched and passed to a local LLM for reasoning (Ollama)

## Tech Stack

### ComponentTool

LibraryAudio Transcriptionwhisper by OpenAIVector DBChromaDBEmbeddingssentence-transformers or OllamaLLMmistral or llama3 via OllamaInterfaceStreamlit / CLI / FastAPIFile Types.mp3, .wav, .m4a

### Project Structure

```text
rag-over-audio/
├── main.py
├── transcribe.py
├── ingest.py
├── query.py
├── audio/
│   └── sample.wav
├── db/
├── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies

```bash
pip install openai-whisper chromadb sentence-transformers pydub ollama
brew install ffmpeg  # required by Whisper
ollama pull mistral
```

* **Step 2:** Transcribe Audio
* **Step 3:** Ingest into Vector DB
* **Step 4:** Query the Audio Content
* **Step 5:** Run the Pipeline

### Output
* You now have an audio-aware RAG system!
* You can ask questions like:

“What was the guest’s opinion on AI safety?”

“Summarize the key argument in the second half.”

### Optional Enhancements

* Add Streamlit upload and chat UI
* Include speaker diarization for multi-speaker audio
* Support longer audio splitting
* Save transcripts with timestamps
* Add a summary generator agent