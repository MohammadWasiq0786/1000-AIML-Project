from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
import sys
 
app = FastAPI()
mode = sys.argv[1] if len(sys.argv) > 1 else "summarize"
 
class Task(BaseModel):
    task_id: str
    text: str
 
@app.post("/summarize")
def summarize(task: Task):
    summary = task.text[:100] + "..." if len(task.text) > 100 else task.text
    return {
        "provider": "SummarizationAgent",
        "task_id": task.task_id,
        "status": "completed",
        "result": summary
    }
 
@app.post("/generate")
def generate_text(task: Task):
    generated_text = task.text + " [This is AI-generated continuation.]"
    return {
        "provider": "TextGeneratorAgent",
        "task_id": task.task_id,
        "status": "completed",
        "result": generated_text
    }
 
@app.post("/analyze")
def sentiment_analysis(task: Task):
    analysis = TextBlob(task.text).sentiment
    sentiment = "Positive" if analysis.polarity > 0 else "Negative" if analysis.polarity < 0 else "Neutral"
    return {
        "provider": "SentimentAnalysisAgent",
        "task_id": task.task_id,
        "status": "completed",
        "result": {
            "sentiment": sentiment,
            "polarity": analysis.polarity,
            "subjectivity": analysis.subjectivity
        }
    }
 
if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
    uvicorn.run(app, host="0.0.0.0", port=port)