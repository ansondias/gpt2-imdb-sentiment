"""Step 13-14: Gradio interface for the fine-tuned GPT-2 IMDb sentiment model."""
import torch, gradio as gr
from transformers import pipeline

MODEL_DIR = "./gpt2-imdb-sentiment"
clf = pipeline("sentiment-analysis", model=MODEL_DIR, tokenizer=MODEL_DIR,
               device=-1, truncation=True, max_length=192)

def predict_sentiment(review):
    if not review or not review.strip():
        return "Please enter a movie review."
    r = clf(review)[0]
    label = "Positive" if r["label"].upper().endswith("1") or r["label"].upper() == "POSITIVE" else "Negative"
    return f"Prediction: {label}   (confidence: {r['score']:.2%})"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Movie Review", placeholder="Type a movie review here..."),
    outputs=gr.Textbox(label="Sentiment"),
    title="GPT-2 IMDb Sentiment Classifier",
    description="Fine-tuned GPT-2 predicts whether a movie review is Positive or Negative.",
    examples=[["This film blew me away - brilliant acting and a gripping story."],
              ["I fell asleep halfway through. Dull and forgettable."]],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
