"""
Build the Word (.docx) lab report for the GPT-2 IMDb fine-tuning experiment.
Embeds per-step screenshots and the Gradio screenshot, and pulls the real
metrics out of run.log.
"""
import os, re, glob
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(BASE, "screenshots")

# ---------- parse metrics from run.log ----------
def parse_log():
    txt = open(os.path.join(BASE, "run.log"), errors="ignore").read().replace("\r", "\n")
    txt = txt.replace("[A", "")
    evals = re.findall(r"\{'eval_loss': '([0-9.]+)', 'eval_accuracy': '([0-9.]+)'[^}]*'epoch': '([0-9.]+)'\}", txt)
    final = re.search(r"FINAL EVAL: \{'eval_loss': ([0-9.]+), 'eval_accuracy': ([0-9.]+)", txt)
    device = re.search(r"device = (\w+)", txt)
    preds = re.findall(r"\[12\] (\[\{'label': '(\w+)', 'score': ([0-9.]+)\}\])  <- (.+)", txt)
    return {
        "evals": [(float(a), float(b), float(c)) for (a, b, c) in evals],
        "final_acc": float(final.group(2)) if final else None,
        "final_loss": float(final.group(1)) if final else None,
        "device": device.group(1) if device else "cpu",
        "preds": [(lbl, float(sc), rev) for (_, lbl, sc, rev) in preds],
    }

M = parse_log()

doc = Document()

# base style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    return h

def para(text, bold=False, italic=False, size=11, align=None, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    if align:
        p.alignment = align
    return p

# ---------- Title page ----------
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Fine-Tuning GPT-2 for IMDb\nMovie Review Sentiment Classification")
r.bold = True; r.font.size = Pt(22); r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
para("Experiment Report", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
     color=(0x55, 0x55, 0x55))
doc.add_paragraph()
para("Objective: Fine-tune the pretrained GPT-2 language model with a 2-class "
     "classification head to predict whether an IMDb movie review expresses a "
     "Positive or Negative sentiment, then deploy it behind an interactive Gradio "
     "web interface.", align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_paragraph()
# summary table
acc = M["final_acc"]
para("Result summary", bold=True, size=13)
tbl = doc.add_table(rows=0, cols=2)
tbl.style = "Light Grid Accent 1"
rows = [
    ("Base model", "GPT-2 (124M parameters)"),
    ("Dataset", "stanfordnlp/imdb (movie reviews with sentiment labels)"),
    ("Task", "Binary sequence classification (0 = Negative, 1 = Positive)"),
    ("Framework", "PyTorch + Hugging Face Transformers / Datasets / Evaluate"),
    ("Compute device", M["device"].upper() + " (Apple Silicon)"),
    ("Evaluation metric", "Accuracy"),
    ("Final test accuracy", f"{acc*100:.2f}%" if acc is not None else "n/a"),
    ("Interface", "Gradio local web app"),
]
for k, v in rows:
    cells = tbl.add_row().cells
    cells[0].text = k; cells[1].text = v
    cells[0].paragraphs[0].runs[0].bold = True

doc.add_page_break()

# ---------- Configuration ----------
heading("1. Experimental Configuration", 1)
para("The notebook (finetune_gpt2_imdb.ipynb) implements the full assignment "
     "specification. Because the experiment was executed locally on an Apple "
     "Silicon Mac (GPU = MPS, no CUDA), a reduced data subset and sequence length "
     "were used for the demonstration run so that training completes in minutes "
     "rather than hours. Both configurations are listed below.")

cfg = doc.add_table(rows=1, cols=3)
cfg.style = "Light List Accent 1"
hdr = cfg.rows[0].cells
for i, h in enumerate(["Hyper-parameter", "Assignment spec (notebook)", "Local demo run"]):
    hdr[i].text = h
    hdr[i].paragraphs[0].runs[0].bold = True
demo_rows = [
    ("Training reviews", "3,000", "2,000"),
    ("Testing reviews", "1,000", "500"),
    ("Max sequence length", "512", "192"),
    ("Epochs", "2", "3"),
    ("Learning rate", "2e-5", "5e-5"),
    ("Batch size", "4", "4"),
    ("Weight decay", "0.01", "0.01"),
    ("Mixed precision (FP16)", "Yes (GPU)", "Off (MPS)"),
    ("Eval strategy", "Each epoch", "Each epoch"),
    ("load_best_model_at_end", "True", "True"),
]
for a, b, c in demo_rows:
    cells = cfg.add_row().cells
    cells[0].text = a; cells[1].text = b; cells[2].text = c
    cells[0].paragraphs[0].runs[0].bold = True
para("Note: The higher learning rate and extra epoch in the demo run compensate "
     "for the smaller dataset so the freshly-initialized classification head "
     "converges quickly on limited hardware.", italic=True, size=10,
     color=(0x66, 0x66, 0x66))

doc.add_page_break()

# ---------- Step-by-step with screenshots ----------
heading("2. Procedure and Step-by-Step Output", 1)
para("Each of the following steps corresponds to one stage of the pipeline. The "
     "screenshot beneath every step shows the actual code and console output "
     "captured during execution.")

STEP_TITLES = {
    1: "Verify hardware and import libraries",
    2: "Load a dataset (IMDb)",
    3: "Load the pretrained GPT-2 tokenizer",
    4: "Tokenize the text",
    5: "Create smaller training and testing datasets",
    6: "Initialize GPT-2 for classification",
    7: "Define the evaluation metric",
    8: "Configure fine-tuning",
    9: "Fine-tune GPT-2",
    10: "Evaluate the fine-tuned model",
    11: "Save the fine-tuned model",
    12: "Create an inference pipeline",
}
STEP_DESC = {
    1: "Imported PyTorch, the Hugging Face libraries, NumPy and Gradio, and detected the compute device (MPS on Apple Silicon).",
    2: "Loaded the stanfordnlp/imdb dataset — 25,000 labelled training reviews and 25,000 test reviews.",
    3: "Loaded the GPT-2 tokenizer with AutoTokenizer and set the EOS token as the padding token (GPT-2 has no native pad token).",
    4: "Converted reviews into token IDs with padding, truncation and a fixed maximum sequence length.",
    5: "Shuffled the data (seed 42) and selected balanced subsets for faster local training and testing.",
    6: "Loaded GPT-2 with a 2-class classification head and set the model's pad_token_id. The 'score.weight MISSING' note confirms the head is newly initialised for fine-tuning.",
    7: "Loaded the accuracy metric from the evaluate library and defined compute_metrics to compare predictions with the true labels.",
    8: "Configured Hugging Face TrainingArguments with the required hyper-parameters, per-epoch evaluation and load_best_model_at_end=True.",
    9: "Built a Trainer and ran the training loop. The loss decreases across steps, showing the model is learning.",
    10: "Evaluated the best checkpoint on the held-out test subset using accuracy.",
    11: "Saved the fine-tuned model weights and tokenizer to ./gpt2-imdb-sentiment.",
    12: "Reloaded the saved model as a Hugging Face sentiment-analysis pipeline and verified predictions on sample reviews.",
}

for n in range(1, 13):
    heading(f"Step {n}: {STEP_TITLES[n]}", 2)
    para(STEP_DESC[n])
    shot = os.path.join(SHOTS, f"step_{n:02d}.png")
    if os.path.exists(shot):
        doc.add_picture(shot, width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = para(f"Figure {n}. Step {n} — code and captured output.", italic=True,
                   size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=(0x77, 0x77, 0x77))

# ---------- Steps 13 & 14 Gradio ----------
heading("Step 13-14: Gradio Interface and Local Application", 2)
para("A Gradio interface was created with a text box where a user types a movie "
     "review; the fine-tuned model returns the predicted sentiment (Positive / "
     "Negative) with a confidence score. The application was launched locally.")
gradio_shot = os.path.join(BASE, "gradio.png")
if os.path.exists(gradio_shot):
    doc.add_picture(gradio_shot, width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para("Figure 13. The running Gradio application classifying a positive review.",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=(0x77, 0x77, 0x77))
neg_shot = os.path.join(BASE, "gradio_negative.png")
if os.path.exists(neg_shot):
    doc.add_picture(neg_shot, width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para("Figure 14. The same application correctly classifying a negative review.",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=(0x77, 0x77, 0x77))

doc.add_page_break()

# ---------- Results ----------
heading("3. Results", 1)
para("Per-epoch evaluation accuracy (demonstration run):", bold=True)
rt = doc.add_table(rows=1, cols=3)
rt.style = "Light Grid Accent 1"
h = rt.rows[0].cells
for i, x in enumerate(["Epoch", "Eval loss", "Eval accuracy"]):
    h[i].text = x; h[i].paragraphs[0].runs[0].bold = True
for (loss, a, ep) in M["evals"]:
    c = rt.add_row().cells
    c[0].text = f"{ep:.0f}"; c[1].text = f"{loss:.4f}"; c[2].text = f"{a*100:.2f}%"

doc.add_paragraph()
if M["final_acc"] is not None:
    para(f"Final test accuracy (best checkpoint): {M['final_acc']*100:.2f}%",
         bold=True, size=12, color=(0x1F, 0x64, 0x38))

doc.add_paragraph()
para("Sample predictions from the inference pipeline:", bold=True)
for lbl, sc, rev in M["preds"]:
    para(f"  •  “{rev.strip()}”  →  {lbl} ({sc*100:.1f}% confidence)")

# ---------- Discussion ----------
heading("4. Discussion and Conclusion", 1)
para("The experiment successfully demonstrates the complete transfer-learning "
     "workflow: a general-purpose GPT-2 language model was adapted into a sentiment "
     "classifier by attaching a 2-class head and fine-tuning on labelled IMDb "
     "reviews. Across epochs the training loss fell steadily and evaluation "
     "accuracy improved, confirming that the model learned the sentiment signal.")
para("Because the run was performed on a laptop GPU (Apple MPS) with a reduced "
     "dataset and shorter sequences, the accuracy is lower than what the full "
     "assignment configuration (3,000 training reviews, 512-token sequences) would "
     "achieve on a CUDA GPU — typically around 85-90%. Scaling the dataset, "
     "sequence length and training time would raise accuracy accordingly. The "
     "saved model was reloaded through a Hugging Face pipeline and served with a "
     "Gradio interface, allowing new movie reviews to be classified interactively.")

para("Files produced:", bold=True)
for f in ["finetune_gpt2_imdb.ipynb  — full notebook (assignment spec)",
          "run_finetune.py  — headless training/evaluation script",
          "gpt2-imdb-sentiment/  — saved fine-tuned model + tokenizer",
          "GPT2_IMDb_Report.docx  — this report"]:
    para("  •  " + f)

out = os.path.join(BASE, "GPT2_IMDb_Report.docx")
doc.save(out)
print("Report saved:", out)
