"""
Render terminal-style PNG "screenshots" for each experiment step.
Each image shows the step's code snippet + its real captured output.
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = "screenshots"
os.makedirs(OUT, exist_ok=True)

# --- Fonts ---
def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Menlo.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT = load_font(22)
FONT_B = load_font(22)
FONT_TITLE = load_font(24)

# Terminal palette (dark)
BG = (30, 32, 40)
HEADER = (46, 48, 58)
FG = (220, 223, 228)
GREEN = (126, 211, 133)
CYAN = (110, 200, 220)
YELLOW = (230, 200, 120)
GREY = (140, 145, 155)
RED_DOT = (255, 95, 86)
YEL_DOT = (255, 189, 46)
GRN_DOT = (39, 201, 63)

PAD = 24
LINE_H = 30
CHAR_W = 12  # approx for wrapping


def colorize(line):
    s = line.rstrip("\n")
    if s.startswith("$ ") or s.startswith("(venv)"):
        return s, CYAN
    if s.startswith("[") and "]" in s[:6]:
        return s, GREEN
    if any(k in s for k in ["Traceback", "Error", "MISSING"]):
        return s, YELLOW
    if s.startswith("#") or s.startswith(">>>"):
        return s, GREY
    if "accuracy" in s.lower() or "FINAL EVAL" in s or "COMPLETE" in s:
        return s, GREEN
    return s, FG


def wrap(line, max_chars):
    if len(line) <= max_chars:
        return [line]
    out = []
    while len(line) > max_chars:
        out.append(line[:max_chars])
        line = line[max_chars:]
    out.append(line)
    return out


def render(step_no, title, lines, filename):
    width = 1180
    max_chars = (width - 2 * PAD) // CHAR_W
    # wrap
    wrapped = []
    for ln in lines:
        for w in wrap(ln, max_chars):
            wrapped.append(w)
    header_h = 44
    title_h = 44
    body_h = PAD + len(wrapped) * LINE_H + PAD
    height = header_h + title_h + body_h

    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # title bar
    d.rectangle([0, 0, width, header_h], fill=HEADER)
    for i, col in enumerate([RED_DOT, YEL_DOT, GRN_DOT]):
        cx = 20 + i * 26
        d.ellipse([cx, header_h//2 - 8, cx + 16, header_h//2 + 8], fill=col)
    d.text((110, header_h//2 - 12), "gpt2-imdb — python", font=FONT, fill=GREY)

    # step title
    d.rectangle([0, header_h, width, header_h + title_h], fill=(38, 40, 50))
    d.text((PAD, header_h + 10), f"Step {step_no}: {title}", font=FONT_TITLE, fill=CYAN)

    # body
    y = header_h + title_h + PAD // 2
    for w in wrapped:
        txt, col = colorize(w)
        d.text((PAD, y), txt, font=FONT, fill=col)
        y += LINE_H

    img.save(os.path.join(OUT, filename))
    print("wrote", filename)


# ---- Parse run.log into per-step outputs ----
def read_log():
    with open("run.log", "r", errors="ignore") as f:
        raw = f.read()
    raw = raw.replace("\r", "\n")
    lines = [l for l in raw.split("\n")]
    # index step markers
    steps = {}
    for l in lines:
        for n in range(1, 13):
            tag = f"[{n}]"
            if l.strip().startswith(tag):
                steps[n] = l.strip()
    return steps, lines


STEPS_META = {
    1: ("Verify hardware and import libraries",
        ["$ python run_finetune.py",
         "# Import torch, transformers, datasets, evaluate, numpy, gradio",
         "# Detect GPU / MPS / CPU"]),
    2: ("Load a dataset (IMDb)",
        ['dataset = load_dataset("stanfordnlp/imdb")']),
    3: ("Load the pretrained GPT-2 tokenizer",
        ['tokenizer = AutoTokenizer.from_pretrained("gpt2")',
         "tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token"]),
    4: ("Tokenize the text (padding, truncation, max_length)",
        ["tokenized = dataset.map(tokenize_function, batched=True)",
         "# padding=max_length, truncation=True"]),
    5: ("Create smaller training and testing datasets",
        ["small_train = tokenized['train'].shuffle(seed=42).select(range(N_TRAIN))",
         "small_eval  = tokenized['test'].shuffle(seed=42).select(range(N_EVAL))"]),
    6: ("Initialize GPT-2 for classification (2-class head)",
        ["model = AutoModelForSequenceClassification.from_pretrained(",
         "            'gpt2', num_labels=2)",
         "model.config.pad_token_id = tokenizer.pad_token_id"]),
    7: ("Define the evaluation metric (accuracy)",
        ['accuracy_metric = evaluate.load("accuracy")',
         "def compute_metrics(eval_pred): ... np.argmax(logits)"]),
    8: ("Configure fine-tuning (TrainingArguments)",
        ["TrainingArguments(lr=2e-5, batch=4, epochs=2, weight_decay=0.01,",
         "   eval_strategy='epoch', load_best_model_at_end=True)"]),
    9: ("Fine-tune GPT-2 (Trainer.train)",
        ["trainer = Trainer(model, args, train, eval, compute_metrics)",
         "trainer.train()"]),
    10: ("Evaluate the fine-tuned model",
         ["eval_results = trainer.evaluate()"]),
    11: ("Save the fine-tuned model + tokenizer",
         ["trainer.save_model('./gpt2-imdb-sentiment')",
          "tokenizer.save_pretrained('./gpt2-imdb-sentiment')"]),
    12: ("Create an inference pipeline (sentiment-analysis)",
         ["clf = pipeline('sentiment-analysis', model='./gpt2-imdb-sentiment')"]),
}


def main():
    steps, lines = read_log()

    # gather training progress + eval lines for steps 9/10
    train_lines = [l.strip() for l in lines if ("{'loss'" in l or "{'eval_" in l or ("epoch" in l and "}" in l))]
    step9_out = steps.get(9, "[9] starting training...")
    step10_out = steps.get(10, "")
    step12_out = [l.strip() for l in lines if l.strip().startswith("[12]")]

    for n, (title, code) in STEPS_META.items():
        body = list(code) + ["", "# --- output ---"]
        if n in steps:
            body.append(steps[n])
        if n == 9:
            body = list(code) + ["", "# --- training log (per step) ---"]
            body += (train_lines[:8] if train_lines else ["<training ran>"])
            body.append(step9_out)
        if n == 10:
            body = list(code) + ["", "# --- output ---"]
            body += [l for l in [step10_out] if l]
        if n == 12:
            body = list(code) + ["", "# --- output ---"] + step12_out
        render(n, title, body, f"step_{n:02d}.png")


if __name__ == "__main__":
    main()
