import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax

# Cargar modelo preentrenado (binaria: correcta vs incorrecta)
MODEL_NAME = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Clasificador: tripleta válida o no en el contexto del texto ===
def is_triplet_valid(text: str, subj: str, pred: str, obj: str, threshold=0.5):
    input_text = f"Texto: {text}\nTripleta: ({subj}; {pred}; {obj})"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1)
        confidence = probs[0][1].item()  # probabilidad de ser "correcta"
        return int(confidence > threshold), confidence

# === Ejemplo de uso ===
if __name__ == "__main__":
    text = "Barack Obama fue presidente de Estados Unidos entre 2009 y 2017."
    triplet = ("Barack Obama", "position_held", "President of the United States")

    label, score = is_triplet_valid(text, *triplet)
    print(f"✅ Tripleta válida: {bool(label)} (confianza: {score:.2f})")
