import torch
import numpy as np
from transformers import pipeline

#input text to be added
text = "Just because I cheated and got somebody else pregnant doesn't mean that I don't want to be with you, I still do. Everything's just messed up rn"

labels = [
    "emotionally manipulative behavior",
    "gaslighting or reality distortion",
    "verbal abuse or insults",
    "love bombing or excessive reassurance",
    "blame shifting responsibility",
    "controlling or possessive behavior"
]

templates = [
    "This message shows signs of {}.",
    "The speaker is engaging in {}.",
    "This message demonstrates {}."
]

THRESHOLD = 0.6 #subject to change

#checks for GPU availa
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=device
)

all_scores = []

for template in templates:
    result = classifier(
        text,
        labels,
        hypothesis_template=template,
        multi_label=True
    )
    all_scores.append(result["scores"])

avg_scores = np.mean(np.array(all_scores), axis=0)
final_results = list(zip(labels, avg_scores))
final_results.sort(key=lambda x: x[1], reverse=True)

print("\n=== ⚠️Behavior Detection Results⚠️ ===\n")

for label, score in final_results:
    flag = "🚩" if score >= THRESHOLD else ""
    print(f"{label:40} {score:.3f} {flag}")

