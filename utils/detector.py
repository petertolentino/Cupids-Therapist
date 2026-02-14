import torch
import numpy as np
from transformers import pipeline
import re

# Input text
#text = "I love you"
text = "Just because I cheated and got somebody else pregnant doesn't mean that I don't want to be with you, I still do. Everything's just messed up rn"

# Refined labels with definitions
labels = [
    "emotionally manipulative behavior: trying to make someone feel guilty, anxious, or insecure to gain control",
    "gaslighting or reality distortion: causing someone to doubt their memory, perception, or feelings",
    "verbal abuse or insults: using words to belittle, demean, or shame another person",
    "love bombing or excessive reassurance: repeatedly promising love or making excessive positive statements to influence or manipulate someone",
    "blame shifting responsibility: placing the blame on others for one's own mistakes or actions",
    "controlling or possessive behavior: restricting freedom, choices, or social interactions of another person"
]

# Multiple hypothesis templates
templates = [
    "This message shows signs of {}.",
    "The speaker is engaging in {}.",
    "This message demonstrates {} behavior.",
    "The text contains {}."
]

# Thresholds for flagging subtle behaviors
THRESHOLD_DEFAULT = 0.25
THRESHOLDS = {
    "love bombing or excessive reassurance": 0.08,
    "blame shifting responsibility": 0.05
}

# GPU check
device = 0 if torch.cuda.is_available() else -1

# Zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=device
)

# Split text into clauses to catch subtle cues
clauses = re.split(r'[.,;]\s*', text)
clauses = [c.strip() for c in clauses if c.strip()]

# Function to get threshold for label
def get_threshold(label_name):
    return THRESHOLDS.get(label_name, THRESHOLD_DEFAULT)

# Score each label independently
final_results = []

for label in labels:
    label_name = label.split(":", 1)[0].strip()
    label_scores = []

    for template in templates:
        for clause in clauses:
            result = classifier([clause], [label], hypothesis_template=template, multi_label=True)
            if isinstance(result, list):
                result = result[0]
            # Only one label, so take the first score
            label_scores.append(result["scores"][0])

    # Average score across all templates and clauses
    avg_score = np.mean(label_scores)

    # Optional: keyword boosting for subtle behaviors
    if label_name == "love bombing or excessive reassurance":
        if re.search(r"\bi love you\b|\bi still do\b", text, re.I):
            avg_score = max(avg_score, 0.2)  # boost score
    if label_name == "blame shifting responsibility":
        if re.search(r"\byou made me\b|\bit's your fault\b", text, re.I):
            avg_score = max(avg_score, 0.2)

    final_results.append((label, avg_score))

# Sort results descending
final_results.sort(key=lambda x: x[1], reverse=True)

# Print results
print("\n=== ⚠️ Behavior Detection Results ⚠️ ===\n")
for label, score in final_results:
    label_name = label.split(":", 1)[0].strip()
    flag = "🚩" if score >= get_threshold(label_name) else ""
    print(f"{label_name:40} {score:.3f} {flag}")

