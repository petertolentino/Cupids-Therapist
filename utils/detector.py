import torch
import numpy as np
from transformers import pipeline
import re

# -----------------------------
# Input text to analyze
# -----------------------------
text = "If you break up with me i will kill myself and i hope you regret it for the rest of your life"

# -----------------------------
# Labels with definitions
# -----------------------------
labels = [
    "emotionally manipulative behavior: using guilt, fear, or insecurity to manipulate or influence someone",
    "gaslighting or reality distortion: causing someone to doubt their memory, perception, or feelings",
    "verbal abuse or insults: using words to belittle, demean, or shame another person",
    "love bombing or excessive reassurance: repeatedly giving affection, compliments, or reassurance to elicit trust or attachment",
    "blame shifting responsibility: deflecting responsibility by blaming others for one's own actions",
    "controlling or possessive behavior: monitoring, restricting, or isolating someone's activities, independance, decisions or social interactions"
]

# -----------------------------
# Hypothesis templates
# -----------------------------
templates = [
    "This message shows signs of {}.",
    "The speaker is engaging in {}.",
    "This message demonstrates {} behavior.",
    "The text contains {}."
]

# -----------------------------
# Thresholds for flagging
# -----------------------------
threshold = 0.4

# -----------------------------
# Keyword boosting rules
# -----------------------------
keyword_boosts = {
    "love bombing or excessive reassurance": [r"\bi love you\b", r"\bi still do\b", r"\bi miss you\b", r"\byou're amazing\b",  r"\byou're perfect\b", r"\bcan't live without you\b", r"\bdon't deserve you\b"],
    "blame shifting responsibility": [r"\byou made me\b", r"\bit's your fault\b", r"\byou caused\b", r"\byou always\b", r"\byou never\b"],
    "emotionally manipulative behavior": [r"\byou should feel\b", r"\byou owe me\b", r"\byou must\b", r"\bnothing without me\b", r"\bworthless\b"],
    "gaslighting or reality distortion": [r"\byou're imagining\b", r"\byou don't remember\b", r"\byou're dramatic\b", r"\bkill myself\b", r"\bnothing without me\b"],
    "verbal abuse or insults": [r"\bstupid\b", r"\bidiot\b", r"\bfool\b", r"\bbitch\b", r"\bhoe\b", r"\bfuck\b", r"\bworthless\b"],
    "controlling or possessive behavior": [r"\byou can't go\b", r"\bmust stay\b", r"\ball yours\b", r"\bnot allowed\b", r"\bonly mine\b", r"\bleave me\b"]
}

# -----------------------------
# Split text into clauses
# -----------------------------
clauses = re.split(r'[.,;]\s*', text)
clauses = [c.strip() for c in clauses if c.strip()]

# -----------------------------
# Device setup
# -----------------------------
device = 0 if torch.cuda.is_available() else -1

# -----------------------------
# Zero-shot classifier
# -----------------------------
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=device
)

# -----------------------------
# Scoring each label independently
# -----------------------------
final_results = []

for label in labels:
    label_name = label.split(":", 1)[0].strip()
    label_scores = []

    # Score every template + clause
    for template in templates:
        for clause in clauses:
            result = classifier([clause], [label], hypothesis_template=template, multi_label=True)
            if isinstance(result, list):
                result = result[0]
            label_scores.append(result["scores"][0])  # Only one label

    # Average score
    avg_score = np.mean(label_scores)

    # Keyword boosting
    if label_name in keyword_boosts:
        for kw_pattern in keyword_boosts[label_name]:
            if re.search(kw_pattern, text, re.I):
                avg_score = max(avg_score, 0.2)  # adjust boost as needed

    final_results.append((label, avg_score))

# -----------------------------
# Sort results descending
# -----------------------------
final_results.sort(key=lambda x: x[1], reverse=True)

# -----------------------------
# Print results with flags
# -----------------------------
print("\n=== ⚠️ Behavior Detection Results ⚠️ ===\n")
for label, score in final_results:
    label_name = label.split(":", 1)[0].strip()
    flag = "🚩" if score >= threshold else ""
    print(f"{label_name:40} {score:.3f} {flag}")

