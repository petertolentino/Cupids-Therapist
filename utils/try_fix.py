import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline
import time
import torch
import re

# ----------------------------
# CONFIG
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(
    page_title="Cupid's Therapist",
    page_icon="💘",
    layout="wide"
)

st.markdown(
    """
<style>
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    .red-flag {
        color: #ff4b4b; font-size: 40px; text-align: center; padding: 20px;
        border-radius: 10px; border: 2px solid #ff4b4b;
        background-color: rgba(255, 75, 75, 0.1);
    }
    .safe-flag {
        color: #00cc66; font-size: 40px; text-align: center; padding: 20px;
        border-radius: 10px; border: 2px solid #00cc66;
        background-color: rgba(0, 204, 102, 0.1);
    }
    .metric-card {
        background-color: #f0f2f6; border-radius: 10px; padding: 15px;
        text-align: center; margin: 5px;
    }
</style>
""",
    unsafe_allow_html=True
)

st.title("💘 **Cupid's Therapist**")
st.markdown("### *AI-Powered Dating App Red Flag Detector*")

# ----------------------------
# STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ----------------------------
# MODEL LOADER
# ----------------------------
@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading AI model... (first time can take 10–20s)"):
        clf = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        return clf

try:
    classifier = load_model()
    st.session_state.model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Using fallback detection mode (no toxicity model).")
    classifier = None

# ----------------------------
# RELATIONSHIP RED FLAG HEURISTIC (non-toxicity)
# ----------------------------
def relationship_red_flag_score(text: str):
    """
    Returns: (score in [0,1], list_of_categories_hit)
    Flags relationship issues like cheating, manipulation, control, threats, etc.
    """
    t = (text or "").lower()
    hits = []

    patterns = {
        "infidelity": [
            r"\bcheat(ed|ing)?\b", r"\baffair\b", r"\bother (girl|guy|someone)\b",
            r"\bpregnan(t|cy)\b", r"\bgot (her|him) pregnant\b", r"\b(he|she) is pregnant\b"
        ],
        "minimizing/gaslighting": [
            r"\bdoesn[’']?t mean\b", r"\byou('?re| are) overreacting\b",
            r"\bit('?s| is) not (a big deal|my fault)\b", r"\byou('?re| are) crazy\b"
        ],
        "guilt_trip": [
            r"\bif you (loved|cared)\b", r"\byou made me\b", r"\bafter everything i did\b",
            r"\blook what you made me do\b"
        ],
        "control": [
            r"\bsend (me )?your location\b", r"\bwho are you with\b",
            r"\bdon[’']?t talk to\b", r"\byou can't\b", r"\byou must\b"
        ],
        "coercion/threat": [
            r"\byou('?ll| will) regret\b", r"\bi know where you live\b",
            r"\bi('?ll| will) ruin\b", r"\bi('?m| am) watching\b"
        ],
        "love_bombing/dependency": [
            r"\bcan[’']?t live without you\b", r"\byou('?re| are) my everything\b",
            r"\bonly you understand me\b", r"\bi need you (all the time|always)\b"
        ],
    }

    for cat, pats in patterns.items():
        for p in pats:
            if re.search(p, t):
                hits.append(cat)
                break

    unique_hits = sorted(set(hits))
    # Simple scoring: more categories hit => higher risk
    score = (0.05 if not unique_hits else min(0.20 + 0.22 * len(unique_hits), 0.95))
    return score, unique_hits

def toxicity_probability(text: str) -> float:
    """
    For unitary/toxic-bert: returns P(toxic).
    If classifier not available, returns a small default.
    """
    if not classifier or not st.session_state.model_loaded:
        return 0.05

    out = classifier(text)[0]  # usually {'label': 'toxic'/'non-toxic', 'score': ...}
    label = str(out.get("label", "")).lower()
    score = float(out.get("score", 0.0))

    if "toxic" in label and "non" not in label:
        return score
    if "non" in label and "toxic" in label:
        return 1.0 - score

    # Fallback (if label naming differs)
    return score if "toxic" in label else (1.0 - score)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("🎯 How it works")
    st.markdown(
        """
1. **Paste text** or **upload screenshot**
2. OCR reads text from the screenshot
3. AI checks **toxicity** + **relationship red flags**
4. You get a safety assessment
"""
    )

    st.divider()

    threshold = st.slider(
        "Overall Sensitivity (🚩 threshold)",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        help="Lower = more sensitive, Higher = stricter"
    )

    st.caption("Tip: Keep ~0.25–0.35 for hackathon demos.")

    st.divider()

    st.header("📊 Stats")
    user_msgs = len([m for m in st.session_state.messages if m.get("role") == "user"])
    red_flags = len([m for m in st.session_state.messages if m.get("is_red_flag")])
    st.metric("Messages Analyzed", user_msgs)
    st.metric("Red Flags Found", red_flags)

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input")

    text_input = st.text_area(
        "Paste your conversation:",
        height=150,
        placeholder="Type or paste the message here..."
    )

    st.markdown("<p style='text-align: center'>— OR —</p>", unsafe_allow_html=True)

    img_file = st.file_uploader(
        "Upload screenshot:",
        type=["png", "jpg", "jpeg"],
        help="Upload a screenshot of the conversation"
    )

    analyze_clicked = st.button("🔍 Analyze Now", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    results_container = st.container()

    with st.expander("💬 Message History", expanded=False):
        if not st.session_state.messages:
            st.info("No messages analyzed yet")
        else:
            for msg in st.session_state.messages[-6:]:
                if msg.get("role") == "user":
                    st.info(f"📝 **You:** {msg.get('content','')}")
                else:
                    if msg.get("is_red_flag"):
                        st.error(f"🚩 **AI:** {msg.get('content','')}")
                    else:
                        st.success(f"✅ **AI:** {msg.get('content','')}")

# ----------------------------
# ANALYZE
# ----------------------------
if analyze_clicked and (text_input or img_file):
    with st.spinner("🔍 Analyzing..."):
        text_to_analyze = (text_input or "").strip()

        if img_file:
            try:
                img = Image.open(img_file)
                with st.expander("📸 Uploaded Screenshot", expanded=False):
                    st.image(img, caption="Uploaded Screenshot", width=320)

                with st.spinner("📸 Reading text from image (OCR)..."):
                    extracted_text = pytesseract.image_to_string(img, lang="spa+eng")
                    if extracted_text.strip():
                        text_to_analyze = extracted_text.strip()
                        st.success(f"✅ OCR extracted: {extracted_text[:150]}{'...' if len(extracted_text)>150 else ''}")
                    else:
                        st.warning("⚠️ No text found in image. Using pasted text (if any).")
            except Exception as e:
                st.error(f"OCR Error: {e}")
                # keep text_to_analyze from text_input

        if not text_to_analyze:
            st.warning("⚠️ Please enter some text or upload an image with text.")
        else:
            # Save user message (shortened)
            st.session_state.messages.append({
                "role": "user",
                "content": text_to_analyze[:200] + ("..." if len(text_to_analyze) > 200 else "")
            })

            # 1) Toxicity
            tox_p = toxicity_probability(text_to_analyze)

            # 2) Relationship red flags (non-toxicity)
            rel_p, rel_hits = relationship_red_flag_score(text_to_analyze)

            # Overall
            overall = max(tox_p, rel_p)
            is_red_flag = overall > threshold

            # Build scores for your dashboard
            # (tox model only gives toxic/non-toxic, so we use it to approximate sub-bars)
            scores = {
                "toxic": tox_p,
                "insult": tox_p * 0.70,
                "threat": tox_p * 0.60,
                "identity_hate": tox_p * 0.30,
                "severe_toxic": tox_p * 0.50,
                "relationship_red_flag": rel_p,
            }

            avg_score = sum(scores.values()) / len(scores)

            with results_container:
                if is_red_flag:
                    st.markdown("<div class='red-flag'>🚩 RED FLAG DETECTED!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-flag'>✅ LOOKS SAFE</div>", unsafe_allow_html=True)

                st.divider()
                st.subheader("📈 Detailed Analysis")

                # Row 1
                a, b, c = st.columns(3)

                def metric_card(col, title, val):
                    with col:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"**{title}**")
                        st.progress(float(val))
                        color = "red" if val > threshold else "orange" if val > threshold/2 else "green"
                        st.markdown(
                            f"<p style='color:{color}; font-size:24px; font-weight:bold; text-align:center;'>{val:.0%}</p>",
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                metric_card(a, "Toxicity", scores["toxic"])
                metric_card(b, "Insults (proxy)", scores["insult"])
                metric_card(c, "Threats (proxy)", scores["threat"])

                # Row 2
                d, e, f = st.columns(3)
                metric_card(d, "Hate Speech (proxy)", scores["identity_hate"])
                metric_card(e, "Severity (proxy)", scores["severe_toxic"])
                metric_card(f, "Relationship Risk", scores["relationship_red_flag"])

                st.divider()
                st.subheader("📝 Safety Summary")

                if is_red_flag:
                    # Decide which dimension triggered it more
                    if rel_p >= tox_p and rel_p > threshold:
                        why = f"Relationship red flags detected: **{', '.join(rel_hits) if rel_hits else 'relationship concerns'}**."
                    elif tox_p > threshold:
                        why = "Toxic/abusive language detected."
                    else:
                        why = "Overall risk exceeded threshold."

                    st.error(
                        f"""
🚩 **RED FLAG ALERT**

{why}

**Safety Advice:**
- Don’t escalate the conversation
- Set boundaries / disengage
- Block + report if needed
- Save screenshots if threats/coercion appears
"""
                    )
                else:
                    st.success(
                        """
✅ **SAFE MESSAGE (based on toxicity + relationship signals)**

**Safety Reminder:**
- Meet in public places first
- Tell a friend where you're going
- Trust your gut if something feels off
"""
                    )

                if rel_hits:
                    st.warning(f"💔 Relationship signals hit: {', '.join(rel_hits)}")

                with st.expander("📄 View analyzed text"):
                    st.write(text_to_analyze)
                    st.caption(f"📊 {len(text_to_analyze.split())} words, {len(text_to_analyze)} characters")

            # Save assistant summary
            st.session_state.messages.append({
                "role": "assistant",
                "content": "RED FLAG DETECTED!" if is_red_flag else "Looks safe",
                "is_red_flag": is_red_flag,
                "scores": scores
            })

# ----------------------------
# EXAMPLES
# ----------------------------
with st.expander("🎮 Try with examples", expanded=False):
    st.markdown("Click any example to test the detector:")
    x1, x2, x3 = st.columns(3)

    with x1:
        if st.button("😊 Safe Example", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Hey! Would you like to grab coffee sometime? I'd love to get to know you better."
            })
            st.rerun()

    with x2:
        if st.button("⚠️ Suspicious Example", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "You're the only one who understands me. I can't live without you. Why haven't you replied?"
            })
            st.rerun()

    with x3:
        if st.button("🚩 Red Flag Example", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Just because I cheated and got someone else pregnant doesn’t mean I don’t wanna be with you."
            })
            st.rerun()

# ----------------------------
# FOOTER + DEBUG
# ----------------------------
st.divider()
mid = st.columns(3)[1]
with mid:
    st.markdown(
        "<p style='text-align:center; color:gray;'>Made with 💘 for dating safety • Use responsibly</p>",
        unsafe_allow_html=True
    )

with st.expander("🔧 Debug Info", expanded=False):
    st.write("Model loaded:", st.session_state.model_loaded)
    st.write("Session messages:", len(st.session_state.messages))
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()
