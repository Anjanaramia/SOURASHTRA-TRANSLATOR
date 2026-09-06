import os
import shutil
import json
import re
import tempfile
from datetime import datetime
import pandas as pd
import streamlit as st
from gtts import gTTS

# --- Page Configuration ---
st.set_page_config(
    page_title="Sourashtra Smart Voice Phrasebook & Translator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.txt")
DATASET_DIR = os.path.join(BASE_DIR, "sourashtra_dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)

if not os.path.exists(METADATA_PATH):
    df_init = pd.DataFrame(columns=["id", "timestamp", "english_phrase", "sourashtra_text", "audio_file"])
    df_init.to_csv(METADATA_PATH, index=False)


# --- Helper Functions ---
@st.cache_data
def load_prompts():
    if os.path.exists(PROMPTS_PATH):
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines
    return [
        "Did you drink tea/coffee?",
        "Where are you going?",
        "What did you cook today?",
        "Please come and sit.",
        "Is the food good?",
        "What is the time now?"
    ]

PROMPTS = load_prompts()

def load_metadata_df():
    if os.path.exists(METADATA_PATH):
        try:
            return pd.read_csv(METADATA_PATH, dtype=str).fillna("")
        except Exception:
            pass
    return pd.DataFrame(columns=["id", "timestamp", "english_phrase", "sourashtra_text", "audio_file"])

def build_few_shot_context():
    df = load_metadata_df()
    examples = []
    for _, row in df.iterrows():
        stext = str(row.get("sourashtra_text", "")).strip()
        eng = str(row.get("english_phrase", "")).strip()
        if stext and eng:
            examples.append(f'- Sourashtra: "{stext}" <-> English: "{eng}"')
    if not examples:
        examples = [
            '- Sourashtra: "Chaa/Kapee piyo ki?" <-> English: "Did you drink tea/coffee?"',
            '- Sourashtra: "Kayi jaatas?" <-> English: "Where are you going?"',
            '- Sourashtra: "Aavjo baihiye." <-> English: "Please come and sit."'
        ]
    return "\n".join(examples)

def get_api_key(key_name):
    # Check Streamlit secrets first, then environment variables, then .env file
    if hasattr(st, "secrets") and key_name in st.secrets:
        return st.secrets[key_name]
    val = os.environ.get(key_name)
    if val:
        return val
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(f"{key_name}="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return None


def call_gemini_raw(gemini_key, full_prompt):
    import requests
    
    # 1. Google Cloud OAuth Bearer Token (starts with AQ... or ya29...)
    if gemini_key.startswith("AQ") or gemini_key.startswith("ya29"):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {
            "Authorization": f"Bearer {gemini_key}",
            "Content-Type": "application/json"
        }
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code == 200:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"Google Cloud OAuth Token Error ({res.status_code}): {res.text}")

    # 2. Standard Gemini API Key (starts with AIzaSy...)
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code == 200:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        raise e


# --- LLM Call Router ---
def call_llm_translation(sourashtra_input):
    context = build_few_shot_context()
    
    system_prompt = (
        "You are an expert linguist specializing in the Sourashtra language (spoken by Sourashtrians in Tamil Nadu, India) and its translation into English and Tamil.\n"
        "Use the following in-context few-shot dataset examples to understand vocabulary, transliteration, and grammar:\n\n"
        f"FEW-SHOT EXAMPLES:\n{context}\n\n"
        "TASK:\n"
        f'Translate the following Sourashtra input into English and Tamil: "{sourashtra_input}"\n\n'
        "OUTPUT REQUIREMENT:\n"
        "You MUST respond ONLY with a strict JSON object with exactly two keys:\n"
        '{"english": "...", "tamil": "..."}\n'
        "Do NOT include markdown wrapping or extra prose."
    )

    xai_key = get_api_key("XAI_API_KEY")
    gemini_key = get_api_key("GEMINI_API_KEY") or get_api_key("GOOGLE_API_KEY")
    openai_key = get_api_key("OPENAI_API_KEY")

    # 1. Try xAI Grok API
    if xai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Sourashtra: {sourashtra_input}"}
                ],
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            parsed = parse_json_response(raw)
            if parsed:
                return parsed, "xAI Grok (grok-beta)"
        except Exception as e:
            st.error(f"Grok API error: {e}")

    # 2. Try Gemini API (Supports AIzaSy... and AQ... keys)
    if gemini_key and not gemini_key.endswith("_YOUR_API_KEY_HERE"):
        try:
            full_p = f"{system_prompt}\n\nSourashtra input: {sourashtra_input}"
            raw_text = call_gemini_raw(gemini_key, full_p)
            parsed = parse_json_response(raw_text)
            if parsed:
                return parsed, "Google Gemini (gemini-1.5-flash)"
        except Exception as e:
            st.error(f"Gemini API Error: {e}")

    # 3. Try OpenAI API
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Sourashtra: {sourashtra_input}"}
                ],
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            parsed = parse_json_response(raw)
            if parsed:
                return parsed, "OpenAI (gpt-4o-mini)"
        except Exception as e:
            st.error(f"OpenAI API error: {e}")

    # 4. Fallback dataset matching
    df = load_metadata_df()
    best_match = None
    for _, row in df.iterrows():
        stext = str(row.get("sourashtra_text", "")).strip().lower()
        if stext and stext in sourashtra_input.lower():
            best_match = str(row.get("english_phrase", ""))
            break

    fallback_eng = best_match if best_match else f"Translation for: '{sourashtra_input}'"
    fallback_tam = f"sourashtra phrase: {sourashtra_input}"
    
    return {
        "english": fallback_eng,
        "tamil": fallback_tam
    }, "Offline Dataset Fallback Mode"

def parse_json_response(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean)
        clean = re.sub(r"```$", "", clean).strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "english" in data and "tamil" in data:
            return data
    except Exception:
        pass
    
    eng_match = re.search(r'"english"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    tam_match = re.search(r'"tamil"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if eng_match and tam_match:
        return {
            "english": eng_match.group(1),
            "tamil": tam_match.group(1)
        }
    return None

def generate_speech(text, lang="en"):
    if not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text.strip(), lang=lang)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(fp.name)
        return fp.name
    except Exception as e:
        st.warning(f"gTTS warning for lang {lang}: {e}")
        return None


# --- Custom CSS ---
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e3c72;
    text-align: center;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 1.05rem;
    color: #475569;
    text-align: center;
    margin-bottom: 20px;
}
.prompt-card {
    background-color: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown('<div class="main-title">🎙️ Sourashtra Smart Voice Phrasebook & Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">4-Day Voice Collector & In-Context Few-Shot LLM Translator</div>', unsafe_allow_html=True)

# Session State for Prompt Index
if "prompt_idx" not in st.session_state:
    st.session_state.prompt_idx = 0

# --- App Sidebar ---
st.sidebar.title("⚙️ Navigation & Status")
st.sidebar.markdown(f"**Total Prompts**: {len(PROMPTS)}")
df_current = load_metadata_df()
st.sidebar.metric("Saved Phrases", len(df_current))

# Check API Keys
xai_k = get_api_key("XAI_API_KEY")
gemini_k = get_api_key("GEMINI_API_KEY") or get_api_key("GOOGLE_API_KEY")
openai_k = get_api_key("OPENAI_API_KEY")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Key Status")
if xai_k:
    st.sidebar.success("✅ xAI Grok Configured")
elif gemini_k and not gemini_k.endswith("_YOUR_API_KEY_HERE"):
    if gemini_k.startswith("AIzaSy") or gemini_k.startswith("AQ") or gemini_k.startswith("ya29"):
        st.sidebar.success("✅ Google Gemini Configured")
    else:
        st.sidebar.warning("⚠️ Unexpected Gemini Key format")
elif openai_k:
    st.sidebar.success("✅ OpenAI Configured")
else:
    st.sidebar.warning("⚠️ No API Key Detected (Running in Fallback Mode)")

st.sidebar.info("Add `XAI_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` in Streamlit Secrets or Environment Variables.")


# --- App Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🎙️ Voice Collector (Father's UI)",
    "🚀 Live Sourashtra Translator",
    "📊 Dataset Explorer"
])

# ==========================================
# TAB 1: Voice Collector
# ==========================================
with tab1:
    st.markdown("### 👨‍🦳 Collector Interface for Native Sourashtra Speaker")
    st.markdown("Record daily English phrases in Sourashtra audio and type transliterated text. Each saved entry expands our LLM few-shot dataset.")

    col1, col2 = st.columns([2, 1])

    with col1:
        current_idx = st.session_state.prompt_idx
        st.progress((current_idx + 1) / len(PROMPTS), text=f"Prompt {current_idx + 1} of {len(PROMPTS)}")

        st.markdown(f"""
        <div class="prompt-card">
            <span style="font-size: 0.95rem; color: #64748b;">English Prompt #{current_idx + 1}</span><br/>
            <span style="font-size: 1.6rem; font-weight: 700; color: #0f172a;">"{PROMPTS[current_idx]}"</span>
        </div>
        """, unsafe_allow_html=True)

        n_col1, n_col2 = st.columns(2)
        if n_col1.button("⬅️ Previous Prompt", use_container_width=True):
            st.session_state.prompt_idx = max(0, current_idx - 1)
            st.rerun()

        if n_col2.button("Next Prompt ➡️", use_container_width=True):
            st.session_state.prompt_idx = min(len(PROMPTS) - 1, current_idx + 1)
            st.rerun()

        st.markdown("---")
        st.markdown("#### 1-Tap Voice Recording")
        
        # Streamlit Audio Input widget
        audio_bytes = None
        if hasattr(st, "audio_input"):
            audio_bytes = st.audio_input("🎙️ Record Sourashtra Audio", key=f"rec_{current_idx}")
        else:
            audio_file = st.file_uploader("Upload Sourashtra Audio (.wav / .mp3)", type=["wav", "mp3"], key=f"up_{current_idx}")
            if audio_file:
                audio_bytes = audio_file.read()

        sourashtra_text = st.text_area(
            "✍️ Sourashtra Transliteration / Text (Optional)",
            placeholder="e.g. Chaa piyo ki? / Kayi jaatas?",
            height=100,
            key=f"text_{current_idx}"
        )

        if st.button("💾 Save & Next Phrase", type="primary", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            phrase_id = f"phrase_{timestamp}"
            
            saved_audio_rel = ""
            if audio_bytes:
                target_filename = f"{phrase_id}.wav"
                target_full = os.path.join(AUDIO_DIR, target_filename)
                with open(target_full, "wb") as f:
                    if hasattr(audio_bytes, "read"):
                        f.write(audio_bytes.read())
                    else:
                        f.write(audio_bytes)
                saved_audio_rel = os.path.join("audio", target_filename)

            new_row = pd.DataFrame([{
                "id": phrase_id,
                "timestamp": datetime.now().isoformat(),
                "english_phrase": PROMPTS[current_idx],
                "sourashtra_text": (sourashtra_text or "").strip(),
                "audio_file": saved_audio_rel
            }])

            df_meta = load_metadata_df()
            df_meta = pd.concat([df_meta, new_row], ignore_index=True)
            df_meta.to_csv(METADATA_PATH, index=False)

            st.success(f"Saved phrase #{current_idx + 1}! Advancing to next prompt...")

            st.session_state.prompt_idx = min(len(PROMPTS) - 1, current_idx + 1)
            st.rerun()

    with col2:
        st.markdown("#### 📈 Progress Stats")
        df_meta = load_metadata_df()
        st.metric("Total Dataset Entries", len(df_meta))
        st.dataframe(df_meta, use_container_width=True, hide_index=True)


# ==========================================
# TAB 2: Live Sourashtra Translator
# ==========================================
with tab2:
    st.markdown("### 🤖 In-Context Few-Shot Translator (Grok / Gemini / OpenAI)")
    st.markdown("Translates Sourashtra phrases into English & Tamil using collected phrasebook examples as few-shot learning context.")

    sourashtra_input = st.text_area(
        "🗣️ Input Sourashtra Phrase (Typed or Transliterated)",
        placeholder="Type Sourashtra phrase (e.g. Chaa piyo ki? / Kayi jaatas? / Aavjo baihiye)",
        height=100
    )

    if st.button("🚀 Translate Phrase", type="primary"):
        if not sourashtra_input or not sourashtra_input.strip():
            st.warning("Please type a Sourashtra phrase to translate.")
        else:
            with st.spinner("Translating using In-Context LLM Engine..."):
                res, provider = call_llm_translation(sourashtra_input.strip())
                eng_text = res.get("english", "")
                tam_text = res.get("tamil", "")

                st.info(f"⚡ Engine Used: **{provider}**")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("🇬🇧 English Translation")
                    st.success(eng_text)
                    eng_audio_file = generate_speech(eng_text, lang="en")
                    if eng_audio_file:
                        st.audio(eng_audio_file, format="audio/mp3")

                with res_col2:
                    st.subheader("🇮🇳 Tamil Translation (தமிழ்)")
                    st.success(tam_text)
                    tam_audio_file = generate_speech(tam_text, lang="ta")
                    if tam_audio_file:
                        st.audio(tam_audio_file, format="audio/mp3")


# ==========================================
# TAB 3: Dataset Explorer
# ==========================================
with tab3:
    st.markdown("### 📂 Collected Sourashtra Dataset & Metadata")

    df_meta = load_metadata_df()
    st.dataframe(df_meta, use_container_width=True)

    csv_data = df_meta.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Metadata CSV",
        data=csv_data,
        file_name="sourashtra_metadata.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📜 Target Phrasebook Prompts (30 Phrases)")
    prompts_df = pd.DataFrame({"Index": range(1, len(PROMPTS) + 1), "English Phrase": PROMPTS})
    st.dataframe(prompts_df, use_container_width=True, hide_index=True)
