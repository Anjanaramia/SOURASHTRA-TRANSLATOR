import os
import shutil
import json
import re
import tempfile
from datetime import datetime
import pandas as pd
import gradio as gr
from gtts import gTTS

# --- Direct Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.txt")
DATASET_DIR = os.path.join(BASE_DIR, "sourashtra_dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Ensure required directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Ensure metadata CSV exists
if not os.path.exists(METADATA_PATH):
    df_init = pd.DataFrame(columns=["id", "timestamp", "english_phrase", "sourashtra_text", "audio_file"])
    df_init.to_csv(METADATA_PATH, index=False)


# --- Prompt & Metadata Helper Functions ---
def load_prompts():
    if os.path.exists(PROMPTS_PATH):
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if prompts:
            return prompts
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
        # Default seed examples if dataset is empty
        examples = [
            '- Sourashtra: "Chaa/Kapee piyo ki?" <-> English: "Did you drink tea/coffee?"',
            '- Sourashtra: "Kayi jaatas?" <-> English: "Where are you going?"',
            '- Sourashtra: "Aavjo baihiye." <-> English: "Please come and sit."'
        ]
    return "\n".join(examples)


# --- LLM API Integration (Grok / Gemini / OpenAI / Fallback) ---
def call_llm_translation(sourashtra_input):
    context = build_few_shot_context()
    
    system_prompt = (
        "You are an expert linguist specializing in the Sourashtra language (an Indo-Aryan language spoken by Sourashtrians in Tamil Nadu, India) and its translation into English and Tamil.\n"
        "Use the following in-context few-shot dataset examples to understand vocabulary, transliteration, and grammar:\n\n"
        f"FEW-SHOT EXAMPLES:\n{context}\n\n"
        "TASK:\n"
        f'Translate the following Sourashtra input into English and Tamil: "{sourashtra_input}"\n\n'
        "OUTPUT REQUIREMENT:\n"
        "You MUST respond ONLY with a strict JSON object with exactly two keys:\n"
        '{"english": "...", "tamil": "..."}\n'
        "Do NOT include markdown wrapping or extra prose."
    )

    xai_key = os.environ.get("XAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

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
            print(f"Grok API error: {e}")

    # 2. Try Gemini API
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"{system_prompt}\n\nSourashtra input: {sourashtra_input}"
            )
            raw = response.text.strip()
            parsed = parse_json_response(raw)
            if parsed:
                return parsed, "Google Gemini (gemini-1.5-flash)"
        except Exception as e:
            print(f"Gemini API error: {e}")

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
            print(f"OpenAI API error: {e}")

    # 4. Smart Fallback matching against collected dataset
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
    }, "Offline Fallback Mode (No API key found or API call failed)"


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
    
    # Regex fallback if JSON string wrapping occurred
    eng_match = re.search(r'"english"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    tam_match = re.search(r'"tamil"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if eng_match and tam_match:
        return {
            "english": eng_match.group(1),
            "tamil": tam_match.group(1)
        }
    return None


# --- gTTS Audio Helper ---
def generate_speech_audio(text, lang="en"):
    if not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text.strip(), lang=lang)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(fp.name)
        return fp.name
    except Exception as e:
        print(f"gTTS error for lang {lang}: {e}")
        return None


# --- Gradio Event Handlers ---

def save_and_next(prompt_idx, audio_file, sourashtra_text):
    if prompt_idx < 0 or prompt_idx >= len(PROMPTS):
        prompt_idx = 0
        
    current_english = PROMPTS[prompt_idx]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phrase_id = f"phrase_{timestamp}"
    
    audio_rel_path = ""
    if audio_file and os.path.exists(audio_file):
        ext = os.path.splitext(audio_file)[1] or ".wav"
        target_filename = f"{phrase_id}{ext}"
        target_full = os.path.join(AUDIO_DIR, target_filename)
        shutil.copy(audio_file, target_full)
        audio_rel_path = os.path.join("audio", target_filename)
        
    # Append to metadata.csv
    new_row = pd.DataFrame([{
        "id": phrase_id,
        "timestamp": datetime.now().isoformat(),
        "english_phrase": current_english,
        "sourashtra_text": (sourashtra_text or "").strip(),
        "audio_file": audio_rel_path
    }])
    
    df = load_metadata_df()
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(METADATA_PATH, index=False)
    
    # Calculate next prompt index
    next_idx = prompt_idx + 1
    if next_idx >= len(PROMPTS):
        next_idx = len(PROMPTS) - 1
        status = "🎉 Excellent! You have recorded all 30 prompts in the phrasebook dataset!"
    else:
        status = f"✅ Saved Phrase #{prompt_idx + 1}! Audio: {'Recorded' if audio_rel_path else 'Skipped'}, Text: '{sourashtra_text or 'N/A'}'"

    next_prompt = PROMPTS[next_idx]
    progress_str = f"Prompt {next_idx + 1} of {len(PROMPTS)}"
    
    return (
        next_idx,
        next_prompt,
        progress_str,
        None,  # Reset audio input
        "",    # Reset text input
        status,
        load_metadata_df(),
        f"Saved Entries: {len(df)}"
    )

def nav_prev(prompt_idx):
    prev_idx = max(0, prompt_idx - 1)
    return (
        prev_idx,
        PROMPTS[prev_idx],
        f"Prompt {prev_idx + 1} of {len(PROMPTS)}",
        None,
        "",
        f"Navigated to Phrase #{prev_idx + 1}"
    )

def nav_next(prompt_idx):
    next_idx = min(len(PROMPTS) - 1, prompt_idx + 1)
    return (
        next_idx,
        PROMPTS[next_idx],
        f"Prompt {next_idx + 1} of {len(PROMPTS)}",
        None,
        "",
        f"Navigated to Phrase #{next_idx + 1}"
    )

def handle_translation(sourashtra_input):
    if not sourashtra_input or not sourashtra_input.strip():
        return (
            "⚠️ Please type or speak a Sourashtra phrase to translate.",
            "N/A",
            None,
            None,
            "⚠️ Input is empty."
        )

    res, provider = call_llm_translation(sourashtra_input.strip())
    eng_text = res.get("english", "")
    tam_text = res.get("tamil", "")

    # Generate Speech Audio via gTTS
    eng_audio = generate_speech_audio(eng_text, lang="en")
    tam_audio = generate_speech_audio(tam_text, lang="ta")

    api_banner = f"⚡ Engine Used: **{provider}**"
    if "Offline" in provider:
        api_banner += "\n\n> ⚠️ *No API key detected for xAI Grok, Gemini, or OpenAI. Set `XAI_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` to enable live LLM synthesis.*"

    return eng_text, tam_text, eng_audio, tam_audio, api_banner


# --- Custom CSS Styling ---
CUSTOM_CSS = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
}
.main-header p {
    font-size: 1.05rem;
    opacity: 0.9;
    margin: 0;
}
.prompt-card {
    background: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 15px;
}
.prompt-text {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-top: 10px;
}
.badge-info {
    display: inline-block;
    background: #3b82f6;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
}
.action-btn {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
"""

# --- Build Gradio Interface ---
with gr.Blocks(title="Sourashtra Smart Voice Phrasebook & Translator") as app:

    # App Header
    gr.HTML("""
    <div class="main-header">
        <h1>🎙️ Sourashtra Smart Voice Phrasebook & Translator</h1>
        <p>4-Day Modern Voice Collector & In-Context Few-Shot LLM Translator</p>
    </div>
    """)

    # State Variables
    current_prompt_index = gr.State(value=0)

    with gr.Tabs() as main_tabs:

        # ==========================================
        # TAB 1: Father's Voice Recording UI (Collector)
        # ==========================================
        with gr.TabItem("🎙️ Voice Collector (Father's UI)", id=1):
            gr.Markdown("### 👨‍🦳 Collector Interface for Native Sourashtra Speaker")
            gr.Markdown("Record daily English phrases in Sourashtra audio and type transliterated text. Each saved entry expands our LLM few-shot context.")

            with gr.Row():
                with gr.Column(scale=2):
                    progress_badge = gr.Markdown(value=f"Prompt 1 of {len(PROMPTS)}", elem_classes=["badge-info"])
                    prompt_display = gr.Markdown(value=f"### 💬 English Phrase:\n# **\"{PROMPTS[0]}\"**", elem_classes=["prompt-card"])

                    with gr.Row():
                        btn_prev = gr.Button("⬅️ Previous", variant="secondary")
                        btn_next = gr.Button("Next ➡️", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown("#### 1-Tap Mobile Audio Recording")
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎙️ Record Sourashtra Audio",
                        elem_id="sourashtra_mic"
                    )

                    sourashtra_text_input = gr.Textbox(
                        label="✍️ Sourashtra Transliteration / Text (Optional)",
                        placeholder="e.g. Chaa piyo ki? / Kayi jaatas?",
                        lines=2
                    )

                    btn_save = gr.Button("💾 Save & Next Phrase", variant="primary", elem_classes=["action-btn"], size="lg")
                    save_status = gr.Markdown(value="Ready to record.")

                with gr.Column(scale=1):
                    gr.Markdown("#### 📈 Collection Progress")
                    initial_df = load_metadata_df()
                    dataset_count_badge = gr.Markdown(value=f"Saved Entries: **{len(initial_df)}**")
                    dataset_preview = gr.Dataframe(
                        value=initial_df,
                        headers=["id", "timestamp", "english_phrase", "sourashtra_text", "audio_file"],
                        interactive=False,
                        wrap=True
                    )

        # ==========================================
        # TAB 2: Live Sourashtra Translator
        # ==========================================
        with gr.TabItem("🚀 Live Sourashtra Translator", id=2):
            gr.Markdown("### 🤖 In-Context Few-Shot Translator (Grok / Gemini / OpenAI)")
            gr.Markdown("Translates Sourashtra phrases into English & Tamil using recorded dataset examples as in-context learning.")

            api_status_banner = gr.Markdown(
                value="⚡ *System initialized. Ready to translate using loaded dataset context.*"
            )

            with gr.Row():
                with gr.Column():
                    sourashtra_query = gr.Textbox(
                        label="🗣️ Input Sourashtra Phrase (Typed or Transliterated)",
                        placeholder="Type Sourashtra phrase (e.g., Chaa piyo ki? / Kayi jaatas? / Aavjo baihiye)",
                        lines=3
                    )
                    btn_translate = gr.Button("🚀 Translate Phrase", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### 📄 Structured Translation Output")
                    translated_english = gr.Textbox(label="🇬🇧 English Translation", interactive=False)
                    translated_tamil = gr.Textbox(label="🇮🇳 Tamil Translation (தமிழ்)", interactive=False)

                    with gr.Row():
                        audio_english = gr.Audio(label="🔊 English Speech Playback (gTTS)", interactive=False)
                        audio_tamil = gr.Audio(label="🔊 Tamil Speech Playback (gTTS)", interactive=False)

        # ==========================================
        # TAB 3: Dataset Explorer & Phrasebook List
        # ==========================================
        with gr.TabItem("📊 Dataset Explorer", id=3):
            gr.Markdown("### 📂 Collected Sourashtra Dataset & Metadata")
            
            with gr.Row():
                btn_refresh = gr.Button("🔄 Refresh Dataset View")
            
            full_dataset_table = gr.Dataframe(
                value=load_metadata_df(),
                headers=["id", "timestamp", "english_phrase", "sourashtra_text", "audio_file"],
                interactive=False,
                wrap=True
            )
            
            gr.Markdown("### 📜 Target Phrasebook Prompts (30 Phrases)")
            prompts_df = pd.DataFrame({"Index": range(1, len(PROMPTS) + 1), "English Phrase": PROMPTS})
            gr.Dataframe(value=prompts_df, interactive=False)

    # --- Event Wiring ---

    # Tab 1 Events
    btn_save.click(
        fn=save_and_next,
        inputs=[current_prompt_index, audio_input, sourashtra_text_input],
        outputs=[
            current_prompt_index,
            prompt_display,
            progress_badge,
            audio_input,
            sourashtra_text_input,
            save_status,
            dataset_preview,
            dataset_count_badge
        ]
    )

    btn_prev.click(
        fn=nav_prev,
        inputs=[current_prompt_index],
        outputs=[current_prompt_index, prompt_display, progress_badge, audio_input, sourashtra_text_input, save_status]
    )

    btn_next.click(
        fn=nav_next,
        inputs=[current_prompt_index],
        outputs=[current_prompt_index, prompt_display, progress_badge, audio_input, sourashtra_text_input, save_status]
    )

    # Tab 2 Events
    btn_translate.click(
        fn=handle_translation,
        inputs=[sourashtra_query],
        outputs=[translated_english, translated_tamil, audio_english, audio_tamil, api_status_banner]
    )

    # Tab 3 Events
    btn_refresh.click(
        fn=load_metadata_df,
        inputs=[],
        outputs=[full_dataset_table]
    )

if __name__ == "__main__":
    # Ensure share=True for public mobile testing URL generation as requested
    try:
        app.launch(share=True, css=CUSTOM_CSS)
    except Exception as e:
        print(f"Share link creation error: {e}. Launching locally.")
        app.launch(share=False, css=CUSTOM_CSS)
