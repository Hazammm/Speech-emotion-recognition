"""
Gradio-based web demo for Speech Emotion Recognition.
Upload or record audio and get instant emotion predictions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.predict import load_latest_model, predict_emotion
from src.feature_extraction import load_audio, extract_features


print("Loading model...")
try:
    model, scaler, label_encoder, model_info = load_latest_model()
    print("Model loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please train a model first: python main.py --train")
    sys.exit(1)


EMOTION_STYLES = {
    "neutral":  {"emoji": "😐", "color": "#95A5A6", "desc": "Calm and neutral tone"},
    "happy":    {"emoji": "😊", "color": "#F1C40F", "desc": "Joyful and upbeat expression"},
    "sad":      {"emoji": "😢", "color": "#3498DB", "desc": "Sorrowful and melancholic tone"},
    "angry":    {"emoji": "😠", "color": "#E74C3C", "desc": "Intense and aggressive expression"},
    "fearful":  {"emoji": "😨", "color": "#9B59B6", "desc": "Anxious and frightened tone"},
    "disgust":  {"emoji": "🤢", "color": "#27AE60", "desc": "Repulsed and displeased expression"},
    "surprise": {"emoji": "😮", "color": "#E67E22", "desc": "Shocked and astonished tone"},
}


def analyze_emotion(audio_path):
    """
    Main prediction function for the Gradio interface.

    Args:
        audio_path: str, path to the audio file from Gradio

    Returns:
        tuple: (emotion_html, probabilities_dict)
    """
    if audio_path is None:
        return "<p style='text-align:center; color:#999;'>Please upload or record an audio file.</p>", {}

    try:
        result = predict_emotion(audio_path, model, scaler, label_encoder)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p style='text-align:center; color:#E74C3C;'>Error analyzing audio: {str(e)}</p>", {}

    emotion = result["emotion"]
    confidence = result["confidence"]
    style = EMOTION_STYLES.get(emotion, {"emoji": "🎭", "color": "#888", "desc": ""})

    html = f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, {style['color']}22, {style['color']}44);
        border: 2px solid {style['color']};
        margin: 10px 0;
    ">
        <div style="font-size: 64px; margin-bottom: 10px;">{style['emoji']}</div>
        <div style="
            font-size: 28px;
            font-weight: bold;
            color: {style['color']};
            text-transform: uppercase;
            letter-spacing: 3px;
        ">{emotion}</div>
        <div style="
            font-size: 16px;
            color: #666;
            margin-top: 8px;
        ">{style['desc']}</div>
        <div style="
            font-size: 42px;
            font-weight: bold;
            color: {style['color']};
            margin-top: 15px;
        ">{confidence:.1%}</div>
        <div style="font-size: 13px; color: #999;">confidence</div>
    </div>
    """

    probs = {
        f"{EMOTION_STYLES.get(e, {}).get('emoji', '🎭')} {e.capitalize()}": p
        for e, p in result["probabilities"].items()
    }

    return html, probs


with gr.Blocks(
    title="Speech Emotion Recognition",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .gradio-container { max-width: 800px !important; }
        .gr-button { border-radius: 12px !important; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🎙️ Speech Emotion Recognition
    ### Detect emotions from voice using Deep Learning

    Upload an audio file or record your voice to analyze the emotion.
    The model detects **7 emotions**: neutral, happy, sad, angry, fearful, disgust, and surprise.

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 Audio Input",
                type="filepath",
                sources=["upload", "microphone"],
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Emotion",
                variant="primary",
                size="lg",
            )
            gr.Markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 12px 16px;
                border-radius: 10px;
                margin-top: 10px;
                font-size: 13px;
                color: #666;
            ">
                <b>Model:</b> {model_info['model_type'].upper()} &nbsp;|&nbsp;
                <b>Trained:</b> {model_info['timestamp']}
            </div>
            """)

        with gr.Column(scale=1):
            emotion_output = gr.HTML(
                label="Detected Emotion",
                value="<p style='text-align:center; color:#999; padding:40px;'>Results will appear here...</p>"
            )
            prob_output = gr.Label(
                label="📊 Emotion Probabilities",
                num_top_classes=7,
            )

    analyze_btn.click(
        fn=analyze_emotion,
        inputs=[audio_input],
        outputs=[emotion_output, prob_output],
    )

    audio_input.change(
        fn=analyze_emotion,
        inputs=[audio_input],
        outputs=[emotion_output, prob_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
