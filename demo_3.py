# demo_3.py

# !pip install transformers evaluate gradio requests rouge_score
# !pip install sacrebleu

import gradio as gr
import evaluate
import requests
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load BLEU and ROUGE evaluators
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

# Load the model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Define available language options
language_options = {
    'en': 'English',
    'fr': 'French',
    'hi': 'Hindi',
    'es': 'Spanish',
    'de': 'German',
    'zh': 'Chinese',
}

# Placeholder for Gemini API endpoint
GEMINI_API_URL = "https://gemini-api-url.com/process"
API_KEY = "YOUR_API_KEY"

def process_with_gemini(source_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "text": source_text
    }

    try:
        response = requests.post(GEMINI_API_URL, json=data, headers=headers)
        response.raise_for_status()
        processed_text = response.json().get('processed_text', source_text)
        return processed_text
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return source_text

def translate_text(source_text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded_text = tokenizer(source_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def evaluate_translation(reference, hypothesis):
    bleu_score = bleu.compute(predictions=[hypothesis], references=[[reference]])
    rouge_score = rouge.compute(predictions=[hypothesis], references=[[reference]])
    return bleu_score, rouge_score

def translation_and_evaluation(source_text, source_lang, target_lang):
    processed_text = process_with_gemini(source_text)
    translated_text = translate_text(processed_text, source_lang, target_lang)
    bleu_score, rouge_score = evaluate_translation(source_text, translated_text)
    return translated_text, bleu_score['score'], rouge_score

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Language Translation with Gemini API and Evaluation")
    with gr.Row():
        source_text_input = gr.Textbox(label="Enter text to translate:")
        source_lang_input = gr.Dropdown(choices=list(language_options.keys()), label="Source Language")
        target_lang_input = gr.Dropdown(choices=list(language_options.keys()), label="Target Language")
    translate_button = gr.Button("Translate and Evaluate")
    translated_output = gr.Textbox(label="Translated Text", interactive=False)
    bleu_output = gr.Textbox(label="BLEU Score", interactive=False)
    rouge_output = gr.Textbox(label="ROUGE Score", interactive=False)

    translate_button.click(
        fn=translation_and_evaluation,
        inputs=[source_text_input, source_lang_input, target_lang_input],
        outputs=[translated_output, bleu_output, rouge_output]
    )

# Launch the interface
demo.launch()
