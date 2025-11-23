from paddleocr import PaddleOCR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re, wordninja
from spellchecker import SpellChecker
import language_tool_python
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os


spell = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')

def normalize_ocr(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    tokens = []
    for w in text.split():
        if len(w) > 12:
            tokens.extend(wordninja.split(w.lower()))
        else:
            tokens.append(w)
    return " ".join(tokens)

def polish(text: str) -> str:
    text = normalize_ocr(text)
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    words = corrected.split()
    misspelled = spell.unknown(words)
    fixed = [spell.correction(w) if w in misspelled and spell.correction(w) else w for w in words]
    final = " ".join(fixed).strip()
    if final and not final.endswith(('.', '!', '?')):
        final += '.'
    return final[0].upper() + final[1:] if final else final

model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

grammar_corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


#PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

letters = (ocr.predict(input="P1000649-e1716905214471.jpg"))

#Grammar fix & result
if letters and isinstance(letters, list) and 'rec_texts' in letters[0]:
    text_lines = letters[0]['rec_texts']
    full_text = "\n".join(text_lines)
    result = grammar_corrector(polish(full_text), max_length=128, num_return_sequences=1)
    print(result)



else:
    print("No text found.")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

selected_file = None

def choose_file():
    global selected_file
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if file_path:
        selected_file = file_path
        file_label.configure(text=f"📄 {os.path.basename(file_path)}")

def run_ocr_background():
    try:
        letters = ocr.predict(input=selected_file)

        if letters and isinstance(letters, list) and 'rec_texts' in letters[0]:
            text_lines = letters[0]['rec_texts']
            full_text = "\n".join(text_lines)

            polished = polish(full_text)
            result = grammar_corrector(polished, max_length=128, num_return_sequences=1)

            output_box.delete("0.0", "end")
            output_box.insert("end", result[0]["generated_text"])
        else:
            output_box.delete("0.0", "end")
            output_box.insert("end", "No text found.")
    except Exception as e:
        output_box.delete("0.0", "end")
        output_box.insert("end", f"Error: {str(e)}")

    loading_label.configure(text="")
    run_btn.configure(state="normal")

def run_ocr():
    if not selected_file:
        messagebox.showerror("Error", "No file selected.")
        return

    loading_label.configure(text="⏳ Processing... Please wait...")
    run_btn.configure(state="disabled")

    worker = threading.Thread(target=run_ocr_background)
    worker.start()

app = ctk.CTk()
app.title("OCR + Grammar Correction")
app.geometry("700x600")

frame = ctk.CTkFrame(app, corner_radius=15)
frame.pack(padx=20, pady=20, fill="both", expand=True)

title = ctk.CTkLabel(frame, text="OCR + Grammar Correction", font=("Segoe UI", 24, "bold"))
title.pack(pady=10)

choose_btn = ctk.CTkButton(frame, text="Choose Image", command=choose_file, height=40, width=200)
choose_btn.pack(pady=10)

file_label = ctk.CTkLabel(frame, text="No file selected", font=("Segoe UI", 14))
file_label.pack()

run_btn = ctk.CTkButton(frame, text="Run OCR", command=run_ocr, height=40, width=200)
run_btn.pack(pady=15)

loading_label = ctk.CTkLabel(frame, text="", font=("Segoe UI", 14))
loading_label.pack(pady=5)

output_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Segoe UI", 13))
output_box.pack(pady=10)

app.mainloop()
