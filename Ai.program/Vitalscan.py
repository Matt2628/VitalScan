import os
import shutil
import threading
import zipfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# -------------------------
# Import: partner sin "oversettelse" av modell-output
# -------------------------
# translate_prediction(pred) skal ta modellens prediksjon og returnere:
#   main_burden (str) + burden_probs (dict)
# burden_probs bør inneholde keys som matcher ORDER-listen.
try:
    from translated_data import translate_prediction
except Exception as e:
    translate_prediction = None
    _translate_import_error = e


# -------------------------
# Config / konstanter
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Standardmodell vi prøver å laste først
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
# Hvis best_model.keras ikke finnes, prøv alternativt navn
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "mri_analysis_model.keras")

# Model path som brukes i appen (kan endres via "Velg modell")
MODEL_PATH = DEFAULT_MODEL_PATH

# Rekkefølge på stadiene (slik det skal vises til bruker/lege)
ORDER = [
    "No evidence of disease (NED)",
    "Low tumor burden",
    "Moderate tumor burden",
    "High tumor burden (Advanced-stage cancer)",
]

# Cache av modellen slik at den bare lastes én gang (raskere)
_MODEL = None
_MODEL_LOCK = threading.Lock()


# -------------------------
# Helper-funksjoner (små verktøy)
# -------------------------
def shorten_path(p: str, max_len: int = 70) -> str:
    """
    Forkorter lange filstier slik at UI ikke blir stygt.
    Eksempel: C:/.../very_long_name.jpg
    """
    if len(p) <= max_len:
        return p
    keep = max_len - 5
    left = keep // 2
    right = keep - left
    return f"{p[:left]}...{p[-right:]}"


def normalize_to_percent(prob_dict: dict) -> dict:
    """
    Noen funksjoner returnerer sannsynligheter i 0..1,
    andre returnerer 0..100.
    Denne normaliserer ALT til prosent (0..100).
    """
    if not prob_dict:
        return {k: 0.0 for k in ORDER}
    max_val = max(prob_dict.values())
    if max_val <= 1.0:
        return {k: float(v) * 100.0 for k, v in prob_dict.items()}
    return {k: float(v) for k, v in prob_dict.items()}


def normalize_date(date_str: str) -> str:
    """
    Godtar:
      - YYYY-MM-DD
      - DD.MM.YYYY
    Returnerer:
      - YYYY-MM-DD (hvis mulig)
    """
    s = date_str.strip()
    if not s:
        return s

    # YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s

    # DD.MM.YYYY -> YYYY-MM-DD
    if len(s) == 10 and s[2] == "." and s[5] == ".":
        dd = s[0:2]
        mm = s[3:5]
        yyyy = s[6:10]
        return f"{yyyy}-{mm}-{dd}"

    # fallback (hvis user skriver noe annet)
    return s


def format_report(patient_name: str, exam_date: str, main_burden: str, probs_pct: dict) -> str:
    """
    Lager den "legevennlige" rapporten i ASCII-boks format,
    så den kan kopieres rett inn i journal/system.
    """
    lines = []
    lines.append("|------------------------------------------------------------|")
    lines.append("|                                                            |")
    lines.append("|              MRI BRAIN - DIAGNOSTIC SUMMARY                |")
    lines.append("|                                                            |")
    lines.append("|------------------------------------------------------------|")
    lines.append("|                                                            |")
    lines.append("| PATIENT AND EXAMINATION DETAILS                            |")
    lines.append("|                                                            |")
    lines.append(f"| Patient Name: {patient_name}".ljust(61) + "|")
    lines.append(f"| Examination Date: {exam_date}".ljust(61) + "|")
    lines.append("| Diagnostic Software: VITALScan                             |")
    lines.append("| Imaging Modality: MRI Brain                                |")
    lines.append("|------------------------------------------------------------|")
    lines.append("|                                                            |")
    lines.append("| DIAGNOSTIC BURDEN DISTRIBUTION                             |")
    lines.append("|                                                            |")
    lines.append("| Tumor Burden Assessment:                                   |")

    for label in ORDER:
        v = probs_pct.get(label, 0.0)
        line = f"| • {label}: {v:.1f}%"
        lines.append(line.ljust(61) + "|")

    lines.append("|------------------------------------------------------------|")
    lines.append("|                                                            |")
    lines.append("| OVERALL CLINICAL IMPRESSION:                               |")
    lines.append(f"| {main_burden}".ljust(61) + "|")
    lines.append("|------------------------------------------------------------|")

    return "\n".join(lines)


def make_plain_summary(main_burden: str, probs_pct: dict) -> str:
    """
    Lager en mer lesbar tekst til resultatsiden.
    Dette er for "vanlige" brukere/oversikt, ikke nødvendigvis journalformat.
    """
    items = sorted(probs_pct.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return f"Overall impression: {main_burden}"

    top_label, top_p = items[0]
    rest = items[1:]

    if rest:
        rest_txt = ", ".join([f"{name} ({p:.1f}%)" for name, p in rest])
        return (
            f"Model’s overall impression: {main_burden}\n"
            f"Most likely category: {top_label} ({top_p:.1f}%).\n"
            f"Other possibilities: {rest_txt}."
        )
    return f"Model’s overall impression: {main_burden}\nMost likely category: {top_label} ({top_p:.1f}%)."


# -------------------------
# Modell-lasting / inferens
# -------------------------
def _try_load_model(path: str):
    """Wrapper rundt tf.keras.models.load_model."""
    return tf.keras.models.load_model(path)


def load_model_cached(model_path: str):
    """
    Laster modellen én gang og cacher den i _MODEL.
    Bruker lock siden inferens kjøres i tråd (thread).
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Fant ikke modellfilen.\n\n"
                f"Path:\n{model_path}\n\n"
                "Velg riktig modell via 'Velg modell'."
            )

        # Keras v3 .keras er egentlig en zip-fil.
        # Hvis det ikke er zip, kan filen være korrupt eller egentlig .h5.
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".keras" and not zipfile.is_zipfile(model_path):
            alt_h5 = os.path.splitext(model_path)[0] + ".h5"
            try:
                # Kopier til .h5 og prøv å laste (hjelper hvis noen har renamed fil)
                if not os.path.exists(alt_h5):
                    shutil.copyfile(model_path, alt_h5)
                _MODEL = _try_load_model(alt_h5)
                return _MODEL
            except Exception:
                raise ValueError(
                    "Modellfilen ser ikke ut som en gyldig .keras (zip).\n\n"
                    "Vanlige årsaker:\n"
                    "• filen er korrupt / ikke ferdig lastet ned\n"
                    "• feil format (.h5 renamed til .keras)\n"
                )

        _MODEL = _try_load_model(model_path)
        return _MODEL


def run_analysis(image_path: str, model_path: str):
    """
    Kjører hele pipeline:
      1) Sjekk at translate_prediction finnes
      2) Last modell
      3) Les bilde -> preprocess -> prediksjon
      4) Oversett prediksjon til "burden" via translate_prediction
      5) Normaliser til prosent
    """
    if translate_prediction is None:
        raise RuntimeError(
            "Fant ikke translate_prediction.\n\n"
            "Sjekk at translated_data.py ligger i samme mappe og har:\n"
            "  def translate_prediction(pred): ...\n\n"
            f"Import-feil: {_translate_import_error}"
        )

    model = load_model_cached(model_path)

    # Preprocess bilde slik ResNet50 forventer
    img = image.load_img(image_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # Modellprediksjon (shape avhenger av modell)
    pred = model.predict(arr, verbose=0)

    # Partner-funksjon gjør pred om til kliniske kategorier
    main_burden, burden_probs = translate_prediction(pred)

    # Sikrer at alle keys finnes + normaliserer
    safe_probs = {k: float(burden_probs.get(k, 0.0)) for k in ORDER}
    probs_pct = normalize_to_percent(safe_probs)

    return str(main_burden), probs_pct


# -------------------------
# UI (Tkinter)
# -------------------------
root = tk.Tk()
root.title("VitalScan")
root.geometry("1040x660")
root.minsize(940, 580)
root.configure(bg="#f3f4f6")

style = ttk.Style()
style.theme_use("clam")

# Farger / theme
HEADER = "#0b1220"
CARD = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
ACCENT = "#2563eb"
BORDER = "#e5e7eb"

# Styles
style.configure("Header.TFrame", background=HEADER)
style.configure("Header.TLabel", background=HEADER, foreground="white", font=("Segoe UI", 16, "bold"))

style.configure("Card.TFrame", background=CARD, relief="solid", borderwidth=1)
style.configure("Title.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 16, "bold"))
style.configure("Body.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 11))
style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 10))
style.configure("Section.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 11, "bold"))

style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 10), foreground="white")
style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("!disabled", ACCENT)])

style.configure("Prob.Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT)

# Entry styling
style.configure("Input.TEntry", padding=(10, 8), relief="solid")

# Root layout
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# Header bar
header = ttk.Frame(root, style="Header.TFrame", padding=(18, 12))
header.grid(row=0, column=0, sticky="ew")
ttk.Label(header, text="VitalScan", style="Header.TLabel").grid(row=0, column=0, sticky="w")

# Main container
main = ttk.Frame(root, padding=24)
main.grid(row=1, column=0, sticky="nsew")
main.grid_columnconfigure(0, weight=1)
main.grid_rowconfigure(0, weight=1)

# Card container
card = ttk.Frame(main, style="Card.TFrame", padding=22)
card.grid(row=0, column=0, sticky="nsew")
card.grid_columnconfigure(0, weight=1)
card.grid_rowconfigure(0, weight=1)

# -------------------------
# "Pages" (2 views): Upload page + Result page
# -------------------------
page_upload = ttk.Frame(card, style="Card.TFrame")
page_result = ttk.Frame(card, style="Card.TFrame")

for p in (page_upload, page_result):
    p.grid(row=0, column=0, sticky="nsew")
    p.grid_remove()

def show_page(which: str):
    """Vis enten upload-page eller result-page."""
    page_upload.grid_remove()
    page_result.grid_remove()
    if which == "upload":
        page_upload.grid()
    else:
        page_result.grid()


# -------------------------
# Placeholder behavior (så input-felt ser bedre ut)
# -------------------------
def attach_placeholder(entry: ttk.Entry, placeholder: str):
    """
    Lager "placeholder text" i ttk.Entry (Tkinter har ikke dette innebygd).
    Når feltet er tomt og ikke aktivt, vises grå tekst.
    """
    entry.placeholder = placeholder  # type: ignore[attr-defined]
    entry.placeholder_active = False  # type: ignore[attr-defined]

    def set_ph():
        if not entry.get():
            entry.insert(0, placeholder)
            entry.configure(foreground=MUTED)
            entry.placeholder_active = True  # type: ignore[attr-defined]

    def clear_ph(_=None):
        if getattr(entry, "placeholder_active", False):
            entry.delete(0, "end")
            entry.configure(foreground=TEXT)
            entry.placeholder_active = False  # type: ignore[attr-defined]

    def on_focus_out(_=None):
        if not entry.get():
            set_ph()

    entry.bind("<FocusIn>", clear_ph)
    entry.bind("<FocusOut>", on_focus_out)
    set_ph()

def entry_value(entry: ttk.Entry) -> str:
    """Henter ekte verdi fra input-felt (ikke placeholder-tekst)."""
    val = entry.get().strip()
    if getattr(entry, "placeholder_active", False):
        return ""
    return val


# =========================================================
# PAGE 1: Upload & Patient info
# =========================================================
page_upload.grid_columnconfigure(0, weight=1)

ttk.Label(page_upload, text="Upload & Pasientinfo", style="Title.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(
    page_upload,
    text="Fyll inn pasientinfo, velg modell, og last opp MR-bilde.",
    style="Muted.TLabel",
    wraplength=880
).grid(row=1, column=0, sticky="w", pady=(6, 14))

ttk.Separator(page_upload).grid(row=2, column=0, sticky="ew", pady=(0, 14))

# Patient inputs (2 kolonner)
form = ttk.Frame(page_upload, style="Card.TFrame")
form.grid(row=3, column=0, sticky="ew")
form.grid_columnconfigure(0, weight=1)
form.grid_columnconfigure(1, weight=1)

ttk.Label(form, text="Fullt navn", style="Section.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
ttk.Label(form, text="Undersøkelsesdato", style="Section.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))

patient_entry = ttk.Entry(form, style="Input.TEntry")
patient_entry.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(6, 2))

date_entry = ttk.Entry(form, style="Input.TEntry")
date_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(6, 2))

attach_placeholder(patient_entry, "F.eks. Ola Nordmann")
attach_placeholder(date_entry, "YYYY-MM-DD eller DD.MM.YYYY")

ttk.Label(page_upload, text="Tips: Dato kan være 2026-02-26 eller 26.02.2026", style="Muted.TLabel")\
    .grid(row=4, column=0, sticky="w", pady=(6, 14))

ttk.Separator(page_upload).grid(row=5, column=0, sticky="ew", pady=(0, 14))

# Model section
ttk.Label(page_upload, text="Modell", style="Section.TLabel").grid(row=6, column=0, sticky="w")

model_display_var = tk.StringVar(value=shorten_path(MODEL_PATH, 95))
ttk.Label(page_upload, textvariable=model_display_var, style="Body.TLabel", wraplength=920)\
    .grid(row=7, column=0, sticky="w", pady=(6, 8))

btn_row_upload = ttk.Frame(page_upload, style="Card.TFrame")
btn_row_upload.grid(row=8, column=0, sticky="w", pady=(0, 14))

ttk.Separator(page_upload).grid(row=9, column=0, sticky="ew", pady=(0, 14))

# Image section (no preview)
ttk.Label(page_upload, text="MR-bilde", style="Section.TLabel").grid(row=10, column=0, sticky="w")

file_display_var = tk.StringVar(value="Ingen fil valgt")
ttk.Label(page_upload, textvariable=file_display_var, style="Body.TLabel", wraplength=920)\
    .grid(row=11, column=0, sticky="w", pady=(6, 10))

ttk.Label(page_upload, text="Når du laster opp bilde, blir du sendt til resultatsiden.", style="Muted.TLabel")\
    .grid(row=12, column=0, sticky="w", pady=(0, 2))


# =========================================================
# PAGE 2: Result page (med tabs: Overview + Diagnostic Report)
# =========================================================
page_result.grid_columnconfigure(0, weight=1)
page_result.grid_rowconfigure(4, weight=1)

# Top row: title + nav buttons
toprow = ttk.Frame(page_result, style="Card.TFrame")
toprow.grid(row=0, column=0, sticky="ew")
toprow.grid_columnconfigure(0, weight=1)

ttk.Label(toprow, text="Resultat", style="Title.TLabel").grid(row=0, column=0, sticky="w")
ttk.Label(toprow, text="Velg hva du vil se i fanene under.", style="Muted.TLabel")\
    .grid(row=1, column=0, sticky="w", pady=(4, 0))

nav_row = ttk.Frame(toprow, style="Card.TFrame")
nav_row.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

# Result headline
pred_var = tk.StringVar(value="—")
ttk.Label(page_result, textvariable=pred_var, style="Body.TLabel", font=("Segoe UI", 12, "bold"))\
    .grid(row=1, column=0, sticky="w", pady=(14, 10))

# Short readable summary
summary_var = tk.StringVar(value="")
ttk.Label(page_result, textvariable=summary_var, style="Body.TLabel", wraplength=980)\
    .grid(row=2, column=0, sticky="w", pady=(0, 12))

ttk.Separator(page_result).grid(row=3, column=0, sticky="ew", pady=(0, 12))

# Tabs
notebook = ttk.Notebook(page_result)
notebook.grid(row=4, column=0, sticky="nsew")

tab_overview = ttk.Frame(notebook)
tab_report = ttk.Frame(notebook)
notebook.add(tab_overview, text="Overview")
notebook.add(tab_report, text="Diagnostic Report")

# ---- Overview tab: bars/probabilities
tab_overview.grid_columnconfigure(0, weight=1)

ttk.Label(tab_overview, text="Tumor burden distribution", style="Section.TLabel")\
    .grid(row=0, column=0, sticky="w", pady=(10, 6))

prob_frame = ttk.Frame(tab_overview, style="Card.TFrame")
prob_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
prob_frame.grid_columnconfigure(1, weight=1)

ttk.Label(tab_overview, text="Merk: Dette er en modell-prediksjon (ikke en medisinsk diagnose).",
          style="Muted.TLabel")\
    .grid(row=2, column=0, sticky="w", pady=(10, 0))

# ---- Diagnostic Report tab: rapport i tekstfelt + scroll + copy
tab_report.grid_columnconfigure(0, weight=1)
tab_report.grid_rowconfigure(1, weight=1)

ttk.Label(tab_report, text="MRI BRAIN - DIAGNOSTIC SUMMARY", style="Section.TLabel")\
    .grid(row=0, column=0, sticky="w", pady=(10, 6))

report_text = tk.Text(
    tab_report, height=18, wrap="none",
    font=("Consolas", 10),
    bg="#0b1220", fg="#f9fafb",
    insertbackground="#f9fafb",
    relief="solid", borderwidth=1
)
report_text.grid(row=1, column=0, sticky="nsew")

scroll_y = ttk.Scrollbar(tab_report, orient="vertical", command=report_text.yview)
scroll_y.grid(row=1, column=1, sticky="ns")
report_text.configure(yscrollcommand=scroll_y.set)

scroll_x = ttk.Scrollbar(tab_report, orient="horizontal", command=report_text.xview)
scroll_x.grid(row=2, column=0, sticky="ew")
report_text.configure(xscrollcommand=scroll_x.set)

report_actions = ttk.Frame(tab_report, style="Card.TFrame")
report_actions.grid(row=3, column=0, sticky="w", pady=(10, 0))

# Progressbar som vises mens vi kjører analyse i tråd
progress = ttk.Progressbar(page_result, mode="indeterminate", length=620)
progress.grid(row=5, column=0, sticky="w", pady=(14, 0))
progress.grid_remove()


# -------------------------
# UI-oppdatering helpers
# -------------------------
def clear_prob_rows():
    """Fjerner gamle probability bars."""
    for w in prob_frame.winfo_children():
        w.destroy()

def set_report(text: str):
    """Setter inn rapporttekst i report-tab."""
    report_text.configure(state="normal")
    report_text.delete("1.0", "end")
    report_text.insert("1.0", text)
    report_text.configure(state="disabled")

def render_probs(probs_pct: dict):
    """Tegner probability bars basert på prosent."""
    clear_prob_rows()
    items = sorted(probs_pct.items(), key=lambda x: x[1], reverse=True)

    for r, (name, p) in enumerate(items):
        is_top = (r == 0)
        label_font = ("Segoe UI", 10, "bold") if is_top else ("Segoe UI", 10)
        pct = f"{p:.1f}%"

        ttk.Label(prob_frame, text=name, style="Body.TLabel", font=label_font)\
            .grid(row=r, column=0, sticky="w", padx=(0, 12), pady=7)

        bar = ttk.Progressbar(
            prob_frame, style="Prob.Horizontal.TProgressbar",
            mode="determinate", maximum=100, value=max(0.0, min(100.0, p))
        )
        bar.grid(row=r, column=1, sticky="ew", pady=7)

        ttk.Label(prob_frame, text=pct, style="Muted.TLabel", font=label_font)\
            .grid(row=r, column=2, sticky="e", padx=(12, 0), pady=7)

def set_busy(busy: bool):
    """
    Slår av/på knapper mens analyse kjører,
    og viser progressbar.
    """
    if busy:
        btn_upload.configure(state="disabled")
        btn_model.configure(state="disabled")
        btn_new.configure(state="disabled")
        btn_copy.configure(state="disabled")

        progress.grid()
        progress.start(10)

        pred_var.set("Kjører analyse…")
        summary_var.set("")
        clear_prob_rows()
        set_report("")
    else:
        progress.stop()
        progress.grid_remove()

        btn_upload.configure(state="normal")
        btn_model.configure(state="normal")
        btn_new.configure(state="normal")
        btn_copy.configure(state="normal")


# -------------------------
# Actions (knappe-funksjoner)
# -------------------------
def choose_model():
    """
    Lar brukeren velge modellfil manuelt.
    Resetter cache (_MODEL) slik at ny modell lastes.
    """
    global MODEL_PATH, _MODEL
    path = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        title="Velg modellfil",
        filetypes=(("Keras model", "*.keras *.h5 *.hdf5"), ("Alle filer", "*.*"))
    )
    if not path:
        return

    with _MODEL_LOCK:
        _MODEL = None

    MODEL_PATH = path
    model_display_var.set(shorten_path(path, 95))

def new_analysis():
    """Går tilbake til upload-siden for ny analyse."""
    show_page("upload")

def copy_report():
    """Kopierer rapporten til utklippstavlen."""
    text = report_text.get("1.0", "end").strip()
    if not text:
        messagebox.showinfo("Ingenting å kopiere", "Rapporten er tom.")
        return
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    messagebox.showinfo("Kopiert", "Rapport kopiert til utklippstavlen.")

def run_analysis_thread(img_path: str, patient_name: str, exam_date: str):
    """
    Kjøres i en bakgrunnstråd (thread) så GUI ikke fryser.
    Når vi har resultatet, oppdateres GUI med root.after().
    """
    try:
        main_burden, probs_pct = run_analysis(img_path, MODEL_PATH)
        report = format_report(patient_name, exam_date, main_burden, probs_pct)

        items = sorted(probs_pct.items(), key=lambda x: x[1], reverse=True)
        top_label, top_p = items[0] if items else (main_burden, 0.0)

        def apply():
            pred_var.set(f"Overall impression: {main_burden}  •  Top match: {top_label} ({top_p:.1f}%)")
            summary_var.set(make_plain_summary(main_burden, probs_pct))
            render_probs(probs_pct)
            set_report(report)
            notebook.select(tab_overview)  # start på overview tab

        root.after(0, apply)

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Feil", str(e)))
        root.after(0, lambda: pred_var.set("—"))
        root.after(0, lambda: summary_var.set(""))
        root.after(0, clear_prob_rows)
        root.after(0, lambda: set_report(""))
    finally:
        root.after(0, lambda: set_busy(False))

def upload_image():
    """
    1) Leser pasientinfo fra input
    2) Velger bilde fra fil
    3) Bytter til result-side
    4) Starter analyse i thread
    """
    patient_name = entry_value(patient_entry)
    exam_date_raw = entry_value(date_entry)
    exam_date = normalize_date(exam_date_raw)

    if not patient_name:
        messagebox.showerror("Mangler info", "Skriv inn fullt navn på pasienten.")
        return
    if not exam_date:
        messagebox.showerror("Mangler info", "Skriv inn undersøkelsesdato.")
        return

    filename = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        title="Velg MR-bilde",
        filetypes=(("Bilder", "*.png *.jpg *.jpeg *.webp *.gif"), ("Alle filer", "*.*"))
    )
    if not filename:
        file_display_var.set("Ingen fil valgt")
        return

    # Kopier filstien til clipboard (praktisk)
    root.clipboard_clear()
    root.clipboard_append(filename)
    root.update()

    file_display_var.set(f"Valgt: {shorten_path(filename, 95)}  (path kopiert)")

    # Bytt til resultatside og start analyse
    show_page("result")
    set_busy(True)

    threading.Thread(
        target=run_analysis_thread,
        args=(filename, patient_name, exam_date),
        daemon=True
    ).start()


# -------------------------
# Buttons (knyttes til actions)
# -------------------------
btn_model = ttk.Button(btn_row_upload, text="Velg modell", style="Accent.TButton", command=choose_model)
btn_model.grid(row=0, column=0, padx=(0, 12))

btn_upload = ttk.Button(btn_row_upload, text="Last opp bilde", style="Accent.TButton", command=upload_image)
btn_upload.grid(row=0, column=1)

btn_new = ttk.Button(nav_row, text="New analysis", command=new_analysis)
btn_new.grid(row=0, column=0, padx=(0, 10))

btn_copy = ttk.Button(nav_row, text="Copy report", command=copy_report)
btn_copy.grid(row=0, column=1)

# -------------------------
# Startup warning hvis translate_prediction mangler
# -------------------------
if translate_prediction is None:
    messagebox.showwarning(
        "translated_data.py mangler",
        "Fant ikke translate_prediction fra translated_data.py.\n\n"
        "Legg translated_data.py i samme mappe.\n"
        "Appen kan åpnes, men analyse vil feile."
    )

# Start på upload page
show_page("upload")
root.mainloop()
