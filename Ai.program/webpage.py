import tkinter as tk
from tkinter import ttk, filedialog
import os

# Window
root = tk.Tk()
root.title("Min app")
root.geometry("800x600")
root.configure(bg="#f3f4f6")

# Styling
style = ttk.Style()
style.theme_use("clam")

style.configure("Header.TFrame", background="#111827")
style.configure("Header.TLabel", background="#111827", foreground="white", font=("Segoe UI", 16, "bold"))

style.configure("Card.TFrame", background="white", relief="solid", borderwidth=1)
style.configure("Card.TLabel", background="white", foreground="#111827", font=("Segoe UI", 11))

style.configure(
    "Accent.TButton",
    font=("Segoe UI", 12, "bold"),
    padding=(18, 10),
    foreground="white"
)

# (valgfritt) gjør at knappen får blå bakgrunn i flere temaer
style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("!disabled", "#2563eb")])

# Header
header = ttk.Frame(root, style="Header.TFrame", padding=(16, 12))
header.grid(row=0, column=0, sticky="ew")
root.grid_columnconfigure(0, weight=1)

site_title = ttk.Label(header, text="VitalScan", style="Header.TLabel")
site_title.grid(row=0, column=0, sticky="w")

# Main
main = ttk.Frame(root, padding=24)
main.grid(row=1, column=0, sticky="nsew")
root.grid_rowconfigure(1, weight=1)

card = ttk.Frame(main, style="Card.TFrame", padding=24)
card.place(relx=0.5, rely=0.5, anchor="center")

title = ttk.Label(card, text="Velkommen!", style="Card.TLabel", font=("Segoe UI", 14, "bold"))
title.grid(row=0, column=0, pady=(0, 10))

subtitle = ttk.Label(card, text="Trykk på knappen for å laste opp en fil.", style="Card.TLabel")
subtitle.grid(row=1, column=0, pady=(0, 16))

# Label som viser valgt fil (må finnes!)
label_file_explorer = ttk.Label(card, text="Ingen fil valgt", style="Card.TLabel", wraplength=520)
label_file_explorer.grid(row=3, column=0, pady=(16, 0))

def klikk():
    filename = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        title="Velg en fil",
        filetypes=(("Bilder", "*.png *.jpg *.jpeg *.webp *.gif"), ("Alle filer", "*.*"))

    )

    if filename:
        # Kopier filepath automatisk
        root.clipboard_clear()
        root.clipboard_append(filename)
        root.update()  # sikrer at clipboard oppdateres

        label_file_explorer.configure(text=f"Kopiert path:\n{filename}")
    else:
        label_file_explorer.configure(text="Ingen fil valgt")

button = ttk.Button(card, text="Last opp fil", style="Accent.TButton", command=klikk)
button.grid(row=2, column=0)

root.mainloop()
