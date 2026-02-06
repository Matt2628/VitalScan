import tkinter as tk 
from tkinter import ttk

#Window
root = tk.Tk()
root.title("Min app")
root.geometry("800x600")
root.configure(bg="#f3f4f6")

#Styling
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
    background="#2563eb",
    foreground="white"
)

#header
header = ttk.Frame(root, style="Header.TFrame", padding=(16, 12))
header.grid(row=0, column=0, sticky="ew")
root.grid_columnconfigure(0, weight=1)

#Name
site_title = ttk.Label(header, text="Vital", style="Header.TLabel")
site_title.grid(row=0, column=1, sticky="w", padx=(12, 0))
header.grid_columnconfigure(2, weight=1)  


main = ttk.Frame(root, padding=24)
main.grid(row=1, column=0, sticky="nsew")
root.grid_rowconfigure(1, weight=1)


card = ttk.Frame(main, style="Card.TFrame", padding=24)
card.place(relx=0.5, rely=0.5, anchor="center")  


#Button box
title = ttk.Label(card, text="Velkommen!", style="Card.TLabel", font=("Segoe UI", 14, "bold"))
title.grid(row=0, column=0, pady=(0, 10))

subtitle = ttk.Label(card, text="Trykk på knappen for å laste opp en fil.", style="Card.TLabel")
subtitle.grid(row=1, column=0, pady=(0, 16))

#Button
def klikk():
    print("Du trykket på knappen!")

button = ttk.Button(card, text="Last opp fil", style="Accent.TButton", command=klikk)
button.grid(row=2, column=0)

#File explorer open



root.mainloop()
