import pickle
import json
import random
import re
import threading

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import END, NORMAL, DISABLED

# ── Model setup (unchanged) ─────────────────────────────────────────────────

stemmer = PorterStemmer()
model       = load_model('chatbot_model.keras')
intents     = json.loads(open('intents.json').read())
words       = pickle.load(open('words.pkl', 'rb'))
classes     = pickle.load(open('classes.pkl', 'rb'))

shipment_statuses = {
    "TRK12345": {"status": "In transit",        "location": "Melbourne linehaul hub",  "eta": "Tomorrow by 5:00 PM"},
    "TRK67890": {"status": "Delayed",            "location": "Sydney depot",            "eta": "1 business day late due to weather disruption"},
    "TRK24680": {"status": "Out for delivery",   "location": "Brisbane local van",      "eta": "Today by 3:00 PM"},
}

conversation_state = {"context": ""}


def clean_up_sentence(sentence):
    sentence_words = nltk.wordpunct_tokenize(sentence)
    return [stemmer.stem(w.lower()) for w in sentence_words]

def bow(sentence, words):
    sw = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sw:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_intent_definition(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent
    return None

def getResponse(ints, intents_json):
    if not ints:
        return "I didn't understand that. Try asking something else."
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def extract_tracking_number(msg):
    match = re.search(r"\bTRK\d{5}\b", msg.upper())
    return match.group(0) if match else None

def parse_quote_details(msg):
    text = msg.lower()
    wm = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?)", text)
    pm = re.search(r"(\d+)\s*(pallet|pallets|carton|cartons)", text)
    if not wm:
        return None
    weight = float(wm.group(1))
    units  = int(pm.group(1)) if pm else 1
    return round(45 + weight * 0.35 + units * 12, 2)

def handle_tracking_context(msg):
    tn = extract_tracking_number(msg)
    if not tn:
        return "Please provide a valid tracking number — format: TRK12345."
    s = shipment_statuses.get(tn)
    conversation_state["context"] = ""
    if s:
        return f"Shipment {tn}: {s['status']}. Location: {s['location']}. ETA: {s['eta']}."
    return f"Shipment {tn} not found. Demo numbers: TRK12345, TRK67890, TRK24680."

def handle_quote_context(msg):
    est = parse_quote_details(msg)
    conversation_state["context"] = ""
    if est is None:
        return "I need the shipment weight — e.g. '120 kg, 2 pallets, Dandenong to Geelong'."
    return f"Estimated freight: AUD {est}. Excludes fuel levy and special handling."

def handle_pickup_context(msg):
    conversation_state["context"] = ""
    if len(msg.split()) < 6:
        return "Please include suburb, collection date, contact name, and freight type."
    return "Pickup logged. Reference: PU-2026-001. Dispatcher will confirm the time window."

def handle_context(msg):
    ctx = conversation_state["context"]
    if ctx == "awaiting_tracking_number": return handle_tracking_context(msg)
    if ctx == "awaiting_quote_details":   return handle_quote_context(msg)
    if ctx == "awaiting_pickup_details":  return handle_pickup_context(msg)
    return None

def chatbot_response(msg):
    reply = handle_context(msg)
    if reply:
        return reply
    ints = predict_class(msg, model)
    if not ints:
        fb = get_intent_definition("fallback")
        return random.choice(fb["responses"])
    top = ints[0]["intent"]
    defn = get_intent_definition(top)
    conversation_state["context"] = defn["context"][0] if defn and defn.get("context") else ""
    return getResponse(ints, intents)


# ── Design tokens ────────────────────────────────────────────────────────────

C = {
    "bg":           "#0a1628",   # deep navy — window background
    "panel":        "#0f1f38",   # slightly lighter — chat area
    "header":       "#0d1a30",   # header bar
    "border":       "#1e3a5a",   # subtle dividers
    "teal":         "#0d9488",   # primary accent
    "teal_dark":    "#0a7a70",   # pressed accent
    "teal_dim":     "#0d948822", # glow / highlight bg
    "bot_bubble":   "#112240",   # bot message fill
    "user_bubble":  "#0d7a70",   # user message fill
    "bot_name":     "#14b8a6",   # "FreightBot" label
    "user_name":    "#5eead4",   # "You" label
    "text":         "#e2eaf4",   # primary text
    "muted":        "#64748b",   # timestamps / hints
    "input_bg":     "#0b1829",   # input field bg
    "tag_bg":       "#0f2540",   # quick-prompt pill bg
    "tag_border":   "#1e3a5a",   # quick-prompt pill border
    "tag_hover":    "#0d9488",   # quick-prompt hover
    "dot":          "#10b981",   # online status dot
}

FONT_BODY   = ("Segoe UI", 11)
FONT_BOLD   = ("Segoe UI", 11, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_TITLE  = ("Segoe UI", 15, "bold")
FONT_SUB    = ("Segoe UI", 10)

QUICK_PROMPTS = [
    ("Track shipment",  "Track my shipment TRK12345"),
    ("Get a quote",     "I need a freight quote for 120 kg"),
    ("Report delay",    "My delivery is delayed and I need help"),
    ("Book pickup",     "I would like to book a pickup"),
]




# ── Rounded rectangle helper ─────────────────────────────────────────────────

def rounded_rect(canvas, x1, y1, x2, y2, r=14, **kw):
    """Draw a filled rounded rectangle on a Canvas."""
    pts = [
        x1+r, y1,   x2-r, y1,
        x2,   y1,   x2,   y1+r,
        x2,   y2-r, x2,   y2,
        x2-r, y2,   x1+r, y2,
        x1,   y2,   x1,   y2-r,
        x1,   y1+r, x1,   y1,
        x1+r, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


# ── Main GUI class ────────────────────────────────────────────────────────────

class FreightBotApp:

    PAD_H     = 28      # horizontal padding inside chat area
    BUBBLE_R  = 14      # bubble corner radius
    BOT_COL   = 520     # max bubble width for bot
    USER_COL  = 480     # max bubble width for user

    def __init__(self):
        self._build_window()
        self._build_header()
        self._build_quick_prompts()
        self._build_chat()
        self._build_composer()
        self._post_bot("Hello! I'm FreightBot — your AI logistics assistant.\n"
                       "I can help with tracking, quotes, pickup bookings, delays, and damaged freight.")
        self.entry.focus_set()
        self.root.mainloop()

    # ── Window ───────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("FreightBot — Logistics AI")
        self.root.geometry("820x720")
        self.root.minsize(720, 600)
        self.root.configure(bg=C["bg"])
        # Smooth resize
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.outer = tk.Frame(self.root, bg=C["bg"])
        self.outer.pack(fill="both", expand=True)
        self.outer.grid_rowconfigure(2, weight=1)
        self.outer.grid_columnconfigure(0, weight=1)

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.outer, bg=C["header"], height=72)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        # Avatar canvas
        av = tk.Canvas(hdr, width=44, height=44, bg=C["header"], highlightthickness=0)
        av.grid(row=0, column=0, padx=(22, 14), pady=14, rowspan=2)
        av.create_oval(2, 2, 42, 42, fill=C["teal"], outline=C["teal_dark"], width=2)
        av.create_text(22, 22, text="🚚", font=("Segoe UI Emoji", 14))

        # Title
        name_row = tk.Frame(hdr, bg=C["header"])
        name_row.grid(row=0, column=1, sticky="sw", pady=(16, 0))

        tk.Label(name_row, text="FreightBot", font=FONT_TITLE,
                 bg=C["header"], fg=C["text"]).pack(side="left")

        # Online dot (Canvas circle)
        dot_c = tk.Canvas(name_row, width=10, height=10,
                          bg=C["header"], highlightthickness=0)
        dot_c.pack(side="left", padx=(8, 0))
        dot_c.create_oval(1, 1, 9, 9, fill=C["dot"], outline="")
        self._dot_canvas = dot_c
        self._dot_pulse_val = 0
        self._animate_dot()

        tk.Label(hdr, text="AI logistics support  ·  tracking  ·  quotes  ·  escalation",
                 font=FONT_SUB, bg=C["header"], fg=C["muted"]).grid(row=1, column=1, sticky="nw", pady=(0, 12))

        # Separator
        sep = tk.Frame(self.outer, bg=C["border"], height=1)
        sep.grid(row=1, column=0, sticky="ew")

    def _animate_dot(self):
        """Pulse the green online dot."""
        self._dot_pulse_val = (self._dot_pulse_val + 1) % 20
        alpha = abs(self._dot_pulse_val - 10) / 10   # 0.0 → 1.0 → 0.0
        # interpolate between dim and bright green
        r = int(10  + alpha * 6)
        g = int(185 + alpha * 10)
        b = int(129 - alpha * 20)
        color = f"#{r:02x}{g:02x}{b:02x}"
        self._dot_canvas.delete("all")
        self._dot_canvas.create_oval(1, 1, 9, 9, fill=color, outline="")
        self.root.after(80, self._animate_dot)

    # ── Quick prompts ────────────────────────────────────────────────────────

    def _build_quick_prompts(self):
        row = tk.Frame(self.outer, bg=C["bg"], pady=10)
        row.grid(row=2, column=0, sticky="ew", padx=self.PAD_H)

        tk.Label(row, text="Quick prompts:", font=FONT_SMALL,
                 bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0, 10))

        for label, prompt in QUICK_PROMPTS:
            self._pill_button(row, label, prompt)

    def _pill_button(self, parent, label, prompt):
        btn = tk.Label(
            parent, text=label,
            font=FONT_SMALL,
            bg=C["tag_bg"], fg=C["muted"],
            padx=10, pady=4,
            relief="flat", cursor="hand2",
        )
        btn.pack(side="left", padx=4)
        btn.bind("<Button-1>",  lambda e, p=prompt: self._use_prompt(p))
        btn.bind("<Enter>",     lambda e, b=btn: b.config(bg=C["teal"], fg="white"))
        btn.bind("<Leave>",     lambda e, b=btn: b.config(bg=C["tag_bg"], fg=C["muted"]))

    def _use_prompt(self, prompt):
        self.entry.delete("1.0", END)
        self.entry.insert("1.0", prompt)
        self.entry.focus_set()

    # ── Chat canvas ──────────────────────────────────────────────────────────

    def _build_chat(self):
        sep = tk.Frame(self.outer, bg=C["border"], height=1)
        sep.grid(row=3, column=0, sticky="ew")

        wrapper = tk.Frame(self.outer, bg=C["panel"])
        wrapper.grid(row=4, column=0, sticky="nsew")
        self.outer.grid_rowconfigure(4, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)
        wrapper.grid_columnconfigure(0, weight=1)

        self.chat_text = tk.Text(
            wrapper,
            bg=C["panel"], fg=C["text"],
            relief="flat", bd=0,
            wrap="word",
            padx=self.PAD_H, pady=20,
            font=FONT_BODY,
            spacing1=6, spacing2=3, spacing3=14,
            cursor="arrow",
            state=DISABLED,
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(wrapper, orient="vertical",
                          command=self.chat_text.yview,
                          bg=C["panel"], troughcolor=C["panel"],
                          activebackground=C["border"], relief="flat", width=6)
        sb.grid(row=0, column=1, sticky="ns")
        self.chat_text["yscrollcommand"] = sb.set

        # Text tags
        self.chat_text.tag_configure("bot_name",
            foreground=C["bot_name"], font=("Segoe UI", 9, "bold"),
            spacing1=10, spacing3=3)
        self.chat_text.tag_configure("bot_msg",
            foreground=C["text"], font=FONT_BODY,
            background=C["bot_bubble"],
            lmargin1=self.PAD_H, lmargin2=self.PAD_H + 10,
            rmargin=140, spacing1=8, spacing3=8,
            relief="flat", borderwidth=12)
        self.chat_text.tag_configure("user_name",
            foreground=C["user_name"], font=("Segoe UI", 9, "bold"),
            justify="right", spacing1=10, spacing3=3)
        self.chat_text.tag_configure("user_msg",
            foreground="white", font=FONT_BODY,
            background=C["user_bubble"],
            lmargin1=180, lmargin2=180,
            rmargin=self.PAD_H, spacing1=8, spacing3=8,
            justify="right", relief="flat", borderwidth=12)
        self.chat_text.tag_configure("ts",
            foreground=C["muted"], font=FONT_MONO,
            spacing3=4)
        self.chat_text.tag_configure("typing",
            foreground=C["muted"], font=("Segoe UI", 10, "italic"),
            lmargin1=self.PAD_H, spacing1=6, spacing3=8)



    # ── Composer ─────────────────────────────────────────────────────────────

    def _build_composer(self):
        sep = tk.Frame(self.outer, bg=C["border"], height=1)
        sep.grid(row=5, column=0, sticky="ew")

        comp = tk.Frame(self.outer, bg=C["bg"], padx=self.PAD_H, pady=14)
        comp.grid(row=6, column=0, sticky="ew")
        comp.grid_columnconfigure(0, weight=1)

        # Input shell
        shell = tk.Frame(comp, bg=C["input_bg"],
                         highlightthickness=1, highlightbackground=C["border"])
        shell.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        shell.grid_columnconfigure(0, weight=1)

        self.entry = tk.Text(
            shell, height=3, bd=0,
            bg=C["input_bg"], fg=C["text"],
            insertbackground=C["teal"],
            wrap="word", font=FONT_BODY,
            padx=14, pady=12,
        )
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>",   self._on_enter)
        self.entry.bind("<FocusIn>",  lambda e: shell.config(highlightbackground=C["teal"]))
        self.entry.bind("<FocusOut>", lambda e: shell.config(highlightbackground=C["border"]))

        # Send button
        self.send_btn = tk.Button(
            comp, text="Send →",
            font=("Segoe UI", 11, "bold"),
            bg=C["teal"], fg="white",
            activebackground=C["teal_dark"], activeforeground="white",
            relief="flat", bd=0,
            padx=20, pady=18,
            cursor="hand2",
            command=self._send,
        )
        self.send_btn.grid(row=0, column=1, sticky="ns")

        tk.Label(comp,
                 text="Enter to send  ·  Shift+Enter for new line",
                 font=FONT_MONO, bg=C["bg"], fg=C["muted"]
                 ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    # ── Messaging ────────────────────────────────────────────────────────────

    def _timestamp(self):
        import datetime
        return datetime.datetime.now().strftime("%H:%M")

    def _post_bot(self, text):
        t = self.chat_text
        t.config(state=NORMAL)
        t.insert(END, f"FreightBot  ·  {self._timestamp()}\n", "bot_name")
        t.insert(END, text + "\n\n", "bot_msg")
        t.config(state=DISABLED)
        t.yview_moveto(1.0)

    def _post_user(self, text):
        t = self.chat_text
        t.config(state=NORMAL)
        t.insert(END, f"You  ·  {self._timestamp()}\n", "user_name")
        t.insert(END, text + "\n\n", "user_msg")
        t.config(state=DISABLED)
        t.yview_moveto(1.0)

    # ── Send logic ───────────────────────────────────────────────────────────

    def _on_enter(self, event):
        if event.state & 0x0001:   # Shift held → newline
            return None
        self._send()
        return "break"

    def _send(self):
        msg = self.entry.get("1.0", "end-1c").strip()
        self.entry.delete("1.0", END)
        if not msg:
            return

        self._post_user(msg)
        self.send_btn.config(state=DISABLED, bg=C["border"])

        # Run inference off the main thread so UI stays responsive
        def worker():
            reply = chatbot_response(msg)
            self.root.after(0, lambda: self._deliver(reply))

        threading.Thread(target=worker, daemon=True).start()

    def _deliver(self, reply):
        self._post_bot(reply)
        self.send_btn.config(state=NORMAL, bg=C["teal"])
        self.entry.focus_set()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FreightBotApp()