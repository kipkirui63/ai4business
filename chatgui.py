#https://github.com/tridibsamanta/Chatbot-using-Python

import pickle
import json
import random
import re

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

stemmer = PorterStemmer()

model = load_model('chatbot_model.keras')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

shipment_statuses = {
    "TRK12345": {
        "status": "In transit",
        "location": "Melbourne linehaul hub",
        "eta": "Tomorrow by 5:00 PM"
    },
    "TRK67890": {
        "status": "Delayed",
        "location": "Sydney depot",
        "eta": "1 business day late due to weather disruption"
    },
    "TRK24680": {
        "status": "Out for delivery",
        "location": "Brisbane local van",
        "eta": "Today by 3:00 PM"
    }
}

conversation_state = {
    "context": "",
}


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.wordpunct_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_intent_definition(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent
    return None

def getResponse(ints, intents_json):
    if not ints:
        return "I didn't understand that. Try asking something else."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def extract_tracking_number(msg):
    match = re.search(r"\bTRK\d{5}\b", msg.upper())
    if match:
        return match.group(0)
    return None

def parse_quote_details(msg):
    text = msg.lower()
    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?)", text)
    pallet_match = re.search(r"(\d+)\s*(pallet|pallets|carton|cartons)", text)

    weight = float(weight_match.group(1)) if weight_match else None
    units = int(pallet_match.group(1)) if pallet_match else 1

    if weight is None:
        return None

    estimated_price = round(45 + (weight * 0.35) + (units * 12), 2)
    return estimated_price

def handle_tracking_context(msg):
    tracking_number = extract_tracking_number(msg)
    if not tracking_number:
        return "Please provide a valid tracking number in the format TRK12345."

    shipment = shipment_statuses.get(tracking_number)
    if shipment:
        conversation_state["context"] = ""
        return (
            f"Shipment {tracking_number}: {shipment['status']}. "
            f"Current location: {shipment['location']}. ETA: {shipment['eta']}."
        )

    conversation_state["context"] = ""
    return (
        f"I could not find shipment {tracking_number} in the prototype dataset. "
        "For the demo, try TRK12345, TRK67890, or TRK24680."
    )

def handle_quote_context(msg):
    estimate = parse_quote_details(msg)
    if estimate is None:
        return (
            "I need more quote details. Please include at least the shipment weight, "
            "for example: 120 kg, 2 pallets, from Dandenong to Geelong."
        )

    conversation_state["context"] = ""
    return (
        f"Estimated freight charge: AUD {estimate}. "
        "This prototype quote excludes fuel levy, tail-lift fees, and special handling."
    )

def handle_pickup_context(msg):
    if len(msg.split()) < 6:
        return (
            "Please include the pickup suburb, collection date, contact name, and freight type "
            "so I can confirm the booking."
        )

    conversation_state["context"] = ""
    return (
        "Pickup request logged for the prototype workflow. "
        "Reference: PU-2026-001. A dispatcher would normally confirm the time window next."
    )

def handle_context(msg):
    context = conversation_state["context"]
    if context == "awaiting_tracking_number":
        return handle_tracking_context(msg)
    if context == "awaiting_quote_details":
        return handle_quote_context(msg)
    if context == "awaiting_pickup_details":
        return handle_pickup_context(msg)
    return None

def chatbot_response(msg):
    contextual_reply = handle_context(msg)
    if contextual_reply:
        return contextual_reply

    ints = predict_class(msg, model)
    if not ints:
        fallback = get_intent_definition("fallback")
        return random.choice(fallback["responses"])

    top_intent = ints[0]["intent"]
    intent_definition = get_intent_definition(top_intent)
    if intent_definition and intent_definition["context"]:
        conversation_state["context"] = intent_definition["context"][0]
    else:
        conversation_state["context"] = ""

    res = getResponse(ints, intents)
    return res


# Creating GUI with tkinter
import tkinter as tk
from tkinter import END, NORMAL, DISABLED, FALSE


COLORS = {
    "bg": "#f3f7fb",
    "panel": "#ffffff",
    "navy": "#0f2942",
    "slate": "#567086",
    "line": "#d7e1ea",
    "accent": "#0e7490",
    "accent_dark": "#155e75",
    "bot_bg": "#e8f3fb",
    "user_bg": "#d9f6ec",
    "input_bg": "#f8fbfd",
}


def append_message(sender, message):
    ChatLog.config(state=NORMAL)

    if sender == "Bot":
        ChatLog.insert(END, "FreightBot\n", "bot_name")
        ChatLog.insert(END, message + "\n\n", "bot_message")
    else:
        ChatLog.insert(END, "You\n", "user_name")
        ChatLog.insert(END, message + "\n\n", "user_message")

    ChatLog.config(state=DISABLED)
    ChatLog.yview_moveto(1.0)


def send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("1.0", END)

    if msg:
        append_message("You", msg)
        res = chatbot_response(msg)
        append_message("Bot", res)


def use_prompt(prompt):
    EntryBox.delete("1.0", END)
    EntryBox.insert("1.0", prompt)
    EntryBox.focus_set()


def on_enter(event):
    if event.state & 0x0001:
        return None
    send()
    return "break"


def build_gui():
    global base, ChatLog, EntryBox

    base = tk.Tk()
    base.title("FreightBot - Logistics Support")
    base.geometry("760x700")
    base.minsize(720, 660)
    base.configure(bg=COLORS["bg"])

    header = tk.Frame(base, bg=COLORS["navy"], height=96)
    header.pack(fill="x")
    header.pack_propagate(False)

    title_frame = tk.Frame(header, bg=COLORS["navy"])
    title_frame.pack(side="left", padx=28, pady=18)

    tk.Label(
        title_frame,
        text="FreightBot",
        font=("Georgia", 24, "bold"),
        bg=COLORS["navy"],
        fg="white"
    ).pack(anchor="w")
    tk.Label(
        title_frame,
        text="AI logistics support for tracking, quotes, pickups, delays, and delivery changes",
        font=("Helvetica", 11),
        bg=COLORS["navy"],
        fg="#c4d4e3"
    ).pack(anchor="w", pady=(4, 0))

    status_card = tk.Frame(header, bg="#123654", padx=18, pady=12)
    status_card.pack(side="right", padx=28, pady=18)
    tk.Label(
        status_card,
        text="Prototype Mode",
        font=("Helvetica", 10, "bold"),
        bg="#123654",
        fg="#90e0ef"
    ).pack(anchor="e")
    tk.Label(
        status_card,
        text="Using simulated freight records",
        font=("Helvetica", 10),
        bg="#123654",
        fg="white"
    ).pack(anchor="e")

    body = tk.Frame(base, bg=COLORS["bg"])
    body.pack(fill="both", expand=True, padx=22, pady=22)
    body.grid_columnconfigure(0, weight=1)
    body.grid_rowconfigure(0, weight=1)

    main_panel = tk.Frame(body, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["line"])
    main_panel.grid(row=0, column=0, sticky="nsew")
    main_panel.grid_rowconfigure(1, weight=1)
    main_panel.grid_columnconfigure(0, weight=1)

    summary = tk.Frame(main_panel, bg=COLORS["panel"], padx=20, pady=16)
    summary.grid(row=0, column=0, sticky="ew")

    tk.Label(
        summary,
        text="Customer Support",
        font=("Helvetica", 16, "bold"),
        bg=COLORS["panel"],
        fg=COLORS["navy"]
    ).pack(anchor="w")
    tk.Label(
        summary,
        text="FreightBot handles common enquiries first, then escalates to dispatch or customer support when needed.",
        font=("Helvetica", 10),
        bg=COLORS["panel"],
        fg=COLORS["slate"],
        wraplength=520,
        justify="left"
    ).pack(anchor="w", pady=(5, 0))

    chat_wrap = tk.Frame(main_panel, bg=COLORS["panel"], padx=20, pady=12)
    chat_wrap.grid(row=1, column=0, sticky="nsew")
    chat_wrap.grid_rowconfigure(0, weight=1)
    chat_wrap.grid_columnconfigure(0, weight=1)

    ChatLog = tk.Text(
        chat_wrap,
        bg="#fcfeff",
        fg=COLORS["navy"],
        relief="flat",
        wrap="word",
        padx=18,
        pady=18,
        font=("Helvetica", 11),
        spacing1=4,
        spacing2=2,
        spacing3=10
    )
    ChatLog.grid(row=0, column=0, sticky="nsew")
    ChatLog.config(state=DISABLED)

    scrollbar = tk.Scrollbar(chat_wrap, command=ChatLog.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    ChatLog["yscrollcommand"] = scrollbar.set

    ChatLog.tag_configure("bot_name", foreground=COLORS["accent_dark"], font=("Helvetica", 10, "bold"))
    ChatLog.tag_configure(
        "bot_message",
        background=COLORS["bot_bg"],
        foreground=COLORS["navy"],
        lmargin1=16,
        lmargin2=16,
        rmargin=90,
        font=("Helvetica", 11)
    )
    ChatLog.tag_configure("user_name", foreground="#146c43", font=("Helvetica", 10, "bold"), justify="right")
    ChatLog.tag_configure(
        "user_message",
        background=COLORS["user_bg"],
        foreground="#0d3b2a",
        lmargin1=140,
        lmargin2=140,
        rmargin=16,
        justify="right",
        font=("Helvetica", 11)
    )

    composer = tk.Frame(main_panel, bg=COLORS["panel"], padx=20, pady=18)
    composer.grid(row=2, column=0, sticky="ew")
    composer.grid_columnconfigure(0, weight=1)

    input_shell = tk.Frame(
        composer,
        bg=COLORS["input_bg"],
        highlightthickness=1,
        highlightbackground=COLORS["line"],
        padx=14,
        pady=14
    )
    input_shell.grid(row=0, column=0, sticky="ew", padx=(0, 12))
    input_shell.grid_columnconfigure(0, weight=1)

    EntryBox = tk.Text(
        input_shell,
        height=3,
        bd=0,
        bg=COLORS["input_bg"],
        fg=COLORS["navy"],
        insertbackground=COLORS["navy"],
        wrap="word",
        font=("Helvetica", 11)
    )
    EntryBox.grid(row=0, column=0, sticky="ew")
    EntryBox.bind("<Return>", on_enter)

    send_button = tk.Button(
        composer,
        text="Send",
        font=("Helvetica", 11, "bold"),
        bg=COLORS["accent"],
        fg="white",
        activebackground=COLORS["accent_dark"],
        activeforeground="white",
        bd=0,
        padx=22,
        pady=16,
        command=send
    )
    send_button.grid(row=0, column=1, sticky="ns")

    helper = tk.Label(
        composer,
        text="Press Enter to send. Use Shift+Enter for a new line.",
        font=("Helvetica", 9),
        bg=COLORS["panel"],
        fg=COLORS["slate"]
    )
    helper.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

    append_message(
        "Bot",
        "Hello, this is FreightBot. I can help with tracking, quotes, pickup booking, delays, and damaged freight."
    )

    EntryBox.focus_set()
    base.mainloop()


if __name__ == "__main__":
    build_gui()
