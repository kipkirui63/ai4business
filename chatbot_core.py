import json
import pickle
import random
import re
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
CONFIDENCE_THRESHOLD = 0.75

shipment_statuses = {
    "TRK12345": {
        "status": "In transit",
        "location": "Melbourne linehaul hub",
        "eta": "Tomorrow by 5:00 PM",
    },
    "TRK67890": {
        "status": "Delayed",
        "location": "Sydney depot",
        "eta": "1 business day late due to weather disruption",
    },
    "TRK24680": {
        "status": "Out for delivery",
        "location": "Brisbane local van",
        "eta": "Today by 3:00 PM",
    },
}


class FreightBotEngine:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.model = load_model(BASE_DIR / "chatbot_model.keras")
        self.intents = json.loads((BASE_DIR / "intents.json").read_text())
        with open(BASE_DIR / "words.pkl", "rb") as words_file:
            self.words = pickle.load(words_file)
        with open(BASE_DIR / "classes.pkl", "rb") as classes_file:
            self.classes = pickle.load(classes_file)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.wordpunct_tokenize(sentence)
        return [self.stemmer.stem(word.lower()) for word in sentence_words]

    def bow(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for sentence_word in sentence_words:
            for index, word in enumerate(self.words):
                if word == sentence_word:
                    bag[index] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        prediction = self.model.predict(np.array([self.bow(sentence)]), verbose=0)[0]
        results = [
            [index, score]
            for index, score in enumerate(prediction)
            if score > CONFIDENCE_THRESHOLD
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return [
            {"intent": self.classes[result[0]], "probability": str(result[1])}
            for result in results
        ]

    def get_intent_definition(self, tag):
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return intent
        return None

    def get_response(self, intents_found):
        if not intents_found:
            fallback = self.get_intent_definition("fallback")
            return random.choice(fallback["responses"])

        tag = intents_found[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I did not understand that. Please rephrase your request."

    @staticmethod
    def extract_tracking_number(message):
        match = re.search(r"\bTRK\d{5}\b", message.upper())
        return match.group(0) if match else None

    @staticmethod
    def parse_quote_details(message):
        text = message.lower()
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?)", text)
        pallet_match = re.search(r"(\d+)\s*(pallet|pallets|carton|cartons)", text)

        if not weight_match:
            return None

        weight = float(weight_match.group(1))
        units = int(pallet_match.group(1)) if pallet_match else 1
        return round(45 + (weight * 0.35) + (units * 12), 2)

    def handle_tracking_context(self, message, state):
        tracking_number = self.extract_tracking_number(message)
        if not tracking_number:
            return "Please provide a valid tracking number in the format TRK12345."

        shipment = shipment_statuses.get(tracking_number)
        state["context"] = ""

        if shipment:
            return (
                f"Shipment {tracking_number}: {shipment['status']}. "
                f"Current location: {shipment['location']}. ETA: {shipment['eta']}."
            )

        return (
            f"I could not find shipment {tracking_number} in the demo dataset. "
            "Try TRK12345, TRK67890, or TRK24680."
        )

    def handle_quote_context(self, message, state):
        estimate = self.parse_quote_details(message)
        state["context"] = ""

        if estimate is None:
            return (
                "I need more quote details. Include at least shipment weight, "
                "for example: 120 kg, 2 pallets, from Dandenong to Geelong."
            )

        return (
            f"Estimated freight charge: AUD {estimate}. "
            "This estimate excludes fuel levy, tail-lift fees, and special handling."
        )

    @staticmethod
    def handle_pickup_context(message, state):
        state["context"] = ""
        if len(message.split()) < 6:
            return (
                "Please include the pickup suburb, collection date, contact name, and freight type "
                "so I can confirm the booking."
            )

        return (
            "Pickup request logged for the demo workflow. "
            "Reference: PU-2026-001. A dispatcher would normally confirm the time window next."
        )

    def handle_context(self, message, state):
        context = state.get("context", "")
        if context == "awaiting_tracking_number":
            return self.handle_tracking_context(message, state)
        if context == "awaiting_quote_details":
            return self.handle_quote_context(message, state)
        if context == "awaiting_pickup_details":
            return self.handle_pickup_context(message, state)
        return None

    def respond(self, message, state):
        contextual_reply = self.handle_context(message, state)
        if contextual_reply:
            return contextual_reply

        intents_found = self.predict_class(message)
        if not intents_found:
            fallback = self.get_intent_definition("fallback")
            return random.choice(fallback["responses"])

        top_intent = intents_found[0]["intent"]
        intent_definition = self.get_intent_definition(top_intent)
        if intent_definition and intent_definition.get("context"):
            state["context"] = intent_definition["context"][0]
        else:
            state["context"] = ""

        return self.get_response(intents_found)
