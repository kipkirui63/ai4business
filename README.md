# FreightBot AI

FreightBot AI is a prototype logistics customer-support chatbot built for the BUS4005 Assessment 2 project. It demonstrates how AI can be used to automate common freight-service enquiries such as shipment tracking, freight quote requests, pickup booking, delivery-delay support, and escalation guidance.

The project combines AI-based intent classification with rule-based business workflow logic. A trained TensorFlow/Keras model identifies what the user is asking, and the application then responds using structured follow-up prompts and simulated freight-support data.

## Features

- Shipment tracking with sample tracking numbers
- Freight quote estimation
- Pickup booking flow
- Delivery-delay support
- Escalation guidance for human support
- Desktop GUI built with Tkinter
- Intent-classification model trained from freight-specific examples

## Tech Stack

- Python
- TensorFlow / Keras
- NLTK
- NumPy
- Tkinter
- JSON
- Pickle

## Project Files

- [chatgui.py](/home/sir-sang/Documents/Chatbot-using-Python/chatgui.py): chatbot GUI and runtime conversation logic
- [train_chatbot.py](/home/sir-sang/Documents/Chatbot-using-Python/train_chatbot.py): model training script
- [intents.json](/home/sir-sang/Documents/Chatbot-using-Python/intents.json): supported freight-support intents and training phrases
- `chatbot_model.keras`: trained model used at runtime
- `words.pkl` and `classes.pkl`: saved vocabulary and labels
- [materials](/home/sir-sang/Documents/Chatbot-using-Python/materials): assignment report, flow, and reflection materials

## Setup

### Option 1: Python virtual environment

```bash
cd /home/sir-sang/Documents/Chatbot-using-Python
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow nltk numpy
```

### Option 2: Conda environment

Use the included `environment.yml`:

```bash
cd /home/sir-sang/Documents/Chatbot-using-Python
conda env create -f environment.yml
conda activate freightbot-ai
```

## Train the Model

If you update `intents.json`, retrain the chatbot before running the GUI:

```bash
python3 train_chatbot.py
```

This will regenerate:

- `chatbot_model.keras`
- `words.pkl`
- `classes.pkl`

## Run the Chatbot

```bash
python3 chatgui.py
```

The GUI opens a logistics support console where you can test the prototype.

## Sample Prompts

Try these in the chatbot:

- `track my shipment`
- `TRK12345`
- `i need a freight quote`
- `120 kg 2 pallets from Dandenong to Geelong`
- `book a pickup`
- `Pick up from Campbellfield tomorrow for Alex, 3 cartons`
- `my shipment is delayed`
- `TRK67890`

## How AI Is Used

AI is used in the intent-classification layer:

- `train_chatbot.py` trains a TensorFlow/Keras model on the examples in `intents.json`
- `chatgui.py` loads that model and predicts the likely intent of each user message

Rule-based logic is then used to:

- manage follow-up conversation state
- look up sample shipment data
- estimate quote values
- return booking references and escalation guidance

## Limitations

- Uses simulated freight data rather than live business systems
- Does not authenticate real users
- Supports common support scenarios only
- Quote outputs are prototype estimates, not production prices

## Assignment Context

This repository was adapted into a freight/logistics use case for:

- `BUS4005: AI for Business`
- `Assessment 2: AI conversational agent solution design – Development and demonstration`

Supporting assignment deliverables are stored in the `materials/` folder.
# ai4business
# ai4business
