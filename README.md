# FreightBot AI

FreightBot AI is a prototype logistics customer-support chatbot that demonstrates how AI can automate common freight-service enquiries such as shipment tracking, freight quote requests, pickup booking, delivery-delay support, and escalation guidance.

The project combines AI-based intent classification with rule-based business workflow logic. A trained TensorFlow/Keras model identifies what the user is asking, and the application responds using structured follow-up prompts and simulated freight-support data.

## Overview

FreightBot was developed as a workflow-focused chatbot rather than a general conversational assistant. It is designed to support first-line logistics customer service by:

- identifying the user's request type
- collecting missing shipment or booking details
- returning a structured response
- escalating where automation should stop

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

## Project Structure

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
pip install -r requirements.txt
```

### Option 2: Conda environment

Use the included `environment.yml`:

```bash
cd /home/sir-sang/Documents/Chatbot-using-Python
conda env create -f environment.yml
conda activate freightbot-ai
```

### Install with pip only

```bash
pip install -r requirements.txt
```

If `tkinter` is missing on Linux, install it separately using your system package manager, for example:

```bash
sudo apt install python3-tk
```

## Train the Model

This repository does not store generated model artifacts in Git. Train the chatbot before first run, and retrain any time you update `intents.json`:

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

If `chatbot_model.keras` is missing, the GUI will not start until training has been run successfully.

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

This means the system uses AI for language understanding and uses deterministic workflow logic for business responses.

## Limitations

- Uses simulated freight data rather than live business systems
- Does not authenticate real users
- Supports common support scenarios only
- Quote outputs are prototype estimates, not production prices

## Repository Policy

Generated runtime artifacts are intentionally ignored:

- `chatbot_model.keras`
- `chatbot_model.h5`
- `words.pkl`
- `classes.pkl`

This keeps the repository focused on source code, configuration, and assignment materials while allowing the model to be rebuilt locally.

## Assignment Context

This repository was adapted into a freight/logistics use case for:

- `BUS4005: AI for Business`
- `Assessment 2: AI conversational agent solution design – Development and demonstration`

Supporting assignment deliverables are stored in the `materials/` folder.
# ai4business
# ai4business
