# FreightBot AI

FastAPI-based logistics chatbot for:

- shipment tracking
- freight quotes
- pickup booking
- delay support
- delivery changes

## Files

- `main.py` - FastAPI app
- `chatbot_core.py` - chatbot logic
- `train_chatbot.py` - model training script
- `intents.json` - intents and responses
- `templates/` - HTML UI
- `static/` - CSS and JavaScript
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment config

## Run Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model

Run this before the first start, or any time you change `intents.json`.

```bash
python3 train_chatbot.py
```

This creates:

- `chatbot_model.keras`
- `words.pkl`
- `classes.pkl`

### 4. Start the app

```bash
uvicorn main:app --reload
```

### 5. Open in browser

Go to:

```text
http://127.0.0.1:8000
```

## API Endpoints

- `GET /` - web UI
- `GET /health` - health check
- `POST /chat` - send a message
- `POST /reset` - clear chat session

Example `POST /chat` body:

```json
{
  "message": "Track my shipment",
  "session_id": null
}
```

## Sample Messages

- `track my shipment`
- `TRK12345`
- `i need a freight quote`
- `120 kg 2 pallets from Dandenong to Geelong`
- `book a pickup`
- `my shipment is delayed`

## Deploy to Render

Create a `Web Service` with:

- `Language`: `Python 3`
- `Branch`: `main`
- `Root Directory`: leave blank
- `Build Command`: `pip install -r requirements.txt && python train_chatbot.py`
- `Start Command`: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- `Health Check Path`: `/health`
- `Environment Variable`: `PYTHON_VERSION=3.11.11`

You can also use the included `render.yaml`.

## Notes

- If `chatbot_model.keras`, `words.pkl`, or `classes.pkl` are missing, the app will not start.
- The current app uses demo freight data, not live business systems.
