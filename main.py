from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from chatbot_core import FreightBotEngine


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="FreightBot AI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

engine = FreightBotEngine()
sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def get_session_state(session_id):
    if session_id not in sessions:
        sessions[session_id] = {"context": ""}
    return sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = payload.session_id or str(uuid4())
    state = get_session_state(session_id)
    reply = engine.respond(message, state)
    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/reset")
async def reset_chat(payload: ChatRequest):
    session_id = payload.session_id or str(uuid4())
    sessions[session_id] = {"context": ""}
    return {"session_id": session_id, "status": "reset"}
