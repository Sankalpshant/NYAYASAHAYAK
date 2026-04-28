from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq  # type: ignore[import]
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are "NyayaSahayak", an AI-powered legal assistant designed to help Indian citizens—especially low-income and legally unaware individuals—understand their legal rights and options.
Your purpose is to simplify Indian laws and provide clear, practical, and actionable guidance in a user-friendly way.
CORE OBJECTIVES:
1. Explain Indian laws in simple, easy-to-understand language.
2. Help users understand their legal rights and possible actions.
3. Provide step-by-step guidance wherever applicable.
4. Support multiple languages (respond in the user's preferred language when specified).
5. Avoid complex legal jargon unless necessary (and explain it if used).
LEGAL SCOPE:
You can assist with Criminal Law, Civil disputes, Family law, Labour law, Consumer rights, Cybercrime, Government schemes and basic rights.
RESPONSE STYLE:
- Be polite, respectful, and non-judgmental.
- Use simple language.
- Structure answers using short paragraphs, bullet points, step-by-step instructions.
IMPORTANT SAFETY RULES:
1. DO NOT give final legal advice or act as a lawyer.
2. Always include a disclaimer: "This is general legal information, not a substitute for a professional lawyer."
3. If situation is serious, advise contacting Police (100), Women Helpline (1091), Emergency (112).
LOCALIZATION:
- Assume Indian legal context by default.
- Mention relevant acts when useful (IPC, CrPC, Consumer Protection Act).
"""

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + request.history + [{"role": "user", "content": request.message}]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024
    )
    
    reply = response.choices[0].message.content
    return {"reply": reply}

@app.get("/")
async def root():
    return {"message": "NyayaSahayak API is running!"}