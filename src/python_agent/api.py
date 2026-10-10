"""FastAPI interface for the existing Pydantic AI task assistant.

Start from the project root:
    uv add fastapi 'uvicorn[standard]'
    uv run uvicorn python_agent.api:app --app-dir src --host 127.0.0.1 --port 8000

Visit http://127.0.0.1:8000/ for a simple chat UI or /docs for the API.
"""

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits

from .agent import agent

logger = logging.getLogger(__name__)
app = FastAPI(title="Task Assistant", version="0.1.0")

# Demo only: chat history lives in RAM and disappears on restart.
# This is single-process storage. For multi-worker / multi-pod deployments,
# replace it with a durable shared store (e.g. PostgreSQL or Redis).
_sessions: dict[str, list] = {}
_session_locks: dict[str, asyncio.Lock] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    response: str


@app.get("/health")
def health():
    """Liveness check, not a connectivity check for Ollama."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    session_id = body.session_id or uuid4()
    key = str(session_id)
    lock = _session_locks.setdefault(key, asyncio.Lock())
    async with lock:
        history = _sessions.get(key)
        try:
            result = await agent.run(
                body.message,
                message_history=history,
                usage_limits=UsageLimits(request_limit=12),
            )
        except Exception:
            logger.exception("Agent request failed")
            raise HTTPException(
                status_code=502,
                detail="The agent could not complete the request. Check Ollama and server logs.",
            ) from None
        _sessions[key] = result.all_messages()
        return ChatResponse(session_id=session_id, response=str(result.output))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: UUID):
    """Forget the conversation history for a chat session (not its saved tasks)."""
    key = str(session_id)
    lock = _session_locks.setdefault(key, asyncio.Lock())
    async with lock:
        _sessions.pop(key, None)
    return {"cleared": True, "session_id": key}


HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task Assistant</title><style>
body{font:16px system-ui,sans-serif;max-width:760px;margin:36px auto;padding:0 16px;background:#fafafa;color:#222}
#log{background:white;border:1px solid #ddd;border-radius:12px;min-height:350px;max-height:65vh;overflow:auto;padding:20px;white-space:pre-wrap}
.entry{margin-bottom:20px}.who{font-weight:700;margin-bottom:4px}form{display:flex;gap:8px;margin-top:14px}input{flex:1;padding:12px;border:1px solid #aaa;border-radius:8px;font:inherit}
button{padding:10px 18px;border:0;border-radius:8px;background:#155eef;color:white;cursor:pointer}button:disabled{opacity:.5}
</style></head><body><h1>Task Assistant</h1><p>Qwen through Ollama · Pydantic AI · SQLite</p>
<div id="log" role="log" aria-live="polite"></div>
<form id="form"><input id="message" autocomplete="off" placeholder="Add a task to review a PR tomorrow at 9 AM" required><button id="send">Send</button></form>
<p><button id="new" type="button">New conversation</button></p>
<script>
let sessionId=null;
const log=document.getElementById('log'),input=document.getElementById('message'),send=document.getElementById('send');
function append(who,msg){let div=document.createElement('div');div.className='entry';let heading=document.createElement('div');heading.className='who';heading.textContent=who;let content=document.createElement('div');content.textContent=msg;div.append(heading,content);log.append(div);log.scrollTop=log.scrollHeight;}
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();const msg=input.value.trim();if(!msg)return;append('You',msg);input.value='';send.disabled=true;
try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg,session_id:sessionId})});let data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');sessionId=data.session_id;append('Agent',data.response);}catch(err){append('Error',err.message)}finally{send.disabled=false;input.focus()}});
document.getElementById('new').addEventListener('click',async()=>{if(sessionId){await fetch('/sessions/'+sessionId,{method:'DELETE'}).catch(()=>{});}sessionId=null;log.replaceChildren();input.focus()});
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )