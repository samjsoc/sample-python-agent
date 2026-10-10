
"""FastAPI chat interface for the Pydantic AI task assistant.

Run:
    uv run python -m python_agent.api

Local:
    http://localhost:8000

OpenChoreo:
    http://development-default.openchoreoapis.localhost:19080/taks-agent-rest-api
"""

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits

from .agent import agent


# -------------------------------------------------------
# Application
# -------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Task Assistant",
    version="0.2.0",
)


# -------------------------------------------------------
# Conversation sessions
# -------------------------------------------------------

# In-memory conversation history.
# For multiple replicas, replace with Redis or PostgreSQL.

_sessions: dict[str, list] = {}
_session_locks: dict[str, asyncio.Lock] = {}


# -------------------------------------------------------
# Request and response models
# -------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    response: str


# -------------------------------------------------------
# Health endpoint
# -------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):

    session_id = body.session_id or uuid4()
    key = str(session_id)

    lock = _session_locks.setdefault(
        key, asyncio.Lock()
    )

    async with lock:
        history = _sessions.get(key)

        try:
            result = await agent.run(
                body.message,
                message_history=history,
                usage_limits=UsageLimits(
                    request_limit=12
                ),
            )

        except Exception:
            logger.exception(
                "Agent request failed"
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "The agent could not complete "
                    "the request. Check Ollama "
                    "and server logs."
                ),
            ) from None

        _sessions[key] = result.all_messages()

        return ChatResponse(
            session_id=session_id,
            response=str(result.output),
        )


# -------------------------------------------------------
# Delete conversation session
# -------------------------------------------------------

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: UUID):

    key = str(session_id)

    lock = _session_locks.setdefault(
        key, asyncio.Lock()
    )

    async with lock:
        _sessions.pop(key, None)

    return {
        "cleared": True,
        "session_id": key,
    }


# -------------------------------------------------------
# Browser Chat UI
# -------------------------------------------------------

# Raw string preserves JavaScript regex backslashes.
HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <title>Task Assistant</title>

    <style>
        body {
            font: 16px system-ui, sans-serif;
            max-width: 760px;
            margin: 36px auto;
            padding: 0 16px;
            background: #fafafa;
            color: #222;
        }

        #log {
            background: white;
            border: 1px solid #ddd;
            border-radius: 12px;
            min-height: 350px;
            max-height: 65vh;
            overflow: auto;
            padding: 20px;
            white-space: pre-wrap;
        }

        .entry {
            margin-bottom: 20px;
        }

        .who {
            font-weight: 700;
            margin-bottom: 4px;
        }

        form {
            display: flex;
            gap: 8px;
            margin-top: 14px;
        }

        input {
            flex: 1;
            min-width: 0;
            padding: 12px;
            border: 1px solid #aaa;
            border-radius: 8px;
            font: inherit;
        }

        button {
            padding: 10px 18px;
            border: 0;
            border-radius: 8px;
            background: #155eef;
            color: white;
            cursor: pointer;
        }

        button:disabled {
            opacity: .5;
        }
    </style>
</head>

<body>

<h1>Task Assistant</h1>

<p>Qwen through Ollama · Pydantic AI · SQLite</p>

<div id="log"
     role="log"
     aria-live="polite">
</div>

<form id="form">
    <input
        id="message"
        autocomplete="off"
        placeholder="Add a task to review a PR tomorrow at 9 AM"
        required
    >
    <button id="send">Send</button>
</form>

<p>
    <button id="new" type="button">
        New conversation
    </button>
</p>

<script>

    // --------------------------------------------------
    // API base path
    // --------------------------------------------------

    // Local:
    //   /                  -> ""
    //
    // OpenChoreo:
    //   /taks-agent-rest-api
    //       -> /taks-agent-rest-api

    const basePath = window.location.pathname
        .replace(/\/+$/, '');

    function apiUrl(path) {
        return `${basePath}${path}`;
    }

    console.log("API base path:", basePath);


    // --------------------------------------------------
    // Application state
    // --------------------------------------------------

    let sessionId = null;

    const log = document.getElementById('log');
    const input = document.getElementById('message');
    const send = document.getElementById('send');


    // --------------------------------------------------
    // Append chat message
    // --------------------------------------------------

    function append(who, msg) {

        const div = document.createElement('div');
        div.className = 'entry';

        const heading = document.createElement('div');
        heading.className = 'who';
        heading.textContent = who;

        const content = document.createElement('div');
        content.textContent = msg;

        div.append(heading, content);
        log.append(div);

        log.scrollTop = log.scrollHeight;
    }


    // --------------------------------------------------
    // HTTP request helper
    // --------------------------------------------------

    async function apiRequest(path, options = {}) {

        const response = await fetch(
            apiUrl(path),
            options
        );

        const rawText = await response.text();

        let data = null;

        if (rawText) {
            try {
                data = JSON.parse(rawText);
            } catch {
                if (response.ok) {
                    throw new Error(
                        "Server returned invalid JSON"
                    );
                }
            }
        }

        if (!response.ok) {
            const detail = data?.detail;

            throw new Error(
                typeof detail === 'string'
                    ? detail
                    : `HTTP ${response.status}: ${
                        rawText || response.statusText
                      }`
            );
        }

        return data;
    }


    // --------------------------------------------------
    // Send chat message
    // --------------------------------------------------

    document.getElementById('form')
        .addEventListener('submit', async (e) => {

            e.preventDefault();

            const msg = input.value.trim();

            if (!msg) return;

            append('You', msg);

            input.value = '';
            send.disabled = true;

            try {

                const data = await apiRequest(
                    '/chat',
                    {
                        method: 'POST',
                        headers: {
                            'Content-Type':
                                'application/json'
                        },
                        body: JSON.stringify({
                            message: msg,
                            session_id: sessionId
                        })
                    }
                );

                sessionId = data.session_id;

                append('Agent', data.response);

            } catch (err) {

                append('Error', err.message);

                console.error(
                    "Chat request failed:",
                    err
                );

            } finally {

                send.disabled = false;
                input.focus();

            }
        });


    // --------------------------------------------------
    // New conversation
    // --------------------------------------------------

    document.getElementById('new')
        .addEventListener('click', async () => {

            if (sessionId) {
                try {
                    await apiRequest(
                        `/sessions/${sessionId}`,
                        { method: 'DELETE' }
                    );
                } catch (err) {
                    console.warn(
                        "Could not clear session:",
                        err
                    );
                }
            }

            sessionId = null;

            log.replaceChildren();

            input.focus();
        });

</script>

</body>
</html>
"""


# -------------------------------------------------------
# Chat UI route
# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# -------------------------------------------------------
# Application entry point
# -------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
