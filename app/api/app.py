import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import uuid

from core.config import get_settings
from core.rag import AdvancedRAG
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Financial News Chat Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,  # Important for cookies
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request schema
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

# Response schema 
class ChatResponse(BaseModel):
    response_type: str
    content: str
    tools_used: list
    tool_results: dict
    conversation_id: str

settings = get_settings()
rag_system = AdvancedRAG(data_path="../data/stock_news.json")

@app.get("/")
async def root():
    return {"message": "Welcome to the Financial News Chat Agent API!"}

@app.post("/chat")
async def chat(
    request: ChatRequest, 
    conversation_id: Optional[str] = Cookie(None)
):
    try:
        # Use the conversation_id from the request if provided
        # Otherwise, use the one from the cookie if available
        active_conversation_id = request.conversation_id or conversation_id
        
        logger.info(f"Processing query with conversation_id: {active_conversation_id}")
        
        response = rag_system.process_query(request.query, active_conversation_id)
        
        # Get the conversation_id that was used (could be a new one)
        used_conversation_id = active_conversation_id
        if used_conversation_id is None:
            # This means a new conversation_id was created in process_query
            # We need to extract it from the latest conversation in memory
            for conv_id, exchanges in rag_system.conversation_manager.conversations.items():
                if exchanges and exchanges[-1]["query"] == request.query:
                    used_conversation_id = conv_id
                    break
        
        # Add the conversation_id to the response
        response_with_id = {**response, "conversation_id": used_conversation_id}
        
        # Create a response with a cookie to persist conversation_id
        api_response = JSONResponse(content=response_with_id)
        api_response.set_cookie(
            key="conversation_id", 
            value=used_conversation_id,
            httponly=True,
            max_age=3600,  # 1 hour
            samesite="lax"
        )
        
        return api_response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to start a new conversation (reset history)
@app.post("/new-conversation")
async def new_conversation():
    new_id = str(uuid.uuid4())
    logger.info(f"Creating new conversation with id: {new_id}")
    
    response = JSONResponse(content={"conversation_id": new_id})
    response.set_cookie(
        key="conversation_id", 
        value=new_id,
        httponly=True,
        max_age=3600,  # 1 hour
        samesite="lax"
    )
    return response

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
