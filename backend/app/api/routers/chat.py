import json
import time
from queue import Queue
from typing import List
from llama_index.core.instrumentation.events.base import BaseEvent
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException, Request, status
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatDeltaReceivedEvent,
    StreamChatStartEvent,
    StreamChatEndEvent,
)
from llama_index.core.utilities.token_counting import TokenCounter
from app.engine import get_chat_engine
import json
import httpx
import logging
import requests
import dataclasses

# from llama_index.core.llms import ChatMessage, MessageRole

chat_router = r = APIRouter()

dispatcher = get_dispatcher()


class EventToSend(BaseModel):
    type: str = Field(default="data", pattern="data|text")
    status: str = Field(default="loading", pattern="loading|done")
    is_last_event: bool = False
    message: str


tokens_used = {
    "in": 0,
    "out": 0,
}
event_q = Queue()


class CustomEventHandler(BaseEventHandler):
    def handle(self, event: BaseEvent) -> None:
        if isinstance(event, RetrievalStartEvent):
            event_q.put(EventToSend(message="Retrieving relevant nodes..."))
        elif isinstance(event, EmbeddingEndEvent):
            event_q.put(
                EventToSend(
                    status="done",
                    message=f"Done embedding {len(event.chunks)} query chunks for retrieval.",
                )
            )
        elif isinstance(event, RetrievalEndEvent):
            event_q.put(
                EventToSend(
                    status="done",
                    message=f"Retrieved {len(event.nodes)} relevant nodes for context.",
                )
            )
        elif isinstance(event, StreamChatStartEvent):
            event_q.put(
                EventToSend(status="done", message="Started streaming chat response.")
            )
        elif isinstance(event, StreamChatDeltaReceivedEvent):
            event_q.put(EventToSend(type="text", message=str(event.delta)))
        elif isinstance(event, LLMChatEndEvent):
            token_counter = TokenCounter()
            tokens_used["in"] += token_counter.estimate_tokens_in_messages(
                event.messages
            )
            tokens_used["out"] += token_counter.get_string_tokens(
                str(event.response.message)
            )
        elif isinstance(event, StreamChatEndEvent):
            event_q.put(
                EventToSend(
                    status="done",
                    is_last_event=True,
                    message=f"Finished streaming chat response. Tokens used -> Input: {tokens_used['in']} & Output: {tokens_used['out']}",
                )
            )


dispatcher.add_event_handler(CustomEventHandler())


class _Message(BaseModel):
    role: MessageRole
    content: str


class _ChatData(BaseModel):
    messages: List[_Message]


stream_part_types = {
    "text": "0",
    "function_call": "1",
    "data": "2",
    "error": "3",
    "assistant_message": "4",
    "assistant_data_stream_part": "5",
    "data_stream_part": "6",
    "message_annotations_stream_part": "7",
}


# @r.post("")
# async def chat(
#     request: Request,
#     data: _ChatData,
#     chat_engine: BaseChatEngine = Depends(get_chat_engine),
# ):
#     # check preconditions and get last message
#     if len(data.messages) == 0:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No messages provided",
#         )
#     lastMessage = data.messages.pop()
#     if lastMessage.role != MessageRole.USER:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Last message must be from user",
#         )
#     # convert messages coming from the request to type ChatMessage
#     messages = [
#         ChatMessage(
#             role=m.role,
#             content=m.content,
#         )
#         for m in data.messages
#     ]

#     # query chat engine
#     response = await chat_engine.astream_chat(lastMessage.content, messages)

#     # stream response
#     async def event_generator():
#         while True:
#             if await request.is_disconnected():
#                 break
#             next_event: EventToSend = event_q.get(timeout=30.0)
#             if next_event.type == "text":
#                 yield f"{stream_part_types[next_event.type]}:{json.dumps(next_event.message)}\n"
#             else:
#                 yield f"{stream_part_types[next_event.type]}:{json.dumps([next_event.model_dump()])}\n"
#             if next_event.is_last_event:
#                 break

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={"X-Experimental-Stream-Data": "true"},
#     )

ERROR_MESSAGE = """The app encountered an error processing your request.
If you are an administrator of the app, view the full error in the logs. See aka.ms/appservice-logs for more information.\n
{error_message} \n
"""
ERROR_MESSAGE_FILTER = (
    """Your message contains content that was flagged by the OpenAI content filter."""
)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def error_dict(error: Exception) -> dict:
    if isinstance(error, APIError) and error.code == "content_filter":
        return {"error": ERROR_MESSAGE_FILTER}

    error_message = error.args[0] if error.args else str(error)
    return {"error": ERROR_MESSAGE.format(error_message=error_message)}


from openai import APIError


@r.post("")
async def chatv2(request: Request):
    print(request.body())
    if not request.headers.get("content-type") == "application/json":
        raise HTTPException(status_code=415, detail="Request must be JSON")

    request_json = await request.json()
    print(f"REQUEST JSON {request_json}")
    # url = request_json["endpoint"]
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        # "Authorization": f'Bearer {request_json["key"]}',
    }
    data = {
        "question": request_json["messages"][-1]["content"],
        "chat_history": [],
    }

    async def fetch_data(url, headers, data):
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", url, headers=headers, json=data
            ) as response:
                async for chunk in response.aiter_text():
                    yield chunk

    async def event_generator():
        message_id = 0  # Initialize message ID
        try:
            async for chunk in fetch_data("http://localhost:8087/score", headers, data):
                print("--------------------------------------------")
                # Ensure chunk is a bytes-like object before encoding and splitting
                if isinstance(chunk, str):
                    chunk_bytes = chunk.encode()
                else:
                    chunk_bytes = chunk
                # Splitting the bytes-like object correctly after converting it from JSON
                # print(f"CHUNK BYTES: {chunk_bytes}")
                try:
                    data_json = json.loads(
                        chunk_bytes.decode().split("data: ")[1]
                    )  # Convert JSON string to a dictionary
                    # print(f"OUTPUT DATA: {data_json['answer']['messages'][-1:]}")
                    # convert messages coming from the request to type ChatMessage
                    last_mesg = data_json["answer"]["messages"][
                        -1
                    ]  # Assuming you want the last message
                    message_id += 1  # Increment message ID for each new message
                    # print(f"LAST MESG DATA: {last_mesg}")
                    test = ChatMessage(
                        role=last_mesg["role"],
                        content=last_mesg["content"],
                    )
                    # Since 'test' is not accessed, you might want to use it or remove this line if unnecessary.
                    # For demonstration, let's yield 'test' as part of the stream.
                    yield f'{stream_part_types["data_stream_part"]}:{json.dumps([{"id":message_id,"role": test.role, "content": test.content}])}\n'
                except IndexError:
                    print(f"Error in processing chunk: {chunk_bytes}")
                    # Handle the case where the split does not work as expected
                    pass
                    # yield f"data:Error in processing chunk\n\n"
        except Exception as e:
            print(f"Error in processing chunk: {e}")
            # yield f"data:Error: {str(e)}\n\n".encode()
        finally:
            # Ensure the final 0-sized chunk is sent
            yield b""

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Experimental-Stream-Data": "true"},
    )
