import asyncio
from asyncio import Queue
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from myagents.core.envs.base import Environment
from myagents.core.llms.config import BaseCompletionConfig



class API(FastAPI):
    """API is the API for the myagents.
    
    Attributes:
        app (FastAPI):
            The FastAPI app.
    """
    app: FastAPI
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = FastAPI()
        
        # Environments
        self.envs: dict[str, Environment] = {}
        
        # Post init
        self.post_init()
        
    def post_init(self) -> None:
        """Post init the API.
        """
        @self.app.post("/environments")
        async def create_environment(request: Request):
            # Create a new environment
            env = Environment()
            # Add the environment to the API
            self.envs[env.uid] = env
            # Return the environment
            return env
        
        @self.app.post("/chat")
        async def chat(content: str, env_id: str):
            # Get the environment
            env = self.envs[env_id]
            # Run the environment
            return await env.run(content)
        
        @self.app.post("/stream_chat")
        async def stream_chat(content: str, env_id: str) -> StreamingResponse:
            # Get the environment
            env = self.envs[env_id]
            # Create a new queue
            queue = Queue()
            # Create a completion config
            completion_config = BaseCompletionConfig(
                stream=True,
                stream_queue=queue,
            )
            # Run the environment
            task = asyncio.create_task(env.run(content, completion_config=completion_config))
            
            # Create a new async generator
            async def stream_generator():
                while True:
                    try:
                        # 尝试在0.5秒内获取队列数据
                        response = await asyncio.wait_for(queue.get(), timeout=0.5)
                        yield response
                    except asyncio.TimeoutError:
                        # 超时后检查task是否完成
                        if task.done():
                            break
                        # 否则继续循环等待
            
            # Return the streaming response
            return StreamingResponse(
                stream_generator(), 
                media_type="application/json",
            )
        
    def add_route(self, path: str, endpoint: Callable):
        self.app.add_route(path, endpoint)
        
    def run(self, *args, **kwargs):
        uvicorn.run(self.app, *args, **kwargs)
