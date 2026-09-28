import asyncio
import os

import uvicorn
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import StreamSubmitResponse, QueryResponse, QueryRequest
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def register_router(app):

    @app.get("/chat")
    def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service)):

        """处理查询请求"""

        session_id = request.session_id or service.generate_session_id()
        original_query = request.query  # 用户问的问题
        task_id = service.generate_task_id()
        is_stream = request.is_stream  # 是否流式输出

        if is_stream:
            # 创建SSE队列。     存放内容： 7个节点状态,LLM输出
            create_sse_queue(task_id)

            # 流式输出  异步
            # service.run_query_graph(original_query, session_id, task_id, is_stream)  # 在事件循环中调用同步函数，会阻塞事件循环。
            background_tasks.add_task(service.run_query_graph, original_query=original_query, session_id=session_id, task_id=task_id, is_stream=is_stream)

            return StreamSubmitResponse(message="查询流程以启动,耐心等待答案!", session_id=session_id, task_id=task_id)

        else:
            """
                await loop.run_in_executor(None, service.run_query_graph, original_query=original_query, session_id=session_id, task_id=task_id, is_stream=is_stream)
                TypeError: BaseEventLoop.run_in_executor() got an unexpected keyword argument 'original_query'
            """

            # 非流式输出 同步
            loop = asyncio.get_running_loop()  # 获取正在运行的事件循环（管理协程）
            # executor=None 表示采用默认线程池处理任务。
            await loop.run_in_executor(None, service.run_query_graph, original_query, session_id, task_id, is_stream)
            # service.run_query_graph(original_query, session_id, task_id, is_stream)

            answer = service.get_task_result(task_id)

            return QueryResponse(message="最终答案", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """SSE流式输出"""
        return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app():
    app = FastAPI(description="检索流程", version=1.0)

    app.add_middleware(
        CORSMiddleware,  # 跨域设置
        allow_origins=["*"],  # 限制请求来源：  * 表示不限制访问来源   Access-Control-Allow-Origin 允许客户端向服务器发送任何请求
        allow_credentials=False,  # 如果为True,另外三个参数不能为*
        allow_methods=["*"],  # 请求方法限制   * 表示允许所有请求方法：  get  post  put   delete  ...
        allow_headers=["*"],  # 限制请求头     * 表示可以携带任何请求头信息
    )

    # 挂载前端静态资源
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
