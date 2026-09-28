import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app):

    @app.get("/import")
    def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    """
        1.文件上传处理  由业务层来处理，需要创建业务层类，及对象。采用依赖注入的方式创建业务层对象，并且单例创建，并且缓存重复利用
        2.异步启动Langgraph流程
    """

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            service: ImportFileService = Depends(get_import_file_service),  # 依赖注入
            file: UploadFile = File(...)):
        """POST 请求处理 —— 接收上传文件"""
        # 文件上传（双写）
        # 任务ID（每次请求创建唯一字符串值 - 用于查询任务状态）
        # 文件上传目录
        # 文件上传后的完整路径
        task_id, file_dir, import_file_path = service.upload_file(file)

        # 同步启动流程   等待流程跑完才能返回结果给前端。用户需要等待很长时间，用户体验非常不好。
        #service.run_import_graph(task_id, file_dir, import_file_path)

        # 异步启动流程   异步开启任务，不会阻塞当前线程
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        # 返回结果数据模型： Panditic
        return UploadResponse(task_id=task_id,message="文件上传处理完成")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """GET 获取任务状态
            1.在文件上传及启动langgraph时，都需要记录各个任务的状态
            2.前端根据task_id，查询任务状态
            3.状态维护容器：
        """
        task_info =  get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def create_app():
    app = FastAPI(description="导入流程", version=1.0)

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
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
