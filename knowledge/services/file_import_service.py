import datetime
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import kb_import_process_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, TASK_STATUS_PROCESSING, update_task_status, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

logger = logging.getLogger(__name__)

class ImportFileService:
    """
    处理上传文件业务层类
    """
    def upload_file(self, file:UploadFile):
        """
        完成文件上传业务处理。  双写： 保存到本地  and  保存minio
        """

        #1.生成task_id
        task_id = self._generate_task_id()

        #2.文件上传存储路径    D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\temp_data\20260828
        date_path = self._get_date_path(get_local_base_dir())
        #   D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\temp_data\20260828\absdfdre
        file_dir = os.path.join(date_path, task_id)
        # 不在创建，在忽略
        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id,"upload_file")  # 名称与task_util.py中 英文转中文名称一致。

        #3.双写
        #3.1 保存到本地
        #D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\temp_data\20260828\absdfdre\万用表RS-12的使用.pdf
        import_file_path = self._upload_file_to_local(file, file_dir)

        #3.2 保存到minio     file.name用于minio存储子路径名称
        #   http://192.168.6.170:9000/knowledge-base-files/origin_files/20230405/万用表RS-12的使用.pdf
        self._upload_file_to_minio(import_file_path,file.filename)

        add_done_task(task_id,"upload_file")

        return task_id,file_dir,import_file_path

    def run_import_graph(self, task_id, file_dir, import_file_path):
        """
        启动导入流程 langgraph
        """
        init_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,  # 导入(出)文件目录
        }
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        try:
            for event in kb_import_process_graph.stream(init_state):  # stream_mode="updates"  增量结果
                for node_name, process_state in event.items():
                    print(f"{task_id}-运行节点: {node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.info(f"导入流程执行出错: {e}")

    def _generate_task_id(self):
        return uuid.uuid4().hex[:8]

    def _get_date_path(self, local_base_dir):
        return os.path.join(local_base_dir, datetime.datetime.now().strftime("%Y%m%d"))

    def _upload_file_to_local(self, file, file_dir) -> Path:
        """将上传文件保存到本地"""
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        try:
            import_file_path = os.path.join(file_dir, file.filename)
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)  # 带缓冲区写操作。效率高   1Mb
        except Exception as e:
            logger.error(f"文件保存本地出错: {e}")
            raise FileProcessingError(f"文件保存本地出错: {e}")

        return import_file_path

    def _upload_file_to_minio(self, import_file_path, file_name):
        """将上传文件保存到minio"""
        #1.获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"获取minio客户端出错: {e}")
            return   # 降级处理： 获取客户端失败，不影响后续流程。
        try:
            config = get_config()
            obj_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}"
            # url = http://192.168.6.170:9000/knowledge-base-files/origin_files/20230405/万用表RS-12的使用.pdf
            minio_client.fput_object(config.minio_bucket, obj_name, import_file_path)
        except MinioError as e:
            logger.warning(f"文件上传Minio出错: {e}")
            return  # 降级处理： 上传Minio失败，不影响后续流程。
