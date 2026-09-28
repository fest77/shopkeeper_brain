import uuid
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, get_task_result


class QueryService:


    def run_query_graph(self,original_query, session_id, task_id, is_stream):
        """启动查询流程"""
        try:
            init_state = {
                "original_query": original_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }

            update_task_status(task_id, TASK_STATUS_PROCESSING) #流程开始状态

            # 虽然Langgraph流程是非流式的。但是答案生成节点输出流式的。通过SSE生成器，将数据从队列中取走，推流给前端。
            final_state = query_app.invoke(init_state)  # 执行查询流程7个节点, 7个节点状态在父节点中记录的。

            update_task_status(task_id, TASK_STATUS_COMPLETED) #流程结束状态
            return final_state
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)

    def generate_session_id(self):
        return str(uuid.uuid4())

    def generate_task_id(self):
        return str(uuid.uuid4())

    def get_task_result(self, task_id):
        return get_task_result(task_id,"answer")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        records.reverse()
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)