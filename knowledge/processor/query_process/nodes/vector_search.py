# knowledge/processor/query_process/nodes/vector_search_node.py

"""向量检索节点

对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
"""

import json
import logging

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union

from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, item_names_filter, \
    milvus_client


class VectorSearchNode(BaseNode):
    """向量检索"""

    name = "search_embedding"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """
        检索主流程：问题向量化 → Milvus 混合检索 → 返回 {"embedding_chunks": [...]}

        state 中已有的字段：
            rewritten_query : str    改写后的查询词
            item_names      : list   商品名列表（用于过滤）
        """
        # ============ 1. 参数校验 ============
        # 调用 _validate_state(state)，返回 (validated_query, validate_item_names)
        #   validated_query      : str   校验通过后的改写问题
        #   validate_item_names  : list  校验通过后的商品名列表
        # TODO 第 1 步
        validated_query, validate_item_names = self._validate_state(state)

        # ============ 2. 获取 BGE-M3 嵌入模型客户端 ============
        # AIClients.get_bge_m3_client()  → embedding_model
        # 可能抛 ConnectionError → logger.error 并 return state（降级）
        # TODO 第 2 步
        try:
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # ============ 3. 获取 Milvus 客户端 ============
        # StorageClients.get_milvus_client()  → milvus_client
        # 可能抛 ConnectionError → logger.error 并 return state（降级）
        # TODO 第 3 步
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")

        # ============ 4. 对问题向量化 ============
        # generate_bge_m3_hybrid_vectors(model=embedding_model, embedding_documents=[validated_query])
        #   → embed_query = {'dense': [...], 'sparse': [...]}
        # 取出 embed_query['dense'][0]（稠密向量）、embed_query['sparse'][0]（稀疏向量）
        # 失败 → logger.error 并 return state
        # TODO 第 4 步
        try:
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_model, embedding_documents=[validated_query])
        except ValueError as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
        # ============ 5. 构建商品名过滤表达式 ============
        # item_names_filter(validate_item_names)
        #   → (filter_expr, filter_expr_param)
        #   filter_expr      形如 "item_name in {item_names}"
        #   filter_expr_param 形如 {"item_names": ["RS-12 数字万用表"]}
        # TODO 第 5 步
        filter_expr , filter_expr_param = item_names_filter(validate_item_names)

        # ============ 6. 创建混合检索请求 ============
        # create_hybrid_search_requests(
        #     dense_vector=..., sparse_vector=...,
        #     expr=filter_expr, expr_params=filter_expr_param)
        #   → hybrid_search_request（稠密+稀疏两个 AnnSearchRequest 的列表）
        # TODO 第 6 步
        try:
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=filter_expr,expr_params=filter_expr_param)

        # ============ 7. 执行混合检索 ============
        # execute_hybrid_search_query(
        #     milvus_client=milvus_client,
        #     collection_name=self.config.chunks_collection,
        #     search_requests=hybrid_search_request,
        #     output_fields=['chunk_id', 'content', 'item_name', 'title'])
        #   → hybrid_search_reps（每路查询的结果列表，[0] 是主结果）
        # TODO 第 7 步
            hybrid_search_reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=validated_query['collection_name'],
                search_requests=hybrid_search_requests,
                output_fields=['chunk_id', 'content', 'item_name', 'title'])

        # ============ 8. 结果非空判断 + 回填 state ============
        # 若 hybrid_search_reps 为空 或 hybrid_search_reps[0] 为空 → return state
        # 否则 return {"embedding_chunks": hybrid_search_reps[0]}
        # TODO 第 8 步
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state
            else:
                return {"embedding_chunks": hybrid_search_reps[0]}
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state


    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """
        校验输入参数：
            1. state.get('rewritten_query')：缺失或非 str → StateFieldError
            2. state.get('item_names')：缺失或非 list → StateFieldError
            3. 返回 (rewritten_query, item_names)
        """
        # TODO 补全本方法
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names


# ==================================================== #
#                        测试入口                        #
# ==================================================== #
if __name__ == '__main__':
    state = {
        "original_query": "万用表的使用如何测量电阻",
        "rewritten_query": "RS-12 数字万用表的使用如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
