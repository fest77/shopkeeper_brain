"""HyDE 检索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""

import json
import logging
from typing import List, Tuple, Union, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, milvus_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点

    流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """
    name = "search_embedding_hyde"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """
        执行 HyDE 检索，返回 {"hyde_embedding_chunks": [...]}

        state 中已有的字段：
            rewritten_query : str   改写后的查询词
            item_names      : list  商品名列表
        """
        # ============ 1. 参数校验 ============
        # _validate_query_inputs(state) → (validated_query, validate_item_names)
        # TODO 第 1 步
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # ============ 2. 生成假设性文档 ============
        # _generate_hy_document(validated_query, validate_item_names)
        #   → hy_document : str   LLM 写的"假设性回答文档"
        # TODO 第 2 步
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # ============ 3. 获取嵌入模型 & Milvus 客户端 ============
        # AIClients.get_bge_m3_client()  → embedding_model
        # StorageClients.get_milvus_client() → milvus_client
        # 任一为空 → return state（降级）
        # TODO 第 3 步
        embedding_model = AIClients.get_bge_m3_client()
        milvus_client = create_hybrid_search_requests(embedding_model)
        if not embedding_model or not milvus_client:
            return state

        # ============ 4. 假设性文档向量化（注入问题 + 假设文档）============
        # embedding_document = f"{validated_query}\n{hy_document}"
        # generate_bge_m3_hybrid_vectors(embedding_model, embedding_documents=[embedding_document])
        #   → embedding_result：{'dense':[...], 'sparse':[...]}
        # 为空 → return state
        # TODO 第 4 步
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_bge_m3_hybrid_vectors(model=embedding_model, embedding_document=[embedding_document])

        if not embedding_result:
            return state

        # ============ 5. 构建商品名过滤表达式 ============
        # _item_name_filter_expr(validate_item_names)
        #   → item_name_filtered_expr 形如 " item_name in ['RS-12 数字万用表']"
        # TODO 第 5 步
        item_name_filtered_expr = self._item_name_filter_expr(validate_item_names)

        # ============ 6. 创建混合检索请求 ============
        # create_hybrid_search_requests(
        #     dense_vector=embedding_result['dense'][0],
        #     sparse_vector=embedding_result['sparse'][0],
        #     expr=item_name_filtered_expr)
        # TODO 第 6 步
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filtered_expr)

        # ============ 7. 执行混合检索 ============
        # execute_hybrid_search_query(
        #     milvus_client, collection_name=self.config.chunks_collection,
        #     search_requests=..., norm_score=True,
        #     output_fields=["chunk_id", "content", "item_name", 'title'])
        #   → reps
        # TODO 第 7 步
        reps = execute_hybrid_search_query(milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id", "content", "item_name", 'title']
        )

        # ============ 8. 结果非空判断 + 返回 ============
        # 若 reps 为空或 reps[0] 为空 → return state
        # 否则 return {"hyde_embedding_chunks": reps[0]}
        # TODO 第 8 步
        if not reps or not reps[0]:
            return state

        return {"hyde_embedding_chunks": reps[0]}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """
        校验输入参数：
            1. state.get('rewritten_query')：缺失/非 str → StateFieldError
            2. state.get('item_names')：缺失/非 list → StateFieldError
            3. 返回 (rewritten_query, item_names)
        """
        # TODO 补全本方法
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not state.get('rewritten_query') or not isinstance(state.get('rewritten_query'), str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )
        if not state.get('item_names') or not isinstance(state.get('item_names'), list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        return rewritten_query, item_names


    def _item_name_filter_expr(self, validate_item_names: List[str]) -> str:
        """
        构建商品名过滤表达式：
            validate_item_names: 商品名列表，如 ["RS-12 数字万用表", "HAK180"]
            返回 str：' item_name in ["RS-12 数字万用表", "HAK180"]'
        """
        # TODO 补全本方法
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        """
        用 LLM 生成假设性文档：
            1. AIClients.get_llm_openai(False) 获取 LLM 客户端，为空返回 ""
            2. 拼用户提示词：HYDE_USER_PROMPT_TEMPLATE.format(item_names=...、rewritten_query=...)
            3. 拼系统提示词：f"您是一位{商品名}的技术文档领域专家..."
            4. llm_client.invoke([SystemMessage(...), HumanMessage(...)]) → llm_response
            5. getattr(llm_response, 'content', "") 取内容并 strip
            6. 为空返回 ""，否则返回内容；异常 logger.error 并返回 ""
        """
        # TODO 补全本方法
        llm_client = AIClients.get_llm_openai(False)

        if llm_client is None:
            return ""

        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names='、'.join(validate_item_names),
            rewritten_query=validated_query
        )
        system_prompt = f"您是一位{'、'.join(validate_item_names)}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            llm_response_content = getattr(llm_response, 'content', "").strip()

            if not llm_response_content:
                return ""
            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""


# ==================================================== #
#                        测试入口                        #
# ==================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")
