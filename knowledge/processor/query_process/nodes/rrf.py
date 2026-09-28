from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """倒序融合排序"""
    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        RRF 融合主流程：统一格式 → 设权重 → RRF 公式打分 → 回填 rrf_chunks

        state 中已有的字段：
            embedding_chunks      : list   向量检索结果
            hyde_embedding_chunks : list   HyDE 检索结果
        """
        # ============ 1. 取两路检索结果 ============
        # state.get("embedding_chunks") or []
        # state.get("hyde_embedding_chunks") or []
        # TODO 第 1 步
        embedding_chunks: List[Dict[str, Any]] = state.get("embedding_chunks") or []
        hyde_embedding_chunks: List[Dict[str, Any]] = state.get("hyde_embedding_chunks") or []

        # ============ 2. 统一格式化 ============
        # _normalize_input(各路结果)  → 抽出每条的 entity 字段
        #   embedding_chunks_result / hyde_embedding_chunks_result
        # TODO 第 2 步
        embedding_chunks_result: List[Dict[str, Any]] = self._normalize_input(embedding_chunks)
        hyde_embedding_chunks_result: List[Dict[str, Any]] = self._normalize_input(hyde_embedding_chunks)

        # ============ 3. 设置每路权重 ============
        # rrf_inputs = [(embedding_chunks_result, 1.0), (hyde_embedding_chunks_result, 1.0)]
        #   每项元组：(该路文档列表, 该路权重)
        # TODO 第 3 步
        rrf_inputs: List[tuple[List[Dict[str, Any]], float]] = [(embedding_chunks_result, 1.0),
                                                          (hyde_embedding_chunks_result, 1.0)]

        # ============ 4. RRF 计算 ============
        # _rrf_merge(rrf_inputs=rrf_inputs)
        #   → rrf_results : List[(entity, 融合得分)] 按得分降序
        # TODO 第 4 步
        rrf_results: List[tuple[Dict[str, Any], float]] = self._rrf_merge(rrf_inputs=rrf_inputs)

        # ============ 5. 回填 state ============
        # state['rrf_chunks'] = [entity for entity, _ in rrf_results]
        # return state
        # TODO 第 5 步
        state['rrf_chunks'] = [entity for entity, _ in rrf_results]
        return state

    def _normalize_input(self, chunks_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一格式化输入数据：
            输入：每路检索结果，如
                [{"pk":1, "distance":0.7, "entity": {"chunk_id":1, "content":"...", "item_name":"...", "title":"..."}}]
            输出：抽取出 entity，如
                [{"chunk_id":1, "content":"...", "item_name":"...", "title":"..."}]
            逻辑：
                1. 遍历 chunks_input，跳过非 dict 的项
                2. chunk.get("entity") 若非 dict → 跳过
                3. entity_results.append(entity)
                4. 返回 entity_results
        """
        # TODO 补全本方法
        entity_results: List[Dict[str, Any]] = []
        for chunk in chunks_input:
            if not isinstance(chunk, dict):
                continue
            entity = chunk.get("entity")
            if not isinstance(entity, Dict):
                continue
            entity_results.append(entity)
        return entity_results


    def _rrf_merge(self, rrf_inputs: List[tuple[List[Dict[str, Any]], float]], rrf_k: int = 60,
                   top_k: int = 5) -> List[tuple[Dict[str, Any], float]]:
        """
        RRF 公式融合排序：
            RRF 得分公式：weight / (rrf_k + 排名)       （排名从 1 开始）

            rrf_inputs : List[ (文档列表, 权重) ]       每路检索
            rrf_k      : 平滑常数，默认 60
            top_k      : 返回前 N 条

            算法：
                1. entity_scores = {}  # {chunk_id: 累计得分}
                   entity_data   = {}  # {chunk_id: 文档实体}
                2. 遍历每路 (chunks, weight)：
                     遍历每路中第 index 个文档（排名 index+1）：
                         chunk_id = entity.get("chunk_id")
                         entity_scores[chunk_id] += weight / (rrf_k + (index+1))
                         entity_data.setdefault(chunk_id, entity)
                3. 按分数降序排序，截断 top_k
                4. 返回 [(entity_data[chunk_id], 分数)] 列表
        """
        # TODO 补全本方法
        entity_scores = {}
        entity_data = {}
        for chunks_input, weight in rrf_inputs:
            for index, entity in enumerate(chunks_input):
                chunk_id = entity.get("chunk_id")
                entity_scores[chunk_id] = entity_scores.get(chunk_id, 0) + weight / (rrf_k + (index + 1))
                entity_data.setdefault(chunk_id, entity)

        rrf_result = [
            (entity_data[item[0]], item[1])
            for item in sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        return rrf_result[:top_k]


# ================================================================== #
#                        测试入口                                     #
# ================================================================== #

if __name__ == "__main__":

    setup_logging()

    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
