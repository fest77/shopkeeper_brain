import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from numpy.f2py.rules import options
from pymilvus import AnnSearchRequest, MilvusClient

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class ItemNameExtractor:

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    def extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """LLM提取商品名称及问题重写
        original_query: "RS-12怎么测量电阻"        原始问题
        history_context: 历史对话上下文信息
        return {
            "item_names": ["RS-12","RS-12 万用表"],
            "rewritten_query": "RS-12 万用表怎么测量电阻"
        }
        """
        result = {"item_names": [], "rewritten_query": original_query}
        # 1.获取LLM客户端
        llm_client = AIClients.get_llm_openai(response_format=False)
        if llm_client is None:
            self.logger.warning("LLM客户端初始化失败")
            return result  # 降级处理

        # 2.调用LLM接口
        # 2.1 准备系统提示词
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        # 2.2准备用户提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query, history_text=history_context)

        # 2.3调用LLM
        llm_response = llm_client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        )

        # 3.处理LLM返回结果
        llm_content = llm_response.content.strip()

        parsed_result = self._clean_parse(llm_content)
        result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
        result["item_names"] = parsed_result.get("item_names")

        return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item.strip() for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    """商品名称对齐器
        1. 向量匹配
        2. 评分对齐
        3. 分数过滤
    """

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    def match_align_filter(self, item_names: List[str], item_name_collection: str) -> Tuple[List[str], List[str]]:
        """向量匹配、评分对齐、分数过滤"""

        # 1. 向量匹配
        search_results: List[Dict[str, Any]] = self._match_vector(item_names, item_name_collection)
        print("search_results:", search_results)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_results)

        # 3. 分数过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed,search_results)

        return confirmed, options

    def _match_vector(self, item_names: List[str], item_name_collection: str) -> List[Dict[str, Any]]:
        """向量匹配
            1.将商品名称向量化
            2.创建查询请求
            3.执行混合搜索
            4.获取搜索结果
            item_names: ['RS-12数字万用表',"HAK180"]           LLM提取的商品名称列表
            return [
                {
                    "extracted_name": "RS-12数字万用表",
                    "matches": [
                        {"item_name": "RS-12数字万用表", "score": 0.9}
                        {"item_name": "RS-13数字万用表", "score": 0.69}
                    ]
                },
                {
                    "extracted_name": "HAK180",
                    "matches": [
                        {"item_name": "HAK180", "score": 0.75}
                        {"item_name": "HAK181", "score": 0.60}
                    ]
                }
            ]
        """
        result: List[Dict[str, Any]] = []

        # 1.将商品名称向量化
        bge_m3_client = AIClients.get_bge_m3_client()
        if bge_m3_client is None:
            self.logger.warning("BGE-M3客户端初始化失败")
            return result

        milvus_client: MilvusClient = StorageClients.get_milvus_client()
        if milvus_client is None:
            self.logger.warning("Milvus客户端初始化失败")
            return result

        # 将LLM提取商品名称，进行向量化。  用于混合检索的查询的条件
        embedding_result = generate_bge_m3_hybrid_vectors(bge_m3_client, item_names)

        for index, extract_item_name in enumerate(item_names):
            # 2.创建查询请求, [稠密查询请求,稀疏查询请求]
            hybrid_search_requests: List[AnnSearchRequest] = create_hybrid_search_requests(
                dense_vector=embedding_result['dense'][index],
                sparse_vector=embedding_result['sparse'][index],
            )

            # 3.执行混合搜索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name=item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,  # RRF倒序融合排序   需要将 IP查询得分进行归一化处理
                limit=5,
                output_fields=["item_name"]
            )
            # 4.获取搜索结果
            result.append({
                "extracted_name": extract_item_name,  # LLM 提取商品名称
                "matches": [
                    {"item_name": h['entity']['item_name'], "score": h['distance']}
                    for h in hybrid_search_result[0] if hybrid_search_result
                ]
            })

        return result

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        """
        评分对齐
        search_results: List[Dict[str, Any]]
            [
                {
                    "extracted_name": "RS-12数字万用表",
                    "matches": [
                        {"item_name": "RS-12数字万用表", "score": 0.9}
                        {"item_name": "RS-13数字万用表", "score": 0.67}
                    ]
                },
                {
                    "extracted_name": "HAK180",
                    "matches": [
                        {"item_name": "HAK 180", "score": 0.75}
                        {"item_name": "HAK181", "score": 0.71}
                        {"item_name": "RS-12数字万用表", "score": 0.71}
                    ]
                }
            ]

        return
            tuple[List[str],List[str]]
                confirmed:确认的商品名列表，传给下游多路检索
                候选商品名列表，用于询问用户
        """
        comfirmed: List[str] = []
        options: List[str] = []

        for search_result in search_results:
            extracted_name = search_result.get("extracted_name")
            sorted_matches = sorted(search_result.get("matches"), key=lambda x: x['score'], reverse=True)  # 按分数降序排序
            # 获取大于0.7元素   高置信
            high = [match for match in sorted_matches if match.get('score') >= 0.7]


            if high:
                # 处理高置信
                # 场景1  数据库中匹配的商品的名称如果与LLM提取的名称一致，直接确认
                extract = next((h for h in high if h['item_name'] == extracted_name),None)
                if extract:
                    picked = extract['item_name']
                    if picked not in comfirmed:
                        comfirmed.append(picked)
                # 场景2
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in comfirmed:
                        comfirmed.append(picked)
                # 场景3
                else:
                    for h in high[:3]:
                        picked = h['item_name']
                        if picked not in comfirmed and picked not in options:
                            options.append(picked)

            else:
                # 处理中置信
                mid = [match for match in sorted_matches if match.get('score') >= 0.6
                       and match.get("item_name") not in comfirmed
                       and match.get("item_name") not in options]
                if mid:
                    for m in mid[:3]:
                        picked = m['item_name']
                        options.append(picked)

        return comfirmed, options[:3]

    def _item_name_score_filter(self, confirmed:List[str], search_results:List[Dict[str, Any]]) -> List[str]:
        """  假设大于0.6都是高置信
            search_results =  [
                {
                    "extracted_name": "RS-12数字万用表",
                    "matches": [
                        {"item_name": "RS-12数字万用表", "score": 0.71}
                        {"item_name": "RS-13数字万用表", "score": 0.67}
                    ]
                },
                {
                    "extracted_name": "HAK180",
                    "matches": [
                        {"item_name": "HAK 180", "score": 0.75}
                        {"item_name": "HAK181", "score": 0.71}
                        {"item_name": "RS-12数字万用表", "score": 0.91}
                    ]
                }
            ]
        临时商品名称对应最大分数字典：
            {
                "RS-12数字万用表":0.9,
                "RS-13数字万用表":0.67,
                "HAK 180":0.75,
                "HAK181":0.71
            }
            字典按照分数倒序排序：获取最大分数： 0.9
             {
                "RS-12数字万用表":0.9,
                "HAK 180":0.75,
                "HAK181":0.71
                "RS-13数字万用表":0.67,
            }
             0.15  分数差过滤阈值： 小于等于0.15保留   否则去除
            0.9 - 0.9 = 0  <= 0.15      true      保留
            0.9 - 0.75 = 0.15 <= 0.15   true      保留
            0.9 - 0.71 = 0.19 <= 0.15   false    不保留
            0.9 - 0.67 = 0.23 <=0.15    false     不保留

            {
                "RS-12数字万用表":0.9,
                "HAK 180":0.75
            }
        """

        # 1. 获取每个商品名在向量数据库中的最高分数
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        # 2. 对 item_name_score 进行排序
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)

        # 3. 取出分数值最大的（作为基准）
        max_item_name_score = sorted_item_name_score[0][1]

        # 4. 保留分数差在阈值内的商品名
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]



class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点"""

    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self.item_name_extractor = ItemNameExtractor(self.logger, self.name)
        self.item_name_aligner = ItemNameAligner(self.logger, self.name)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.Step1 获取历史对话
        session_id = state.get("session_id")
        original_query = state.get("original_query")

        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id)
        history_messages.reverse()
        history_context = "暂无历史对话信息"
        for message in history_messages:
            history_context += message["role"] + ":" + message["text"] + "\n"

        print("10条数据（最近5轮对话）history_context:", history_context)

        # 2.Step2 LLM提取商品名称
        extractor_item_name: Dict[str, Any] = self.item_name_extractor.extract_item_name(original_query,
                                                                                         history_context)
        #print(extractor_item_name)
        item_name = extractor_item_name['item_names']
        rewritten_query = extractor_item_name['rewritten_query']

        # 3.Step3 向量匹配、评分对齐、分数过滤
        confirmed: List[str] = []
        options: List[str] = []
        if extractor_item_name['item_names']:
            confirmed, options = self.item_name_aligner.match_align_filter(item_name,
                                                                           self.config.item_name_collection)

        # 4.Step4 决策分支   1.无答案 -> 三路检索    2.有答案    2.1 用户选择  2.2 抱歉重问
        self._decide(state, item_name, confirmed, options, rewritten_query)

        # 5.Step5 回填处理 等
        # 将历史对话写入 state，供下游使用
        state["history"] = history_messages

        return state

    def _decide(self, state: QueryGraphState, item_names: List[str],
                confirmed: List[str], options: List[str], rewritten_query: str):
        """根据对齐结果更新 state"""
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options:
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmNode()
    init_state = {
        "session_id": "123456",
        #"original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        "original_query": "数字万用表和LA2608室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        #"original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
        #"original_query": "今天天气怎么样?"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node(init_state)

    print(llm_result)
