from typing import List, Dict, Tuple

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    """生成答案解决"""

    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """生成答案"""

        session_id = state.get("session_id")
        task_id = state.get("task_id")

        # 1. 检查是否有答案

        if state.get("answer"):
            # 1.1 有答案：  在商品名称确认节点中，生成答案：[用户选择  or 抱歉],不进行三路检索，直接生成答案
            self._push_existing_answer(state)  # 有答案,非流式
        else:
            # 1.2 无答案： 进行三路检索，排序，再生成答案，输出答案
            prompt = self._build_prompt(state)
            self._generate_answer(state, prompt)

        # 2. 写入历史对话
        self._write_history(state)

        # 3. 流式输出结束
        if state.get("is_stream"):
            # 流式输出，不管有答案还是没答案，最终都需要输出终止事件状态，即：通知客户端关闭SSE连接。
            # 有答案&流式输出
            # 无答案&流式输出
            push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get("answer")})

        return state

    def _push_existing_answer(self, state):
        """处理已有的答案，并且是非流式"""
        if not state.get("is_stream"):
            set_task_result(state['task_id'], "answer", state['answer'])

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        """根据检索结果、历史对话组装 LLM 提示词"""
        char_budget = self.config.max_context_chars  # 12000

        # 1. 获取问题和商品名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"

            if used_chars + len(formatted_line) > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars

    def _generate_answer(self, state, prompt):
        """无答案-> 三路检索 -> 生成答案"""

        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
        except Exception as e:
            return "抱歉,获取LLM客户端失败,无法生成答案!"

        if state.get("is_stream"):
            state["answer"] = self._generate_answer_stream(llm_client, state, prompt)
        else:
            state["answer"] = self._generate_answer_invoke(llm_client, state, prompt)
            set_task_result(state['task_id'], "answer", state['answer'])

    def _generate_answer_stream(self, llm_client, state, prompt):
        """流式生成答案"""
        try:
            answer = ""
            for chunk in llm_client.stream(prompt):  # 流式
                delta_text = chunk.get('content')
                answer += delta_text
                push_sse_event(task_id=state.get("task_id"), event=SSEEvent.DELTA, data={"delta": delta_text})
            return answer
        except Exception as e:
            return "抱歉,stream生成答案失败!"

    def _generate_answer_invoke(self, llm_client, state, prompt):
        try:
            llm_response = llm_client.invoke(prompt)  # invoke() 非流式
            return llm_response.content.strip()
        except Exception as e:
            return "抱歉,invoke生成答案失败!"

    def _write_history(self, state):
        """写入历史对话"""
        try:
            session_id = state.get("session_id")
            # 1.保存问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state.get("original_query", ""),
                rewritten_query=state.get("rewritten_query", "") or state.get("original_query", ""),
                item_names=state.get("item_names", []),
            )

            # 2.保存答案
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state.get("answer", ""),
                rewritten_query=state.get("rewritten_query", "") or state.get("original_query", ""),
                item_names=state.get("item_names", []),
            )
        except Exception as e:
            self.logger.error(f"保存会话失败: {e}")



if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")