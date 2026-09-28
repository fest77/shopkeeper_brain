import asyncio
import json
from typing import Dict, Tuple

from agents.mcp import MCPServerStreamableHttp
from openai.types.responses import web_search_tool

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> Dict:
        """
        三路检索之一：基于 web 联网的 MCP 服务调用。
        state 中已有的字段：
            rewritten_query : str    改写后的查询词
            item_names      : list   商品名列表
        返回：
            {"web_search_docs": [...]}
        """
        # ============ 1. 参数校验 ============
        # _validate_input(state) → (validateed_rewritten_query, validated_item_names)
        # TODO 第 1 步
        validateed_rewritten_query, validated_item_names = self._validate_input(state)

        # ============ 2. 创建 MCP 客户端并调用搜索（异步转同步）============
        # asyncio.run(self._web_mcp(validateed_rewritten_query))
        #   → web_search_docs : list   [{snippet, title, url}, ...]
        # TODO 第 2 步
        web_search_docs = asyncio.run(self._web_mcp(validateed_rewritten_query))

        # ============ 3. 封装返回结果 ============
        # return {"web_search_docs": web_search_docs}
        # TODO 第 3 步
        return {"web_search_docs": web_search_docs}

    def _validate_input(self, state) -> Tuple[str, list]:
        """
        校验输入参数：
            1. state.get("rewritten_query")：缺失/非 str → StateFieldError
            2. state.get("item_names")：缺失/非 list → StateFieldError
            3. 返回 (rewritten_query, item_names)
        """
        # TODO 补全本方法
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"Invalid rewritten_query: {rewritten_query}")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"Invalid item_names: {item_names}")
            raise StateFieldError(self.name, "item_names", str)
        return rewritten_query, item_names

    async def _web_mcp(self, validateed_rewritten_query):
        """
        异步调用 MCP 网络搜索服务：
            1. async with MCPServerStreamableHttp(
                   name="网络搜索",
                   params={
                       "url": self.config.mcp_dashscope_base_url,        # MCP 服务端点
                       "headers": {"Authorization": f"Bearer {self.config.dashscope_api_key}"},
                       "timeout": 300,
                       "terminate_on_close": True,
                   },
                   max_retry_attempts=2, client_session_timeout_seconds=30, cache_tools_list=True) as client:
            2. await client.call_tool(tool_name="bailian_web_search",
                                      arguments={"query": validateed_rewritten_query, "count": 3})
               → execute_tool_result
            3. 判空：无结果 → return []
            4. 解析：json.loads(execute_tool_result.content[0].text) → text_json
               text_json.get("pages") → pages，为空 → return []
            5. 遍历 pages，提取 snippet/title/url，封装成 web_search_docs 列表返回
        """
        # TODO 补全本方法
        async with MCPServerStreamableHttp(
                name="网络搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                    "headers": {"Authorization": f"Bearer {self.config.dashscope_api_key}"},
                    "timeout": 300,
                    "terminate_on_close": True,
                },
                max_retry_attempts=2,
                client_session_timeout_seconds=30,
                cache_tools_list=True
        ) as client:
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validateed_rewritten_query, "count": 3},
            )
            print(execute_tool_result)

            if not execute_tool_result or not execute_tool_result.content or not execute_tool_result.content[0] :
                return []
            text_json = json.loads(execute_tool_result.content[0].text)
            if not text_json:
                return []
            pages = text_json.get("pages")
            if not pages:
                return []
            web_search_docs = []
            for page in pages:
                snippet = page.get("snippet")
                title = page.get("title")
                url = page.get("url")
                web_search_docs.append(
                    {
                        "snippet": snippet,
                        "title": title,
                        "url": url
                    }
                )
            return web_search_docs


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
