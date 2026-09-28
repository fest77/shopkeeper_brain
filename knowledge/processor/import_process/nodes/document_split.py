"""

    导入流程的 文档切分节点

"""
import json
import os
import re
from typing import Tuple, Dict, List, Any
# 引入 LangChain 的文本切分器，核心工具
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 引入项目内部的基类和工具
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """文档切分节点"""

    name: str = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """文档切分处理流程"""

        # 1.获取输入进行数据校验
        md_content, file_title, max_content_length, min_content_length = self._get_input_validation(state)

        # 2.标题切分
        sections: List[Dict[str, Any]] = self._split_by_title(md_content, file_title)
        # print(sections)

        # 3.切分合并：  长切短合
        final_sections: List[Dict[str, Any]] = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4.组装切分
        chunks = self._assemble_chunk(final_sections,)

        # 5.日志统计并备份
        self._log_summary(md_content, chunks, max_content_length)

        # 6.更新状态并返回
        self._backup_chunks(state, chunks)
        state['chunks'] = chunks
        return state

    def _get_input_validation(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        """对输入数据进行校验"""
        self.log_step("step1","切分文档的参数校验以及获取...")
        # 从状态中获取 Markdown 内容
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(self.name, "md_content", str)

        # 获取文件名
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)

        # 获取配置的长度限制
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length

        if max_content_length < 0 or min_content_length < 0 or max_content_length <= min_content_length:
            raise ValidationError("切分最大值和最小值参数错误", self.name)

        return md_content, file_title, max_content_length, min_content_length

    def _split_by_title(self, md_content, file_title) -> List[Dict[str, Any]]:
        """根据标题切分文档
        md_content: 整个文档内容
        file_title: 文件标题 不带扩展名
        return [
            {
                "parent_title": "# HAK 180",
                "title": "## HAK 180 烫金机",
                "body": "产品安全手册（简体中文）...",
                "file_title": "hak180产品安全手册"
            }
        ]
        """
        self.log_step("step2", "根据标题进行切分...")
        sections: List[Dict[str, Any]] = []
        in_fence = False  # 是否在围栏内
        heading_re = re.compile(
            r"^\s*(#{1,6})\s+.+")  # 匹配标题正则表达式,(#{1,6}) 匹配组,取标题级别：match.group(1)     原文match.group(0)
        body: List[str] = []  # 收集正文
        content_lines = md_content.split("\n")
        current_title = ""  # 当前标题
        current_level = 0  # 当前标题级别
        hierarchy = [""] * 7  # 标题层级     ["","一级","二级","三级","四级","五级"，""]

        # 封装section(段落 + 元信息)
        def _flush():

            if current_title or body:
                parent_title = ""
                for lev in range(current_level - 1, 0, -1):
                    if hierarchy[lev]:
                        parent_title = hierarchy[lev]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "parent_title": parent_title,
                    "title": current_title if current_title else file_title,  # 特殊情况，例如：处理第一个标题前的段落。这个段落没有标题，也没有父标题
                    "body": "\n".join(body),
                    "file_title": file_title
                })

        for index, line in enumerate(content_lines):
            # 处理围栏（代码块/表格）
            if line.startswith("~~~~") or line.startswith("```"):
                in_fence = not in_fence

            # 匹配标题 (且不在围栏内)
            match = heading_re.match(line) if not in_fence else None

            # 是标题
            if match:
                _flush()  # 帮我把当前标题，之前的内容封装成secion ->  List
                # 更新状态
                level = len(match.group(1))  # 标题级别   1-6 值
                current_level = level
                current_title = line
                hierarchy[current_level] = current_title

                # hierarchy = [""] * 7  # 重置，因为不重置，这里存放的都是上个段落对应的标题。 全部置空是错误的。
                # 清理子级：比如从 H2 跳到 H3，要把原来的 H4, H5 清空，防止层级错乱
                for i in range(level + 1, 7):  # 只清理大于当前级别的子级别，因为这些子级别是上个段落遗留的。当前级别父级别不能清理，因为多个子对应同一个父时,其他子还要用到这个父。
                    hierarchy[i] = ""
                body = []  # 清空正文缓存，只装填当前标题的正文行

            else:
                body.append(line.strip())  # 收集正文

        # 处理最后一个段落
        _flush()
        return sections

    # ------------------------------------------------------------------ #
    #                     Step 3: 二次切分 + 合并短章节                      #
    # ------------------------------------------------------------------ #

    def split_and_merge(self, sections: List[Dict[str, Any]],
                        max_content_length: int, min_content_length: int):
        """
        二次切分和合并

        Args:
            sections: 根据一级标题切分后的所有 section（章节）块
            max_content_length: 每一个 section 的 content 内容最大长度
            min_content_length: 触发合并的最小长度
        """
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            #  current_sections.append([{},{}])            [[{},{}],[{},{}],[{},{}]]
            #  current_sections.extend([])                 [{},{},{},{}]
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)
        print(final_sections)

        # 3. 返回
        return final_sections

    def split_long_section(self, section: Dict[str, Any], max_content_length: int = 1000) -> List[Dict[str, Any]]:
        """切分长段落"""

        self.log_step("step3", "进行长内容的切分")

        # 1.获取section属性值
        title = section.get("title")
        parent_title = section.get("parent_title")
        body = section.get("body")  # 段落内容
        file_title = section.get("file_title")

        # 2.表格处理
        if "<table>" in body:
            self.logger.info("检测到了表格数据...")
            # 降维处理
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 最大标题长度限制
        MAX_TITLE_LENGTH = 50
        if len(title) > MAX_TITLE_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:MAX_TITLE_LENGTH]

        # 3.定义标题前缀
        title_prefix = title + "\n\n"

        # 4.标题+正文  小于 阈值  ， 不用切分直接返回,但是也存储列表返回。
        if len(title_prefix) + len(body) <= max_content_length:
            return [section]

        # 5.获取body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        sub_sections = []

        # 6.需要切分
        if len(title_prefix) + len(body) > max_content_length:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", ".", ",", "!", " ", ""],  # 用于切分的分隔符
                chunk_size=body_length,  # 切片最大长度
                chunk_overlap=0,  # 不重叠
                keep_separator=False,  # 不保留分隔符
            )

            texts = splitter.split_text(body)
            if len(texts) <= 1:
                return [section]

            for index, text in enumerate(texts):
                # 重新封装section
                sub_sections.append({
                    "parent_title": parent_title,
                    "title": section.get("title") + f"-{index + 1}",
                    "body": text,
                    "file_title": file_title,
                    "part": f"{index + 1}"
                })
        return sub_sections

    def merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length) -> List[Dict[str, Any]]:
        """合并短章节内容"""
        final_sections: List[Dict[str, Any]] = []

        # 当前section
        current_section = current_sections[0]

        # 用于指向同一个父标题的下一个元素section
        for next_section in current_sections[1:]:
            same_parent = current_section["parent_title"] == next_section["parent_title"]
            if same_parent and len(current_section['body']) < min_content_length:
                current_section['body'] = current_section['body'].rstrip() + "\n\n" + next_section['body'].lstrip()
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                final_sections.append(current_section) #将同一个标题的短的正文 合并成大正文
                current_section = next_section

        final_sections.append(current_section)

        # 4. 对所有 section 的 part 做处理
        part_counter = {}
        #result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            #result.append(final_section)

        return final_sections



    # ------------------------------------------------------------------ #
    #               Step 4: 组装最终 chunk                                 #
    # ------------------------------------------------------------------ #

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最终组合 chunk"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            # 1. 获取 chunk 的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk = {
                "title": title, # 当前所属的标题
                "file_title": file_title, # 文件名
                "parent_title": parent_title, # 父级标题
                "content": content, # 正文文本
            }

            # 3. 判断 part 是否存在
            "part是章节过长时,需要二次切分时的标记"
            if "part" in chunk:
                # 把这个 part 编号（比如 1 或 2）从旧 chunk 复制到 assemble_chunk
                assemble_chunk['part'] = chunk.get('part')
            chunks.append(assemble_chunk)

        return chunks



    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")
        # 统计文档相关信息
        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")
        # 预览前5个章节的标题
        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30] # 只取标题的前30个字符
                self.logger.info(f" {i + 1}. {title}...")
            if len(chunks) > 5: # 如果章节超过5个，提示还有多少
                self.logger.info(f"...还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"E:\Atemp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表RS-12的使用",
        "md_content": content,
        "file_dir": r"D:\A_Py_Java\pyFile\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\万用表RS-12的使用\auto" # 此处需要修改!!!
    }
    document_node.process(state)
