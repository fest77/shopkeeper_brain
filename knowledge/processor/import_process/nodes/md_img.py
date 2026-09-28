"""

    导入流程的 MD中图片的处理节点

        替换图片url
        为图片生成摘要

"""
import base64
import json
import re
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Dict, Deque

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str  # 最近的章节标题
    pre_text: str  # 图片上方的正文内容
    post_text: str  # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str  # 图片文件名
    path: str  # 图片完整路径
    context: ImageContext  # 在 MD 中的上下文


# ------1.MD文件读取 & 备份----------------
class MdFileHandler:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state) -> tuple[str, Path, Path]:
        """读取MD文件内容"""
        self.logger.info("【step_1】读取MD内容及构建图片目录")

        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(self.node_name, "md_path", str, "属性不存在")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(f"路径不存在：{md_path}", self.node_name)

        with open(md_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        image_path_obj = md_path_obj.parent / "images"

        return md_content, md_path_obj, image_path_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


# ------2.图片扫描----------------
"""
1.pdf中表格图片，在md中以<table>标签表示的。所以md中没有表格图片引用。

2..正则表达式匹配图片时
     if re.search(r"^!\[.*?\]\(.*?" + re.escape(image_name)$ + ".*?\)$", line):              必须以)结尾，而pdf转md时，图片标记行   ) 后面有空格 则无法匹配。
     去掉 $
     if re.search(r"^!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)$", line):
"""


class ImageScanner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, image_path_obj: Path, md_content, image_extensions, img_content_length) -> List[ImageInfo]:
        """
        扫描图片目录，获取图片名称。从md_content中，匹配图片标记 ![]() ,找到图片位置上下文，封装ImageInfo对象，返回它的列表。
        """
        self.logger.info(f"【step_2】扫描图片目录 {image_path_obj}")
        imageinfo_list = []

        # 迭代图片目录下的资源名称
        for image_path in image_path_obj.iterdir():
            # images目录下不是文件，则是目录，跳过
            if not image_path.is_file():
                continue

            # {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}  判断是否是一个图片
            if image_path.suffix.lower() not in image_extensions:
                continue

            ctx: ImageContext = self._find_context(image_path.name, md_content, img_content_length)

            if ctx:
                image_info = ImageInfo(
                    name=image_path.name,
                    path=str(image_path),
                    context=ctx
                )
                imageinfo_list.append(image_info)
        self.logger.info(f"找到 {len(imageinfo_list)} 张有效图片")
        return imageinfo_list

    def _find_context(self, image_name: str, md_content: str, img_content_length: int) -> ImageContext | None:
        """
            根据图片名称，从md_context中找图片位置。找到图片上文和下文，找到向上标题，封装ImageContext返回。找不到则返回None
        """
        # 获取md的每个行列表。  包含空行，图片行，文本行
        line_list: List[str] = md_content.split("\n")
        # line_list[100:150]  # 用于截取 上文或下文，只要找到 向上标题索引，图片索引，向下标题索引
        for idx, line in enumerate(line_list):

            if re.search(r"^!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)", line):
                # 找向上标题和标题索引
                heading, pre_index = self._find_heading_above(idx, line_list)
                pre_text_list = line_list[pre_index + 1:idx]
                # 找向下标题索引。向下标题不需要。与当前图片关系不大。
                post_index = self._find_heading_below(idx, line_list)
                post_text_list = line_list[idx + 1:post_index]

                pre_text: str = self._extract_limited_context(pre_text_list, img_content_length,
                                                              direction="front")  # 两次逆序
                post_text: str = self._extract_limited_context(post_text_list, img_content_length,
                                                               direction="end")  # 不需要逆序

                return ImageContext(
                    heading=heading,
                    pre_text=pre_text,
                    post_text=post_text
                )

        return None

    def _find_heading_above(self, idx, line_list) -> tuple[str, int]:
        """从图片索引位置，向上找标题及标题索引"""
        for i in range(idx - 1, -1, -1):
            line = line_list[i]
            if re.match(r"^\s*#{1,6}\s+", line):
                return line.strip(), i
        return "", -1

    def _find_heading_below(self, idx, line_list):
        """从图片索引位置，向下找标题索引"""
        for i in range(idx + 1, len(line_list), 1):
            line = line_list[i]
            if re.match(r"^\s*#{1,6}\s+", line):
                return i
        return len(line_list)

    def _extract_limited_context(self, context_list: List[str], img_content_length: int, direction: str):
        """
        处理图片的上文或下文：只要指定长度的上下文的文本。  贪心算法
        """
        current_paragraph: List[str] = []  # 装填段落的行列表    ->  “\n”.join(current_paragraph) 段落字符串
        paragraphs: List[str] = []  # 存放所有段落列表
        for line in context_list:
            is_blank_line = not line.strip()
            is_other_image = re.match(r"!\[.*?\]\(.*?\)", line)
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []  # 重置
                continue
            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 贪心算法，找到最接近的段落
        # 第一次逆序。  贪心算法 累计 段落，找离图片最近段落。一定大于img_content_length
        if direction == "front":
            paragraphs.reverse()

        # 用于判断累加长度
        total: int = 0
        selected: List[str] = []  # 需要累加的段落

        # 从里图片最近段落 -> 最远段落      找符合长度段落
        for paragraph in paragraphs:

            if total + len(paragraph) >= img_content_length and selected:  # 避免第一个段落大于阈值
                # if total >= img_content_length  and selected:  #   img_content_length万一是负数
                break

            total += len(paragraph)
            selected.append(paragraph)

        # 拼接段落 作为上文  即 LLM 提示词时，必须按照段落原有顺序拼接。
        if direction == "front":
            selected.reverse()

        # 拼接最终上文 或 下文
        return "\n\n".join(selected)


# ------3.VLM 生成摘要----------------
class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def summarize_all(self, file_title, imageinfo_list, vl_model, requests_per_minute) -> Dict[str, str]:
        """
        调用VLM视觉语言模型，为每个图片生成摘要
        {
            '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）',
            '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示',
            '115adcddd73aeacbccd21861a542e8c23f78937f8680317548ea8393bcb0801b.jpg': '中文说明书标识',
            'de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg': '万用表电阻测量接线示意图'
        }
        """
        self.logger.info("【step_3】提取图片摘要")

        summarizes: Dict[str, str] = {}

        # 双端队列，用于存放每个请求的当前时间戳    1970-1-1 0:0:0   ~   现在时间
        requests_timestamp: Deque[float] = Deque()

        try:
            client = AIClients.get_openai()
        except Exception as e:
            # 全局降级处理。 无法或客户端，无法为图片生成摘要时，我们降级处理，为每个图片生成默认摘要
            self.logger.warning(
                f"VLM 不可用，跳过图片摘要生成: {e}"
            )
            for image_info in imageinfo_list:
                summarizes[image_info.name] = "默认摘要"

            return summarizes

        for image_info in imageinfo_list:
            # 速率控制   - 滑动窗口限流算法
            self._enforce_rate_limit(requests_timestamp, requests_per_minute)  # 默认时间窗口60秒

            summarizes[image_info.name] = self.summarize_one(file_title, image_info, vl_model, client)
        self.logger.info(f"生成 {len(summarizes)} 张图片摘要")
        return summarizes

    def summarize_one(self, file_title: str, image_info: ImageInfo, vl_model: str, client: OpenAI) -> str:
        """
        调用VLM为当前图片生成摘要
        """

        parts = [p for p in (image_info.context.heading, image_info.context.pre_text, image_info.context.post_text) if
                 p]

        final_context = "\n".join(parts)

        #  r    read — 只读模式打开文件
        #  b    binary — 以二进制方式读取
        with open(image_info.path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{file_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )

            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {image_info.path}: {e}")
            # 局部降级处理。
            return "默认摘要"

    def _enforce_rate_limit(self, requests_timestamp: Deque[float], requests_per_minute: int = 3, window: int = 60):
        """限流函数   时间滑动窗口"""
        now = time.time()

        # 1.每次请求VLM大模型前，移除窗口外(>60秒)的时间戳
        while requests_timestamp and now - requests_timestamp[0] > window:
            requests_timestamp.popleft()  # 进入队列最早元素，已经超过时间窗口，移除

        # 2.检查窗口内请求数是否大于上限阈值
        if len(requests_timestamp) >= requests_per_minute:
            # 3.达到上限,等待最早请求过期。
            sleep_dur = window - (now - requests_timestamp[0])  # 剩余时间
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)

            now = time.time()
            # 再次清理超过时间窗口的请求时间戳
            while requests_timestamp and now - requests_timestamp[0] > window:
                requests_timestamp.popleft()

                # 4.记录本次请求时间戳
        requests_timestamp.append(now)  # 往队列里增加当前请求的时间戳


# ------4.图片上传 & 替换----------------
class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, file_title, md_content, imageinfo_list, summarizes: Dict[str, str], minio_bucket,
                           endpoint_minio_url) -> str:
        """
           {'01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': 'http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg',
        '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': 'http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg',}
        """
        self.logger.info("【step_4】上传图片到MinIO并更新MD")
        image_name_urls: Dict[str, str] = {}

        # 1.上传所有图片，返回  图片名称和URL  字典
        image_name_urls: Dict[str, str] = self._upload_all(file_title, imageinfo_list, minio_bucket, endpoint_minio_url)

        # 替换MD中图片的摘要和地址
        new_md_content = self._replace_in_md(md_content, summarizes, image_name_urls)

        # 返回新MD内容。处理图片摘要和地址
        return new_md_content

    def _upload_all(self, file_title: str, imageinfo_list: List[ImageInfo], minio_bucket: str,
                    endpoint_minio_url: str) -> Dict[str, str]:
        """
        上传所有图片，返回图片名称和URL字典
        """
        image_name_urls: Dict[str, str] = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning(
                f"MinIO 不可用，所有图片保留本地路径: {e}"
            )
            # 全局降级处理   所有图片地址保留原地址
            for image_info in imageinfo_list:
                image_name_urls[image_info.name] = image_info.path
            return image_name_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=f"{file_title}/{image_info.name}",  # Object名称含路径
                    file_path=image_info.path,  # 上传的本地文件路径
                    content_type="image/jpeg"  # MIME类型
                )
                self.logger.info(f"{image_info.name} 上传成功")
                image_name_urls[image_info.name] = f"{endpoint_minio_url}/{minio_bucket}/{file_title}/{image_info.name}"
            except Exception as e:
                self.logger.warning(
                    f"{image_info.name} 上传失败，保留本地路径"
                )
                # 局部降级处理
                image_name_urls[image_info.name] = image_info.path
        self.logger.info(
            f"成功上传 {len(image_name_urls)} 张图片到 MinIO"
        )
        return image_name_urls

    @staticmethod
    def _replace_in_md(
            md_content: str,
            summaries: Dict[str, str],
            remote_urls: Dict[str, str],
    ) -> str:
        """替换 MD 中的图片引用为远程 URL + 摘要。
        {
            '01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': '万用表RS-12直流电流测量接线示意图（10A档位）',
            '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': '蜂鸣器功能符号指示',
        }

        {'01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg': 'http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg',
        '10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg': 'http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/10d2f007e02047a07d46e75a81db7f96811916c0f5ff662fa23ce215dadcbbe1.jpg',}

        """
        #  (.*?) 匹配组    表达式中含两个匹配值
        #  第一个匹配组： match.group(1)     =>  中括号内容    "默认"
        #  第二个匹配组：match.group(2)      =>   小括号内容    "images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg"
        #  返回匹配原文： match.group(0)     =>    ![默认](images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg)
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        # ![默认](images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg)
        # ![万用表RS-12直流电流测量接线示意图（10A档位）](http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)
        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(
                original_path).name  # e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg
            for img_name, summary in summaries.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)  # 匹配正则表达式，则调用replacer函数，替换图片的摘要和地址


class MarkDownImageNode(BaseNode):
    """MD中图片的处理节点"""

    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """MD中图片的处理流程
            借助于四个辅助类，完成业务开发。
        """

        # 1.读取md文件内容
        md_content, md_path_obj, image_path_obj = self.md_file_handler.read_md(state)
        if not image_path_obj.exists():
            state['md_content'] = md_content
            return state

        # 2.图片扫描
        imageinfo_list: List[ImageInfo] = self.image_scanner.scan_img_dir(image_path_obj,
                                                                          md_content,
                                                                          self.config.image_extensions,
                                                                          self.config.img_content_length)

        # 3.VLM 生成摘要  qwen3-vl-flash
        summarizes: dict[str, str] = self.vlm_summarizer.summarize_all(state['file_title'], imageinfo_list,
                                                                       self.config.vl_model,
                                                                       self.config.requests_per_minute)

        # 4.上传图片到Minio & 替换( 摘要+URL)
        new_md_content: str = self.image_uploader.upload_and_replace(
            state['file_title'],  # 作为上传图片的父路径名称
            md_content,
            imageinfo_list,
            summarizes,
            self.config.minio_bucket,
            self.config.get_minio_base_url()  # http://192.168.6.170:9000
        )

        # 5.新md文件保存
        self.md_file_handler.backup(md_path_obj, new_md_content)

        # 6.更新state并返回结果数据
        state['md_content'] = new_md_content

        return state


if __name__ == "__main__":
    setup_logging()

    node = MarkDownImageNode()
    state = {
        "file_title": "hak180产品安全手册",
        "md_path": r"D:\workspace\workspaceAI260706\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto\hak180产品安全手册.md"
    }
    node.process(state)

"""
2026-08-22 17:15:14 - knowledge.utils.client.base - INFO - OpenAI API 创建成功:https://dashscope.aliyuncs.com/compatible-mode/v1
2026-08-22 17:15:16 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:16 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:16 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:17 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:17 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:18 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:18 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-22 17:15:18 - knowledge.utils.client.storage_clients - INFO - 自动创建存储桶:knowledge-base-files
2026-08-22 17:15:18 - knowledge.utils.client.storage_clients - INFO - MinIO 配置完成
2026-08-22 17:15:18 - import.md_img_node - INFO - 【step_5】备份新文件
2026-08-22 17:15:18 - import.md_img_node - INFO - 处理后的文件已备份至: D:\workspace\workspaceAI260706\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto\hak180产品安全手册_new.md

"""