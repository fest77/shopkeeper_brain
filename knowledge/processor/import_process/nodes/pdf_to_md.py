
    #
    # 导入流程的 PDF转换Markdown节点
    #
    #     1.指定模型下载位置：
    #
    #         搬家： 如果不设置默认下载到  ~/.cache文件夹。
    #
    #         MODELSCOPE_CACHE=D:/ai_models/modelscope_cache
    #
    #     2.首次执行转换命令，自动下载模型：自动生成：C:\Users\zy\mineru.json
    #
    #     {
    #         "bucket_info": {
    #             "bucket-name-1": [
    #                 "ak",
    #                 "sk",
    #                 "endpoint"
    #             ],
    #             "bucket-name-2": [
    #                 "ak",
    #                 "sk",
    #                 "endpoint"
    #             ]
    #         },
    #         "latex-delimiter-config": {
    #             "display": {
    #                 "left": "$$",
    #                 "right": "$$"
    #             },
    #             "inline": {
    #                 "left": "$",
    #                 "right": "$"
    #             }
    #         },
    #         "llm-aided-config": {
    #             "title_aided": {
    #                 "api_key": "your_api_key",
    #                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    #                 "model": "qwen3.5-plus",
    #                 "enable_thinking": false,
    #                 "enable": false
    #             }
    #         },
    #         "models-dir": {
    #             "pipeline": "D:\\ai_models\\modelscope_cache\\models\\OpenDataLab--PDF-Extract-Kit-1.0\\snapshots\\master",
    #             "vlm": "D:\\ai_models\\modelscope_cache\\models\\OpenDataLab--MinerU2.5-Pro-2605-1.2B\\snapshots\\master"
    #         },
    #         "model-source": "modelscope",
    #         "config_version": "1.3.2"
    #     }

import json
import subprocess
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):

    """PDF转换Markdown节点"""

    name: str = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState|dict :
        """PDF转MarkDown处理流程"""

        #1.数据校验
        import_file_path_obj, file_dir_obj = self._validate_state_inputs_path(state)

        #2.执行mineru 转换
        code = self._execute_mineru(import_file_path_obj, file_dir_obj)
        if code!=0:
            raise PdfConversionError("PDF转换Markdown失败",self.name)

        #3.获取md路径
        md_path_obj = self._get_md_path(import_file_path_obj,file_dir_obj)


        #4.更新state
        state['md_path'] = str(md_path_obj)

        #5.返回
        return state

    def _validate_state_inputs_path(self, state) -> Tuple[Path,Path]:
        self.log_step("step1", "对状态的路径输入参数做校验")
        import_file_path =  state.get("import_file_path","")

        if not import_file_path:
            raise ValidationError("import_file_path参数不存在",self.name)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():  #是否为有效路径
            raise FileProcessingError("PDF文件不存在",self.name)

        file_dir =  state.get("file_dir","")
        if not file_dir:
            file_dir = import_file_path_obj.parent # 降级处理

        file_dir_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")
        return import_file_path_obj,file_dir_obj

    def _execute_mineru(self, import_file_path_obj, file_dir_obj) -> int:
        """利用mineru命令，将pdf转换为md"""
        #    mineru -p input.pdf -o output_dir/ --source local  --backend  pipeline
        #    pipeline 使用原始推理模型，完成PDF解析，无需使用GPU
        self.log_step("step2", "执行MinerU解析PDF")

        process_start_time = time.time()

        proc = subprocess.Popen(
            args=["mineru", "-p", str(import_file_path_obj), "-o", str(file_dir_obj), "--backend", "pipeline"],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")

        return_code = proc.wait()

        process_end_time = time.time()
        if return_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path_obj.name} "
                f"耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path_obj.name}失败")

        return return_code

    def _get_md_path(self, import_file_path_obj, file_dir_obj) -> Path:
        """获取pdf转换为md之后完整路径"""
        return file_dir_obj / import_file_path_obj.stem / "auto" / ( import_file_path_obj.stem + ".md")


#测试
if __name__ == "__main__":
    setup_logging()

    input_state = {
        "import_file_path": r"E:\doc\万用表RS-12的使用.pdf",  # 导入文件路径（原始输入）
        "file_dir": r"E:\temp_dir",  # 导入(出)文件目录
    }

    current_node = PdfToMdNode()

    node_state =  current_node(input_state)

    print(json.dumps(node_state,ensure_ascii=False,indent=4))


"""
{
    "import_file_path": "E:\\doc\\万用表RS-12的使用.pdf",
    "file_dir": "E:\\temp_dir",
    "md_path": "E:\\temp_dir\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md"
}

"""