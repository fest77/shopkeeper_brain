"""

    导入流程的入口节点：

        判断上传文件是 pdf 还是  md  走后续不同的流程步骤

"""
import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):

    """入口节点"""

    name: str = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState|dict :
        """入口节点处理流程"""
        import_file_path = state.get('import_file_path')  # 检查输入的源文件路径
        file_dir = state.get('file_dir')  # 检查处理后的结果放在哪个目录

        if not import_file_path or not file_dir:
            raise ValidationError("数据校验失败",self.name)

        import_file_path_obj = Path(import_file_path)

        suffix = import_file_path_obj.suffix.lower()
        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            raise ValidationError("文件类型暂不支持",self.name)

        filte_title = import_file_path_obj.stem
        state['file_title'] = filte_title

        return state



#测试
if __name__ == "__main__":
    setup_logging()

    input_state = {
        "import_file_path": r"D:\Asgg_class\a课堂资料\4阶段四_git_Langchain_张宇8.8-8\项目【掌柜智库】_8.18\软件\pdf文档\doc\万用表RS-12的使用.pdf",  # 导入文件路径（原始输入）
        "file_dir": r"E:\Atemp_dir",  # 导入(出)文件目录
    }

    entry = EntryNode()

    #node_state = entry.process(input_state)  # 不父类执行__call__()
    #node_state =  entry.__call__(input_state)  # 调用父类__call__()
    node_state =  entry(input_state)  # 自动调用父类__call__()

    print(json.dumps(node_state,ensure_ascii=False,indent=4))

    """
    {
        "import_file_path": "E:\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "E:\\temp_dir",
        "is_pdf_read_enabled": true,
        "pdf_path": "E:\\doc\\万用表RS-12的使用.pdf",
        "file_title": "万用表RS-12的使用"
    }
    """