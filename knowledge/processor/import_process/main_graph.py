"""
    导入流程主图
    使用 LangGraph 构建文档导入工作流
"""
import json

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def load_file_route(state: ImportGraphState):
    """根据上传文件走不同路由"""
    # state['is_pdf_read_enabled'] = False   赋值
    # if state['is_pdf_read_enabled']: #取值时属性不存在，会报KeyError
    if state.get("is_pdf_read_enabled",False):  #获取属性值不存在，会取默认值。
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled",False):
        return "md_img_node"
    else: #兜底程序   降级处理
        return "END"


def create_import_graph() -> CompiledStateGraph:
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ───┐
              │                               │
              └── (MD) ──────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """

    # 1.定义状态  state.py

    # 2.定义节点   参看 节点文件
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 3.创建图实例并循环添加节点
    import_graph = StateGraph(ImportGraphState)
    for key, node in nodes.items():
        import_graph.add_node(key, node)

    # 4.添加边
    import_graph.set_entry_point("entry_node")  # 设置入口节点

    import_graph.add_conditional_edges(
        "entry_node",
        load_file_route,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "md_img_node": "md_img_node",
            "END": END
        }
    )
    #import_graph.add_edge(START, "entry_node")  #省略
    import_graph.add_edge("pdf_to_md_node", "md_img_node")
    import_graph.add_edge("md_img_node", "document_split_node")
    import_graph.add_edge("document_split_node", "item_name_rec_node")
    import_graph.add_edge("item_name_rec_node", "bge_embedding_node")
    import_graph.add_edge("bge_embedding_node", "import_milvus_node")
    #import_graph.add_edge("import_milvus_node", END)  # 可省略

    # 5.编译节点
    compiled_graph = import_graph.compile()
    return compiled_graph


kb_import_process_graph = create_import_graph()

def run_import_graph(inputState: ImportGraphState) -> ImportGraphState:
    """
    inputState: {
            "import_file_path": r"E:\doc\万用表RS-12的使用.pdf",           # 导入文件路径（原始输入）
            "file_dir": r"E:\temp_dir"                                  # 导入(出)文件目录
    }
    """

    # 创建图输入 state
    init_state = create_default_state(
        import_file_path = inputState.get("import_file_path"),
        file_dir = inputState.get("file_dir")
    )

    # 运行图
    final_state = None
    for event in kb_import_process_graph.stream(inputState):  #stream_mode="updates"  增量结果
        for node_name, process_state in  event.items():
            #print(f"Node: {node_name}, Process_state: {process_state}")
            print(f"运行节点: {node_name}")
            final_state = process_state


    return final_state


# 测试
if __name__ == "__main__":

    setup_logging()

    # 打印图
    kb_import_process_graph.get_graph().print_ascii()

    input_state = {
        "import_file_path": r"D:\Asgg_class\a课堂资料\4阶段四_git_Langchain_张宇8.8-8\项目【掌柜智库】_8.18\软件\pdf文档\doc\万用表RS-12的使用.pdf",  # 导入文件路径（原始输入）
        "file_dir": r"E:\Atemp_dir",  # 导入(出)文件目录
        #"is_pdf_read_enabled": False,
        #"is_md_read_enabled": True
    }

    final_state = run_import_graph(input_state)

    print(json.dumps(final_state, indent=4, ensure_ascii=False))

"""
                                 +-----------+                        
                                 | __start__ |                        
                                 +-----------+                        
                                       *                              
                                       *                              
                                       *                              
                                +------------+                        
                                | entry_node |.                       
                           .....+------------+ .....                  
                      .....            .            .....             
                 .....                .                  ....         
              ...                     .                      .....    
+----------------+                   ..                           ... 
| pdf_to_md_node |                 ..                               . 
+----------------+               ..                                 . 
               ***            ...                                   . 
                  **        ..                                      . 
                    **    ..                                        . 
                +-------------+                                     . 
                | md_img_node |                                     . 
                +-------------+                                     . 
                        *                                           . 
                        *                                           . 
                        *                                           . 
            +---------------------+                                 . 
            | document_split_node |                                 . 
            +---------------------+                                 . 
                        *                                           . 
                        *                                           . 
                        *                                           . 
            +--------------------+                                  . 
            | item_name_rec_node |                                  . 
            +--------------------+                                  . 
                        *                                           . 
                        *                                           . 
                        *                                           . 
            +--------------------+                                  . 
            | bge_embedding_node |                                  . 
            +--------------------+                                  . 
                        *                                           . 
                        *                                           . 
                        *                                           . 
            +--------------------+                                ... 
            | import_milvus_node |                           .....    
            +--------------------+                       ....         
                               **                   .....             
                                 ***           .....                  
                                    **      ...                       
                                  +---------+                         
                                  | __end__ |                         
                                  +---------+                         

"""
