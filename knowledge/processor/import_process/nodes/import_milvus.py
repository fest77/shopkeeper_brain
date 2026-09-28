"""

    导入流程的 保存数据到Milvus数据库节点

"""
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence, Dict, Any

from pymilvus import MilvusClient, DataType, CollectionSchema

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


# ================================================================== #
#                        标量字段规范                                   #
# ================================================================== #

@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段规范"""
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 预定义的标量字段（复用）
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


# ================================================================== #
#                        建造者：Schema 构建                           #
# ================================================================== #

class _MilvusSchemaBuilder:
    """职责：专门负责构建约束"""

    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建约束(schema)...")

        # 1. 构建约束对象(动态映射)
        schema = client.create_schema(enable_dynamic_field=True)

        # 2. 构建主键字段约束
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 3. 构建向量字段约束
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 4. 构建标量字段约束
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length
            schema.add_field(**kwargs)

        logger.info(f"构建约束(schema)完成...")
        return schema


# ================================================================== #
#                        建造者：索引构建                               #
# ================================================================== #

class _MilvusIndexBuilder:
    """职责：负责处理Milvus的索引"""

    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")

        index = client.prepare_index_params(collection_name=collection_name)

        # 稠密向量索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 稀疏向量索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        logger.info(f"构建集合 {collection_name} 索引完成...")
        return index




# ================================================================== #
#                        插入器：数据插入与回填                          #
# ================================================================== #

class _MilvusInserter:
    """职责：将数据插入到Milvus 以及 回填chunk_id"""

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[dict[str, Any]]:
        logger.info(f"开始插入{len(chunks)}块到Milvus...")

        inserted_result = self._client.insert(
            collection_name=self._collection_name,
            data=chunks
        )
        inserted_count = inserted_result.get('insert_count')
        ids = inserted_result.get('ids')

        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}记录,并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id



class ImportMilvusNode(BaseNode):

    """保存数据到Milvus数据库节点"""

    name: str = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState|dict :
        """保存数据到Milvus数据库节点任务处理"""

        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)

        # 2. 获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()

        if milvus_client is None:
            return state

        # 3. 获取集合名字
        collection = getattr(config, 'chunks_collection')

        # 4. 确保集合存在
        #self._ensure_has_collection(milvus_client, collection, dim,True)
        self._ensure_has_collection(milvus_client, collection, dim)

        # 5. 插入
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        final_chunks = inserter.insert(chunks=validated_chunks)

        # 6. 更新state
        state['chunks'] = final_chunks

        return state



    def _validate_get_inputs(self, state: ImportGraphState)-> Tuple[List,int,ImportConfig]:
        """参数校验"""
        self.log_step("step1", "参数校验")

        config = get_config()
        chunks = state.get('chunks')

        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        validated_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在")

        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        dim = len(validated_chunks[0].get('dense_vector'))
        self.logger.info(f"导入Milvus向量数据库的有效块：{len(validated_chunks)},且chunk的向量维度{dim}")

        return validated_chunks, dim, config


    def _ensure_has_collection(self, milvus_client: MilvusClient, collection_name: str,
                               dim: int, delete_flag: bool = False):
        """确保集合存在"""
        self.log_step("step2", f"准备集合 {collection_name} 创建")

        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"Milvus中的集合 {collection_name}已被删除")
            milvus_client.drop_collection(collection_name=collection_name)

        if milvus_client.has_collection(collection_name=collection_name):
            return

        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index
        )

from pathlib import Path
import json

if __name__ == "__main__":
    setup_logging()

    temp_dir = Path(r"D:\A_Py_Java\pyFile\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

        state: ImportGraphState = {
            "chunks": content.get("chunks", [])
        }

        import_milvus = ImportMilvusNode()
        result_state = import_milvus(state)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result_state, fh, ensure_ascii=False, indent=4)

            logger.info(f"备份临时文件{output_path}成功")