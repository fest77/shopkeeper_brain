"""

    导入流程的 商品名识别节点

"""
import json
import os
from typing import Tuple, List, Dict


from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    """商品名识别节点"""

    name: str = "item_name_rec_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """商品名识别节点处理流程"""

        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size= self._validate_state(state)

        # 2. 构建商品名识别上下文
        item_name_recognition_context:str = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别
        item_name:str = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名
        dense_vector: List[float]
        sparse_vector: Dict[int, float]
        dense_vector,sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                            self.config.item_name_collection)

        # 6. 回填item_name信息
        self._fill_item_name(item_name, state, chunks)

        # 7.备份，给下个节点准备下测试数据。
        self._backup_chunks(state, chunks)

        return state

    def _validate_state(self, state: ImportGraphState) ->Tuple[str,List,int,int]:
        # 参数校验
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        item_name_chunks_k = self.config.item_name_chunk_k
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title, chunks, item_name_chunks_k, item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks, item_name_chunks_k, item_name_chunk_size)->str:
        # 准备识别商品名的上下文(前k个chunk)
        total = 0
        final_context = []

        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk, dict):
                continue

            chunk_content = chunk.get('content')
            context = f"【切片】-{index}-{chunk_content}"

            if total + len(context) > item_name_chunk_size:
                break
                
            total += len(context)
            final_context.append(context)

        return "\n".join(final_context)

    def _recognition_name(self, file_title, item_name_recognition_context)->str:
        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title, context=item_name_recognition_context
            )

            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            llm_result = llm_response.content.strip()
            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return json.loads(llm_result).get("item_name")
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title


    def _embedding_item_name(self, item_name)-> Tuple[List[float], Dict[int, float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])

            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None


    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=collection_name,schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk['item_name'] = item_name
        state['item_name'] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将回填item_name的切片列表结果备份到json文件，给下个节点单元测试使用。
        :param state:
        :param chunks:
        :return:
        """
        local_dir = state.get("file_dir", "")
        os.makedirs(local_dir, exist_ok=True) #exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\A_Py_Java\pyFile\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_dir": r"D:\A_Py_Java\pyFile\shopkeeper_brain\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto",
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node(state)

    # # 5. 输出结果
    # print(f"商品名: {result.get('item_name')}")
    # print(f"chunks数量: {len(result.get('chunks', []))}")
    # print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")
    from rich import print as uprint
    uprint(result)

"""
{
    'file_dir': 
'D:\A_Py_Java\pyFile\shopkeeper_brain\output\5d8cd12b-c657-4e47
-aa4f-d69b281d1f74\hak180产品安全手册\auto',
    'file_title': '万用表的使用',
    'chunks': [
        {
            'title': 'hak180产品安全手册',
            'file_title': 'hak180产品安全手册',
            'parent_title': 'hak180产品安全手册',
            'content': 'hak180产品安全手册\n\n![D01WD7001-00 
型号产品条形码标识](http://192.168.6.170:9000/knowledge-base-files/hak180产品安
全手册/f93ddcd837fa399f0c165da43b0badb013c2961acc828a095c6d98f0a660dbca.jpg)\n\
nD01WD7001-00\n\nSCHN\n\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## HAK 180 烫金机- 1',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## HAK 180 烫金机',
            'content': '## HAK 180 烫金机- 
1\n\n\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 
烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，
请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 
(中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 
上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排
除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南
中规定的说明操作而导致的损害，Brother 
不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 
呼叫中心或您当地的Brother 
经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，
然后联系 Brother 呼叫中心或您当地的 Brother 
经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经
授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备
制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第
三方提出的任何索赔，我们不承担任何责任。\n',
            'part': 1,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 警告- 1',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 警告',
            'content': '## 警告- 1\n\n\n\n',
            'part': 1,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '### 警告- 1',
            'file_title': 'hak180产品安全手册',
            'parent_title': '### 警告',
            'content': '### 警告- 
1\n\n#####\n不遵守说明和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟
、发热、爆炸、火灾或人员受伤的风险。\n\n不遵守说明和警告可能导致人员死亡或严重
受伤。遵守这些指引以避免冒烟、发热、爆炸、火灾或人员受伤的风险。\n\n不遵守说明
和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟、发热、爆炸、火灾或人
员受伤的风险。\n',
            'part': 1,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备   -1- 1',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备   ',
            'content': '## 设备   -1- 
1\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操
作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对
于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 
不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内
部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本
设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您
接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请
联系 Brother 呼叫中心或您当地的 Brother 
经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 
之间，湿度保持在 20% 和 80% 
之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气
或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/ 
或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则
当水（包括加热/空调/通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风
险。\n\n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上
倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的
Brother 
经销商。\n\n•\t如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备
的插头，然后联系 Brother 呼叫中心或您当地的 Brother 
经销商。\n\n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设
备的插头，然后联系 Brother 呼叫中心或您当地的 
Brother经销商。\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。
纸张与定影单元长时间接触可能导致火灾。\n\n•\t请勿使用任何易燃物品、任何类型的喷
雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请
改用无绒干抹布。有关如何清洁本设备的说明，请参阅使用说明书。',
            'part': 1,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备   -2- 2',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备   ',
            'content': '## 设备   -2- 
2\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置
。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油
漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电
的风险。这些化学品或其他化学品可能导致本设备故障或褪色。\n\n•\t本设备的包装中使
用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，
并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。
如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t
使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿
触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件
。\n\n![禁止触摸高温区域及错误操作警示图](http://192.168.6.170:9000/knowledge-b
ase-files/hak180产品安全手册/6d9684e7665e2c54275f875898b5b15438927850755a836934
072bf10365504b.jpg)\n儎\u245fഴḽ䆜\u0a80ᛞ࠽व䀜\u1aae儎\u245fⲺ䇴༽䜞ԬȾ\n\n![打开设
备前盖时勿触碰灰色标记的高温部件](http://192.168.6.170:9000/knowledge-base-file
s/hak180产品安全手册/5046f02e6d9af4138369cffda93e78f900bff96c6b0e26d54c514dddf1
42fc4c.jpg)',
            'part': 2,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 电源线',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 电源线',
            'content': '## 电源线\n\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 
电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或
触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类
型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏
，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再
清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电
或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人
们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，
否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏
，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏/ 
磨损的部分。\n\n•\t请勿让设备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备
。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾
或人员受伤。必须按照使用说明书正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液
体落在设备的电源插头上。否则可能导致触电或火灾。\n\n•\tBrother 
强烈建议您不要使用任何类型的延长线。\n\n•\t定期拔出电源插头进行清洁。使用干布清
洁插头插脚根部以及插脚之间的位置。如果电源插头长时间插入在电源插座中，灰尘会堆
积在插头插脚周围，这可能会导致短路，从而引起火灾。\n\n•\t本设备装有接地的插头。
此插头只能插入接地的电源插座中。这是一项安全功能。如果您无法将插头插入到插座中
，请让电工更换过时的插座。请勿试图破坏接地插头的作用。\n\n不遵守说明和警告可能
导致人员中度或严重受伤。\n遵守这些指引以避免人员受伤。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备',
            'content': '## 
设备\n\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。
\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避
免将手指放置在图中所示的区域中。\n\n![禁止将手指伸入设备内部传动部件区域](http:
//192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/682b4624098519d3b6
3c2ff3d5a6e69d01f4c4400117da3d150534d8a8da92e7.jpg)\n\n![禁止将手指伸入设备进纸
口区域](http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/5d0bd
e7d6dd06c1aefbd41f70a27e569444a16325a5c3d4c2263b4e90d0adfc9.jpg)\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 电源线',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 电源线',
            'content': '## 
电源线\n\n\n•\t如果您长时间不会使用本设备，请从电源插座中拔掉电源线以确保安全。
\n\n•\t本设备必须安装在可轻松使用电源插座的位置附近。如果发生意外情况，必须从电
源插座中拔掉电源线以完全关闭电源。\n\n•\t请勿将手放在纸张边缘。纸张锋利的边缘可
能导致受伤。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 为设备选择一个安全的位置',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 为设备选择一个安全的位置',
            'content': '## 
为设备选择一个安全的位置\n\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果
抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运
本设备。\n\n![正确与错误的设备搬运方式示意图](http://192.168.6.170:9000/knowled
ge-base-files/hak180产品安全手册/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d
23b8fad0ad72fe18d0.jpg)\n\n确保本设备的任何部位均未伸出设备所在的桌面或支架。特
别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平
且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人
身伤害以及设备严重损坏。\n\n![禁止将设备置于桌面边缘且出纸盒打开的危险摆放方式]
(http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/656491d40001
e15ccb22f229934e447a274682d5e18e70aa11b9b686960a5063.jpg)\n\n“重要事项”表示可能
导致财产损失或本设备功能丧失的潜在危险情况。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备',
            'content': '## 
设备\n\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中
涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本
设备恢复到正常操作。Brother不建议使用 Brother 
正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备
的任何零件，由此导致的任何维修可能不在保修范围内。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 电源线',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 电源线',
            'content': '## 
电源线\n\n\n请勿将设备连接到受墙壁开关或自动计时器控制的电源插座，或者与大型设
备或需要大量电力的其他设备连接到同一个电路中。否则可能会损坏电源。电源损坏还可
能会从本设备的内存中删除信息，并且反复打开/关闭电源可能会损坏本设备。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 警告标签',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 警告标签',
            'content': '## 警告标签\n\n\n请勿撕下或损坏设备上的任何注意 
/警告标签以及序列号标签。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备保修和责任',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备保修和责任',
            'content': '## 
设备保修和责任\n\n\n本手册中的任何内容都将不会影响现有设备保修，也不应被视为授
予任何其他设备保修。不遵循本手册中的安全说明可能导致本设备的保修失效。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 设备和电源线',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 设备和电源线',
            'content': '## 
设备和电源线\n\n\n•\t请仅使用本设备随附的电源线。\n\n•\t不要在本设备周围放置任
何物体。在紧急情况下，此类物体会阻碍接近电源插座。必须保证在需要时可以拔出设备
的插头。\n\n•\t请遵守所有适用法规来处理本设备。\n',
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 产品中有害物质的名称及含量-1- 1',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 产品中有害物质的名称及含量',
            'content': '## 产品中有害物质的名称及含量-1- 1\n\n- 
【HAK180】(对应型号)：有害物质为铅，有害物质为汞，有害物质为镉，有害物质为六价
铬，有害物质为多溴联苯，有害物质为多溴二苯醚。\n- 
【部件名称】(对应型号)：有害物质为(Pb)，有害物质为(Hg)，有害物质为(Cd)，有害物
质为(Cr(VI))，有害物质为(PBB)，有害物质为(PBDE)。\n- 
【框架L单元】(对应型号)：有害物质为X，有害物质为O，有害物质为O，有害物质为O，有
害物质为O，有害物质为O。\n- 【框架 R 
单元】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，有害物质
为O，有害物质为O。\n- 
【中框架单元】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，
有害物质为O，有害物质为O。\n- 
【框架】(对应型号)：有害物质为X，有害物质为O，有害物质为O，有害物质为O，有害物
质为O，有害物质为O。\n- 
【顶盖单元】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，有
害物质为O，有害物质为0。\n- 
【进纸器单元】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，
有害物质为O，有害物质为O。\n- 
【热熔器】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，有害
物质为O，有害物质为O。\n- 
【盖板】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，有害物
质为O，有害物质为O。\n- 
【标签】(对应型号)：有害物质为O，有害物质为O，有害物质为O，有害物质为O，有害物
质为O，有害物质为0。\n- 
【金属薄片保持单元】(对应型号)：有害物质为O，有害物质为O，有害物质为O，有害物质
为O，有害物质为O，有害物质为O。\n- 
【主电路板】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为O，有
害物质为O，有害物质为O。\n- 
【低压电源电路板】(对应型号)：有害物质为×，有害物质为O，有害物质为O，有害物质为
O，有害物质为O，有害物质为0。\n- 
【选配件】(对应型号)：有害物质为O，有害物质为O，有害物质为O，有害物质为O，有害
物质为O，有害物质为O。\n- 
【包装材料】(对应型号)：有害物质为O，有害物质为O，有害物质为O，有害物质为O，有
害物质为O，有害物质为O。',
            'part': 1,
            'item_name': 'HAK 180 烫金机'
        },
        {
            'title': '## 产品中有害物质的名称及含量-2- 2',
            'file_title': 'hak180产品安全手册',
            'parent_title': '## 产品中有害物质的名称及含量',
            'content': '## 产品中有害物质的名称及含量-2- 2\n\n本表格依据 SJ/T 
11364 
的规定编制。\n\n○：表示该有害物质在该部件所有均质材料中的含量均在GB/T26572 
规定的限量要求以下。\n\n×：表示该有害物质至少在该部件的某一均质材料中的含量超出
GB/T 26572 规定的限量要求。\n\n（由于技术的原因暂时无法实现替代或减量化）',
            'part': 2,
            'item_name': 'HAK 180 烫金机'
        }
    ],
    'item_name': 'HAK 180 烫金机'
}

"""