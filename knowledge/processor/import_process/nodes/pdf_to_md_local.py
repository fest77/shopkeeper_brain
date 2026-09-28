"""
对接本地 mineru-api 服务
启动方式：mineru-api --host 127.0.0.1 --port 8000
API文档：http://127.0.0.1:8000/docs

本地API接口：
    POST /tasks          — 异步任务提交（上传文件，立即返回task_id）
    GET  /tasks/{id}     — 轮询任务状态
    GET  /tasks/{id}/result — 获取解析结果（ZIP包）
    GET  /health         — 健康检查
"""
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


class PdfToMdLocalNode(BaseNode):
    """
    节点: PDF转Markdown (本地mineru-api版)
    将 PDF 非结构化数据转换为 Markdown 结构化数据。
    通过HTTP调用本地 mineru-api 服务完成解析。
    """

    name = "pdf_to_md_local_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        必要参数：pdf_path、local_dir
        更新参数：md_path、md_content
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：调用本地MinerU API同步解析，获取ZIP包
        zip_content = self._step_2_parse(pdf_path_obj)

        # 步骤3：解压ZIP包并提取MD文件
        md_path = self._step_3_extract(zip_content, output_dir_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1、参数非空校验
        pdf_path = state.get("pdf_path", "").strip()
        local_dir = state.get("local_dir", "").strip()
        if not pdf_path:
            raise ValueError("缺失参数：pdf_path")
        if not local_dir:
            raise ValueError("缺失参数：local_dir")

        # 2、转换为Path对象统一处理路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(local_dir)

        # 3、PDF文件有效性校验
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")

        # 4、确保输出目录存在，不存在则递归创建
        if not output_dir_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{output_dir_obj.absolute()}")
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, output_dir_obj

    def _step_2_parse(self, pdf_path_obj: Path) -> bytes:
        """
        步骤2：调用本地MinerU API同步解析接口 POST /file_parse
        核心流程：配置校验 → 提交同步解析请求 → 等待完成并返回ZIP字节
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：ZIP结果包的字节内容
        异常：ValueError(配置缺失)、RuntimeError(请求失败)
        """
        # 1、配置校验
        if not self.config.mineru_base_url:
            raise ValueError("MinerU配置缺失：请在 .env 文件中配置 MINERU_BASE_URL，例如：http://127.0.0.1:8000")
        self.logger.info(f"【配置校验】MinerU本地API配置校验成功，开始处理文件：{pdf_path_obj.name}")

        # 2、调用同步解析接口：POST /file_parse
        url = f"{self.config.mineru_base_url}/file_parse"
        self.logger.info(f"【同步解析】调用接口：{url}")

        start_time = time.time()

        with open(pdf_path_obj, "rb") as f:
            files = {"files": (pdf_path_obj.name, f, "application/pdf")}
            data = {
                "return_md": "true",      # 返回 Markdown
                "return_images": "true",  # ← 新增！返回图片
                "return_content_list": "true",  # ← 新增！返回结构化数据
                "response_format_zip": "true",  # 以 ZIP 格式返回
                "return_original_file": "true",  # 保留原始 PDF
                "backend": "pipeline", #基于CPU模型
            }
            response = requests.post(url, files=files, data=data, timeout=600)  #文件上传请求。post请求.

        elapsed_time = time.time() - start_time

        if response.status_code != 200:
            raise RuntimeError(
                f"【同步解析】请求失败：状态码：{response.status_code}，响应内容：{response.text}"
            )

        content_type = response.headers.get("content-type", "")
        self.logger.info(f"【同步解析】解析完成，耗时{int(elapsed_time)}s，响应类型：{content_type}，大小：{len(response.content) / 1024:.1f}KB")

        return response.content

    def _step_3_extract(self, zip_content: bytes, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：解压MinerU解析结果ZIP包，提取目标MD文件
        核心流程：保存ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_content-ZIP字节内容；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(解压失败)、FileNotFoundError(无MD文件)
        """

        # 1、保存ZIP包
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(zip_content)
        self.logger.info(f"【ZIP保存】保存路径：{zip_save_path}，大小：{len(zip_content) / 1024:.1f}KB")

        # 2、清空解压目录（ZIP内部已包含{pdf_stem}/前缀，直接解压到output_dir_obj即可）
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir, ignore_errors=True)
            self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3、直接解压到output_dir_obj（ZIP内自带{pdf_stem}/前缀，解压后自动形成 output_dir_obj/{pdf_stem}/auto/...）
        self.logger.info(f"【ZIP解压】开始解压ZIP包到：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(output_dir_obj)  #解压
            all_files = zip_file_obj.namelist()
        self.logger.info(f"【ZIP解压】解压完成")
        self.logger.info(f"【ZIP解压】ZIP内文件列表（共{len(all_files)}个）：")
        for f_name in all_files:
            self.logger.info(f"  - {f_name}")

        # 4、查找MD文件（ZIP内结构：{pdf_stem}/auto/{pdf_stem}.md）
        md_auto_dir = extract_target_dir / "auto"
        target_md_file = md_auto_dir / f"{pdf_stem}.md"

        if not target_md_file.exists():
            # 兜底：优先找full.md，再递归搜索任意.md
            full_md = md_auto_dir / "full.md"
            if full_md.exists():
                target_md_file = full_md
            else:
                md_files = list(extract_target_dir.rglob("*.md")) #Recursive Glob — 递归搜索所有匹配 *.md 的文件
                if not md_files:
                    raise FileNotFoundError(f"【MD查找】解压目录中未找到任何MD文件：{extract_target_dir}")
                target_md_file = md_files[0]
            self.logger.info(f"【MD查找】未找到预期路径，使用：{target_md_file}")

        self.logger.info(f"【MD查找】找到MD文件：{target_md_file}")

        # 5、检查images目录
        images_dir = target_md_file.parent / "images"
        if images_dir.exists():
            img_files = list(images_dir.iterdir())
            self.logger.info(f"【资源检查】images目录存在，共{len(img_files)}个图片文件 ✅")
            for img_file in img_files:
                self.logger.info(f"  - {img_file.name} ({img_file.stat().st_size / 1024:.1f}KB)")
        else:
            self.logger.warning(f"【资源检查】images目录不存在：{images_dir}，MD中的图片引用将无法显示 ")

        return str(target_md_file.absolute())
# ================================================================== #
#                        测试                                        #
# ================================================================== #
if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging

    setup_logging()

    pdf_path = os.path.join(r"E:\doc\hak180产品安全手册.pdf")
    local_dir = os.path.join(r"E:\doc\output2")

    init_state = create_default_state(
        task_id="task_001",
        pdf_path=pdf_path,
        local_dir=local_dir
    )

    node_pdf_to_md = PdfToMdLocalNode()
    final_state = node_pdf_to_md(init_state)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))



"""
D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\.venv\Scripts\python.exe D:\workspace\workspaceAI260706\shopkeeper_brain\knowledge\processor\import_process\nodes\pdf_to_md_local.py 
2026-08-20 21:01:17 - import.pdf_to_md_local_node - INFO - --- pdf_to_md_local_node 开始 ---
2026-08-20 21:01:17 - import.pdf_to_md_local_node - INFO - 输出目录不存在，自动创建：E:\doc\output2
2026-08-20 21:01:17 - import.pdf_to_md_local_node - INFO - 【配置校验】MinerU本地API配置校验成功，开始处理文件：hak180产品安全手册.pdf
2026-08-20 21:01:17 - import.pdf_to_md_local_node - INFO - 【同步解析】调用接口：http://127.0.0.1:8000/file_parse
2026-08-20 21:01:44 - import.pdf_to_md_local_node - INFO - 【同步解析】解析完成，耗时27s，响应类型：application/zip，大小：816.9KB
2026-08-20 21:01:44 - import.pdf_to_md_local_node - INFO - 【ZIP保存】保存路径：E:\doc\output2\hak180产品安全手册_result.zip，大小：816.9KB
2026-08-20 21:01:44 - import.pdf_to_md_local_node - INFO - 【ZIP解压】开始解压ZIP包到：E:\doc\output2 ...
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO - 【ZIP解压】解压完成
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO - 【ZIP解压】ZIP内文件列表（共12个）：
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/hak180产品安全手册.md
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/hak180产品安全手册_content_list.json
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/hak180产品安全手册_content_list_v2.json
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/3235bebe94722f45d984e70a5dc6463a2134cf5a1a92e9b8a4971685359106c0.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/5046f02e6d9af4138369cffda93e78f900bff96c6b0e26d54c514dddf142fc4c.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/5d0bde7d6dd06c1aefbd41f70a27e569444a16325a5c3d4c2263b4e90d0adfc9.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/656491d40001e15ccb22f229934e447a274682d5e18e70aa11b9b686960a5063.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/682b4624098519d3b63c2ff3d5a6e69d01f4c4400117da3d150534d8a8da92e7.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/6d9684e7665e2c54275f875898b5b15438927850755a836934072bf10365504b.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/images/f93ddcd837fa399f0c165da43b0badb013c2961acc828a095c6d98f0a660dbca.jpg
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - hak180产品安全手册/auto/hak180产品安全手册_origin.pdf
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO - 【MD查找】找到MD文件：E:\doc\output2\hak180产品安全手册\auto\hak180产品安全手册.md
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO - 【资源检查】images目录存在，共8个图片文件 ✅
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 3235bebe94722f45d984e70a5dc6463a2134cf5a1a92e9b8a4971685359106c0.jpg (103.4KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 5046f02e6d9af4138369cffda93e78f900bff96c6b0e26d54c514dddf142fc4c.jpg (13.2KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 5d0bde7d6dd06c1aefbd41f70a27e569444a16325a5c3d4c2263b4e90d0adfc9.jpg (17.1KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 656491d40001e15ccb22f229934e447a274682d5e18e70aa11b9b686960a5063.jpg (28.2KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 682b4624098519d3b63c2ff3d5a6e69d01f4c4400117da3d150534d8a8da92e7.jpg (28.9KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - 6d9684e7665e2c54275f875898b5b15438927850755a836934072bf10365504b.jpg (46.8KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg (32.1KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO -   - f93ddcd837fa399f0c165da43b0badb013c2961acc828a095c6d98f0a660dbca.jpg (9.0KB)
2026-08-20 21:01:45 - import.pdf_to_md_local_node - INFO - --- pdf_to_md_local_node 完成 ---
{
    "task_id": "task_001",
    "is_pdf_read_enabled": false,
    "is_md_read_enabled": false,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "E:\\doc\\hak180产品安全手册.pdf",
    "md_path": "E:\\doc\\output2\\hak180产品安全手册\\auto\\hak180产品安全手册.md",
    "file_title": "",
    "md_content": "![](images/f93ddcd837fa399f0c165da43b0badb013c2961acc828a095c6d98f0a660dbca.jpg)\n\nD01WD7001-00\n\nSCHN\n\n## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。\n\n## 警告\n\n不遵守说明和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟、发热、爆炸、火灾或人员受伤的风险。\n\n## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/ 或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热/空调/通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。\n\n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother经销商。\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。\n\n•\t请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅使用说明书。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。\n\n•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![](images/6d9684e7665e2c54275f875898b5b15438927850755a836934072bf10365504b.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ\n\n![](images/5046f02e6d9af4138369cffda93e78f900bff96c6b0e26d54c514dddf142fc4c.jpg)\n\n## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏/ 磨损的部分。\n\n•\t请勿让设备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照使用说明书正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。\n\n•\tBrother 强烈建议您不要使用任何类型的延长线。\n\n•\t定期拔出电源插头进行清洁。使用干布清洁插头插脚根部以及插脚之间的位置。如果电源插头长时间插入在电源插座中，灰尘会堆积在插头插脚周围，这可能会导致短路，从而引起火灾。\n\n•\t本设备装有接地的插头。此插头只能插入接地的电源插座中。这是一项安全功能。如果您无法将插头插入到插座中，请让电工更换过时的插座。请勿试图破坏接地插头的作用。\n\n不遵守说明和警告可能导致人员中度或严重受伤。  \n遵守这些指引以避免人员受伤。\n\n## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![](images/682b4624098519d3b63c2ff3d5a6e69d01f4c4400117da3d150534d8a8da92e7.jpg)\n\n![](images/5d0bde7d6dd06c1aefbd41f70a27e569444a16325a5c3d4c2263b4e90d0adfc9.jpg)\n\n## 电源线\n\n•\t如果您长时间不会使用本设备，请从电源插座中拔掉电源线以确保安全。\n\n•\t本设备必须安装在可轻松使用电源插座的位置附近。如果发生意外情况，必须从电源插座中拔掉电源线以完全关闭电源。\n\n•\t请勿将手放在纸张边缘。纸张锋利的边缘可能导致受伤。\n\n## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![](images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg)\n\n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。\n\n![](images/656491d40001e15ccb22f229934e447a274682d5e18e70aa11b9b686960a5063.jpg)\n\n“重要事项”表示可能导致财产损失或本设备功能丧失的潜在危险情况。\n\n## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n\n## 电源线\n\n请勿将设备连接到受墙壁开关或自动计时器控制的电源插座，或者与大型设备或需要大量电力的其他设备连接到同一个电路中。否则可能会损坏电源。电源损坏还可能会从本设备的内存中删除信息，并且反复打开/关闭电源可能会损坏本设备。\n\n## 警告标签\n\n请勿撕下或损坏设备上的任何注意 /警告标签以及序列号标签。\n\n## 设备保修和责任\n\n本手册中的任何内容都将不会影响现有设备保修，也不应被视为授予任何其他设备保修。不遵循本手册中的安全说明可能导致本设备的保修失效。\n\n## 设备和电源线\n\n•\t请仅使用本设备随附的电源线。\n\n•\t不要在本设备周围放置任何物体。在紧急情况下，此类物体会阻碍接近电源插座。必须保证在需要时可以拔出设备的插头。\n\n•\t请遵守所有适用法规来处理本设备。\n\n## 产品中有害物质的名称及含量\n\n<table><tr><td rowspan=1 colspan=1>型号</td><td rowspan=1 colspan=6>有害物质</td></tr><tr><td rowspan=1 colspan=1>HAK180</td><td rowspan=1 colspan=1>铅</td><td rowspan=1 colspan=1>汞</td><td rowspan=1 colspan=1>镉</td><td rowspan=1 colspan=1>六价铬</td><td rowspan=1 colspan=1>多溴联苯</td><td rowspan=1 colspan=1>多溴二苯醚</td></tr><tr><td rowspan=1 colspan=1>部件名称</td><td rowspan=1 colspan=1>(Pb)</td><td rowspan=1 colspan=1>(Hg)</td><td rowspan=1 colspan=1>(Cd)</td><td rowspan=1 colspan=1>(Cr(VI))</td><td rowspan=1 colspan=1>(PBB)</td><td rowspan=1 colspan=1>(PBDE)</td></tr><tr><td rowspan=1 colspan=1>框架L单元</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>框架 R 单元</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>中框架单元</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>框架</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>顶盖单元</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>进纸器单元</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>热熔器</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>盖板</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>标签</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>金属薄片保持单元</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>主电路板</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>低压电源电路板</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>选配件</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>包装材料</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td></tr></table>\n\n本表格依据 SJ/T 11364 的规定编制。\n\n○：表示该有害物质在该部件所有均质材料中的含量均在GB/T26572 规定的限量要求以下。\n\n×：表示该有害物质至少在该部件的某一均质材料中的含量超出GB/T 26572 规定的限量要求。\n\n（由于技术的原因暂时无法实现替代或减量化）",
    "chunks": [],
    "item_name": "",
    "local_dir": "E:\\doc\\output2"
}

"""