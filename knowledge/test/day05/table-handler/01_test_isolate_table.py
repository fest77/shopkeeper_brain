import re

def isolate_tables(md_content: str):
    """将 Markdown 表格从正文中隔离出来"""
    # 匹配标准 Markdown 表格（以 | 开头的连续行）
    table_pattern = re.compile(r'(\n?(?:\|[^\n]+\|\n)+)', re.MULTILINE)

    tables = []
    text_parts = []
    last_end = 0

    for match in table_pattern.finditer(md_content):
        # 收集表格前的正文
        text_parts.append(md_content[last_end:match.start()])
        # 收集表格（作为独立 Chunk）
        tables.append(match.group())
        last_end = match.end()

    # 收集最后一段正文
    text_parts.append(md_content[last_end:])

    return text_parts, tables  # 正文送切分器，表格直接作为独立 Chunk


table_str = """
| 功能     | 量程   | 精确度              |
|---------|--------|---------------------|
| 直流电压 | 200mV  | ± (0.5% + 2 digits) |
| 直流电压 | 20V    | ± (0.5% + 2 digits) |
| 交流电压 | 600V   | ± (1.2% + 10 digits)|
"""
text_parts, tables = isolate_tables(table_str)
print(table_str)
print(tables)


"""
| 功能     | 量程   | 精确度              |
|----------|--------|---------------------|
| 直流电压 | 200mV  | ± (0.5% + 2 digits) |
| 直流电压 | 20V    | ± (0.5% + 2 digits) |
| 交流电压 | 600V   | ± (1.2% + 10 digits)|

['\n| 功能     | 量程   | 精确度              |\n|----------|--------|---------------------|\n| 直流电压 | 200mV  | ± (0.5% + 2 digits) |\n| 直流电压 | 20V    | ± (0.5% + 2 digits) |\n| 交流电压 | 600V   | ± (1.2% + 10 digits)|\n']
"""