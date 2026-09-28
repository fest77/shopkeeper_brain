from bs4 import BeautifulSoup

def html_table_to_grid(html: str):
    """将 HTML 表格（含 rowspan/colspan）展开为规整的二维数组"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    # 1. 计算表格的实际列数
    num_rows = len(rows)
    num_cols = max(
        sum(int(td.get("colspan", 1)) for td in row.find_all("td"))
        for row in rows
    )

    # 2. 创建空的二维数组
    grid = [[None] * num_cols for _ in range(num_rows)]

    # 3. 遍历每个单元格，处理 rowspan/colspan 的物理填充
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for td in row.find_all("td"):
            # 找到当前行中第一个空位
            while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1

            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            text = td.get_text(strip=True)

            # 向下、向右物理填充
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < num_rows and col_idx + c < num_cols:
                        grid[row_idx + r][col_idx + c] = text

            col_idx += colspan

    return grid


html = """
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V 电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2 至 8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V 电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22 至 1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>
"""

grid = html_table_to_grid(html)
print(grid)

"""
[['', '良好', '较弱', '坏的'], ['9V 电池：', '>8.2V', '7.2 至 8.2V', '<7.2V'], ['1.5V 电池：', '>1.35V', '1.22 至 1.35V', '<1.22V']]
"""