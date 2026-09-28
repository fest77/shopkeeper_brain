#pip install playwright --index-url https://pypi.tuna.tsinghua.edu.cn/simple
#playwright install   安装浏览器

from playwright.sync_api import sync_playwright


def html_to_image(html_content: str, output_path: str = "table2.png"):
    """将 HTML 表格渲染为图片"""
    # 包装成完整的 HTML 页面，添加样式让表格更清晰
    full_html = f"""
    <html><body style="padding:20px; font-family:Arial;">
    {html_content}
    </body></html>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # 无界面模式
        page = browser.new_page()
        page.set_content(full_html)  # 喂入 HTML
        page.screenshot(path=output_path)  # 截图保存
        browser.close()

    return output_path


html = """
<table border="1"><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V 电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2 至 8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V 电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22 至 1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>
"""
result = html_to_image(html)
print(result)
