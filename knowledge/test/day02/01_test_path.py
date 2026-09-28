from pathlib import Path

# 创建 Path 对象
path = Path("D:/documents/万用表的使用.pdf")

# 常用属性
path.name       # "万用表的使用.pdf" - 完整文件名
path.stem       # "万用表的使用" - 不含扩展名的文件名
path.suffix     # ".pdf" - 扩展名
path.parent     # Path("D:/documents") - 父目录

# 常用方法
path.exists()   # True/False - 文件是否存在
path.is_file()  # True/False - 是否为文件
path.is_dir()   # True/False - 是否为目录

# 路径拼接（使用 / 运算符）
output_path = path.parent / "output" / path.stem / ".md"
#D:\documents\output\万用表的使用\.md
print(output_path)