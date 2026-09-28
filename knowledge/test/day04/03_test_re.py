import re

#   MD中如何识别段落中图片标记 ：     ![](image/xxx.jpg)       <img src="image/xxx.jpg">

md_content = "这是一段文字 ![图片描述1](images/photo1.jpg) 后面的文字 ![图片描述2](images/photo2.jpg) 后面的文字"

# 非贪婪匹配  *?
#pattern = "!\[.*?\]\(.*?\)"
# [    '![图片描述1](images/photo1.jpg)'  ,    '![图片描述2](images/photo2.jpg)'     ]

# 贪婪匹配  *
pattern = "!\[.*\]\(.*\)"
# [    '![图片描述1](images/photo1.jpg) 后面的文字 ![图片描述2](images/photo2.jpg)'    ]

matches = re.findall(pattern, md_content)
print(matches)


image_filename = "photo.jpg"
specific_pattern = r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)"
print(specific_pattern)