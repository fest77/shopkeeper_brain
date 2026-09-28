from typing import List

paragraphs: List[str] = []  # 存放所有段落列表
current_paragraph: List[str] = []

#           ['', '']
# paragraphs.append("\n".join(current_paragraph))
# paragraphs.append("\n".join(current_paragraph))


if current_paragraph:      #        []
    paragraphs.append("\n".join(current_paragraph))
    paragraphs.append("\n".join(current_paragraph))


print(paragraphs)
