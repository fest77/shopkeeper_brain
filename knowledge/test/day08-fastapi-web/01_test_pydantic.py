import datetime

from pydantic import BaseModel, Field, ValidationError

"""
对输入和输出数据进行类型检查，类型转换，参数数量限制，默认值设置等
"""

class Student(BaseModel):
    name: str = Field(..., description="姓名")  # ...表示参数必填    description用于生成api文档描述
    age: int = Field(..., description="年龄")  # 必填
    score: float = Field(default=0.0, description="成绩")  # 选填，默认0.0
    birthday: datetime.date = Field(default=datetime.date(2000, 5, 5), description="出生日期")


# 1. 正常创建
s1 = Student(name="张三", age=18, score=95.5)
print(s1)
# name='张三' age=18 score=95.5
#
# # 2. 转成字典
print(s1.model_dump())
# {'name': '张三', 'age': 18, 'score': 95.5, 'birthday': datetime.date(2000, 5, 5)}
#
# # 3. 转成 JSON 字符串
print(s1.model_dump_json())
# {"name":"张三","age":18,"score":95.5,"birthday":"2000-05-05"}
#
# # 4. 自动类型转换：字符串 "20" → int 20
s2 = Student(name="李四", age="20")
print(s2.age, type(s2.age))  # 20  <class 'int'>
#
# # 5. 类型校验失败：直接报错
# try:
#     Student(name="王五", age="不是数字")
# except ValidationError as e:
#     print(e)  # age - Input should be a valid integer

# # 6. 缺少必填字段 ───
# try:
#     s4 = Student(age=18)  # 缺少必填的 name
# except ValidationError as e:
#     print(e)
    # name - Field required
#
# 7. 多余字段自动丢弃
# s3 = Student(name="赵六", age=22, hobby="篮球")
# print(s3.model_dump())
# {'name': '赵六', 'age': 22, 'score': 0.0}   ← hobby 被丢弃了

# 8. 从字典创建（模拟后端返回数据)
data = {"name": "孙七", "age": 19, "score": 85, "email": "sun7@my_test.com"}
s6 = Student(**data)  # ** 解包字典，等价于 Student(name="孙七", age=19, ...)
print(s6)