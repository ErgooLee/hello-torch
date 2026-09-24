# ==========================================
# Python 基础语法练习
# 适合 Java / Kotlin 开发者
# ==========================================


# ==========================================
# 1. 变量和基本类型
# ==========================================

name = "Tom"
age = 18
score = 95.5
is_ok = True
nothing = None

print("===== 1. 变量和基本类型 =====")
print(name)
print(age)
print(score)
print(is_ok)
print(nothing)

print(type(name) == str)
print(isinstance(age, int))
print(type(score))
print(type(is_ok))
print(type(nothing))

# ==========================================
# 2. 字符串
# ==========================================

print("\n===== 2. 字符串 =====")

name = "Alice"

print(name)
print(len(name))
print(name.upper())
print(name.lower())

# f-string
age = 18
print(f"{name} is {age} years old")

# ==========================================
# 3. List
# ==========================================

print("\n===== 3. List =====")

numbers = [1, 2, 3, 4, 5]

# 访问元素
print(numbers[0])
print(numbers[1])
# -1是最后一个元素
print(numbers[-1])

# 添加元素
numbers.append(6)
print(numbers)

# 删除元素
numbers.remove(3)
print(numbers)

# 修改元素
numbers[0] = 100
print(numbers)

# List 长度
print(len(numbers))

# 切片 左闭右开
numbers = [1, 2, 3, 4, 5]

print(numbers[1:4])  # [2, 3, 4]
print(numbers[:3])  # [1, 2, 3]
print(numbers[2:])  # [3, 4, 5]
print(numbers[:])  # [1, 2, 3, 4, 5]

# ==========================================
# 4. Tuple
# ==========================================

print("\n===== 4. Tuple =====")

point = (10, 20, 30, 40, 50)

print(point)
print(point[0])
print(point[1])

# Tuple 不允许修改
# point[0] = 100  # 取消注释会报错


# ==========================================
# 5. Dictionary
# ==========================================

print("\n===== 5. Dictionary =====")

person = {
    "name": "Alice",
    "age": 18,
    "city": "Shanghai"
}

print(person)

# 根据 key 获取 value
print(person["name"])
print(person["age"])

# 添加
person["job"] = "Android Developer"

print(person)

# 修改
person["age"] = 20

print(person)

# 遍历 key 和 value
for key, value in person.items():
    print(key, value)

# ==========================================
# 6. Set
# ==========================================

print("\n===== 6. Set =====")

numbers = {1, 2, 3, 3, 4}

# Set 会自动去重
print(numbers)

numbers.add(5)

print(numbers)

numbers.remove(2)

print(numbers)

# ==========================================
# 7. if / elif / else
# ==========================================

print("\n===== 7. if =====")

age = 20

if age < 18:
    print("child")
elif age < 60:
    print("adult")
else:
    print("old")

# ==========================================
# 8. for
# ==========================================

print("\n===== 8. for =====")

# range(5) -> 0, 1, 2, 3, 4
for i in range(5):
    print(i)

names = ["Tom", "Bob", "Alice"]

for name in names:
    print(name)

# ==========================================
# 9. while
# ==========================================

print("\n===== 9. while =====")

i = 0

while i < 5:
    print(i)
    i += 1

# ==========================================
# 10. function
# ==========================================

print("\n===== 10. function =====")


def add(a: int, b: int) -> int:
    return a + b


result = add(1, 2)

print(result)

# ==========================================
# 11. 函数默认参数
# ==========================================

print("\n===== 11. 默认参数 =====")


def greet(name="Tom"):
    print(f"Hello {name}")


greet()
greet("Alice")

# ==========================================
# 12. class
# ==========================================

print("\n===== 12. class =====")


class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print(f"Hello, I'm {self.name}")


person = Person("Tom", 18)

print(person.name)
print(person.age)

person.say_hello()

# ==========================================
# 13. import
# ==========================================

print("\n===== 13. import =====")

import math

print(math.sqrt(16))
print(math.pi)

# ==========================================
# 14. try / except
# ==========================================

print("\n===== 14. try / except =====")

try:
    x = 10 / 0
    print(x)
except Exception as e:
    print("发生异常：", e)

# ==========================================
# 15. List Comprehension
# ==========================================

print("\n===== 15. List Comprehension =====")

numbers = [1, 2, 3, 4, 5]

# 普通写法
result = []

for n in numbers:
    if n % 2 == 0:
        result.append(n)

print("普通写法：", result)

# Python 写法
result = [n for n in numbers if n % 2 == 0]

print("List Comprehension：", result)

# ==========================================
# 16. 一个简单的 Android Log 分析例子
# ==========================================

print("\n===== 16. Android Log 分析 =====")

logs = [
    "INFO Activity started",
    "ERROR MediaCodec failed",
    "INFO Activity stopped",
    "ERROR MediaCodec timeout",
    "INFO Player started",
    "ERROR MediaCodec crash"
]

# 找出所有 ERROR
for log in logs:
    if "ERROR" in log:
        print(log)

# 使用 List Comprehension
errors = [log for log in logs if "ERROR" in log]

print("\n所有 ERROR：")
print(errors)

# ==========================================
# 17. 综合练习
# ==========================================

print("\n===== 17. 综合练习 =====")

students = [
    {"name": "Tom", "score": 90},
    {"name": "Bob", "score": 60},
    {"name": "Alice", "score": 95},
    {"name": "Jack", "score": 70},
]

# 找出分数 >= 80 的学生
good_students = [
    student
    for student in students
    if student["score"] >= 80
]

print("优秀学生：")

for student in good_students:
    print(
        f"name={student['name']}, "
        f"score={student['score']}"
    )

# ==========================================
# 18. 一个稍微真实一点的脚本
# ==========================================

print("\n===== 18. 综合脚本 =====")


def analyze_logs(logs):
    """
    分析日志，统计 ERROR 数量
    """

    error_logs = [
        log
        for log in logs
        if "ERROR" in log
    ]

    return error_logs


logs = [
    "INFO Activity start",
    "INFO MediaPlayer start",
    "ERROR MediaCodec failed",
    "INFO Activity resume",
    "ERROR MediaCodec timeout",
    "INFO Activity pause",
]

errors = analyze_logs(logs)

print("ERROR 数量：", len(errors))

for error in errors:
    print(error)

# ==========================================
# 结束
# ==========================================

print("\n==========================================")
print("Python 基础语法练习完成！")
print("==========================================")

class Student:
    def __init__(self, name: str, id: int):
        self.name = name
        self.id = id

    def print_info(self):
        print(f"Student {self.name}, ID {self.id}")

student = Student("Tom", 1)
student.print_info()
