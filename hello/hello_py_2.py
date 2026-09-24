
"""
Python 高级语法练习
建议：从上到下运行，观察每个知识点的输出。

适合 Java / Kotlin 开发者学习 Python。
"""


# ============================================================
# 1. *args / **kwargs
# ============================================================

def test_args(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)


print("\n===== 1. *args / **kwargs =====")

test_args(1, 2, 3, name="Tom", age=20)


# ============================================================
# 2. lambda
# ============================================================

print("\n===== 2. lambda =====")

add = lambda x, y: x + y

print(add(1, 2))


users = [
    {"name": "Tom", "age": 30},
    {"name": "Jack", "age": 20},
    {"name": "Bob", "age": 25},
]

# 根据 age 排序
users.sort(key=lambda x: x["age"])

print(users)


# ============================================================
# 3. enumerate
# ============================================================

print("\n===== 3. enumerate =====")

names = ["Tom", "Jack", "Bob"]

for i, name in enumerate(names):
    print(i, name)


# ============================================================
# 4. zip
# ============================================================

print("\n===== 4. zip =====")

names = ["Tom", "Jack", "Bob"]
ages = [20, 30, 25]

for name, age in zip(names, ages):
    print(name, age)


# ============================================================
# 5. 解包
# ============================================================

print("\n===== 5. 解包 =====")

# -------------------------
# 序列解包
# -------------------------

a, b, c = [1, 2, 3]

print(a)
print(b)
print(c)


# -------------------------
# * 解包
# -------------------------

nums = [1, 2, 3]

print(*nums)
# 等价于：
# print(1, 2, 3)


# -------------------------
# 合并 list
# -------------------------

list1 = [1, 2]
list2 = [3, 4]

list3 = [*list1, *list2]

print(list3)


# -------------------------
# 字典解包
# -------------------------

user1 = {
    "name": "Tom",
    "age": 20,
}

user2 = {
    **user1,
    "city": "Shanghai",
}

print(user2)


# ============================================================
# 6. yield / Generator
# ============================================================

print("\n===== 6. yield / Generator =====")


def numbers():
    for i in range(5):
        print("生成:", i)
        yield i


gen = numbers()

print("第一次 next:")
print(next(gen))

print("第二次 next:")
print(next(gen))

print("继续遍历:")

for x in gen:
    print("得到:", x)


# ============================================================
# 7. 装饰器 Decorator
# ============================================================

print("\n===== 7. Decorator =====")


def log(func):

    def wrapper():
        print("before")
        func()
        print("after")

    return wrapper


@log
def hello():
    print("hello")


hello()


# @log 本质上相当于：
#
# hello = log(hello)


# ============================================================
# 8. with / Context Manager
# ============================================================

print("\n===== 8. with / Context Manager =====")


class MyContext:

    def __enter__(self):
        print("进入 with")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("退出 with")


with MyContext():
    print("执行任务")


# ============================================================
# 9. property
# ============================================================

print("\n===== 9. property =====")


class User:

    def __init__(self, age):
        self._age = age

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        self._age = value


user = User(20)

print("age =", user.age)

user.age = 30

print("age =", user.age)


# ============================================================
# 10. dataclass
# ============================================================

print("\n===== 10. dataclass =====")

from dataclasses import dataclass


@dataclass
class UserInfo:
    name: str
    age: int


user = UserInfo("Tom", 20)

print(user)
print(user.name)
print(user.age)


# ============================================================
# 11. async / await
# ============================================================

print("\n===== 11. async / await =====")

import asyncio


async def task():
    print("任务开始")

    # 模拟异步等待
    await asyncio.sleep(1)

    print("任务结束")

    return "result"


async def main():
    result = await task()
    print("result =", result)


asyncio.run(main())


# ============================================================
# 12. 魔术方法 __xxx__
# ============================================================

print("\n===== 12. 魔术方法 =====")


class MyList:

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return f"MyList({self.data})"


my_list = MyList([10, 20, 30])

print("len:", len(my_list))

print("my_list[1]:", my_list[1])

print("my_list:", my_list)


# 实际上：
#
# len(my_list)
# 等价于：
# my_list.__len__()
#
# my_list[1]
# 等价于：
# my_list.__getitem__(1)
#
# print(my_list)
# 会调用：
# my_list.__str__()


# ============================================================
# 13. Iterator / Iterable
# ============================================================

print("\n===== 13. Iterator / Iterable =====")

numbers = [10, 20, 30]

# list 是 Iterable
iterator = iter(numbers)

print(next(iterator))
print(next(iterator))
print(next(iterator))


# 等价于 for 循环内部的大致过程：
#
# iterator = iter(numbers)
#
# while True:
#     try:
#         x = next(iterator)
#         print(x)
#     except StopIteration:
#         break


# ============================================================
# 14. match / case
# ============================================================

print("\n===== 14. match / case =====")


def handle_command(command):

    match command:

        case "start":
            print("启动")

        case "stop":
            print("停止")

        case "pause":
            print("暂停")

        case _:
            print("未知命令")


handle_command("start")
handle_command("stop")
handle_command("hello")


# ============================================================
# 15. 综合练习：Generator + Decorator + dataclass
# ============================================================

print("\n===== 15. 综合练习 =====")


@dataclass
class Student:
    name: str
    score: int


def log_call(func):

    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print("调用完成")
        return result

    return wrapper


@log_call
def get_students():

    students = [
        Student("Tom", 90),
        Student("Jack", 80),
        Student("Bob", 95),
    ]

    for student in students:
        yield student


for student in get_students():
    print(student)

