# 导入模块 (类似于 Kotlin 的 import)
from dataclasses import dataclass
from typing import List, Dict, Optional

# ==========================================
# 1. 变量与基础类型 (Variables & Types)
# ==========================================
print("--- 1. Variables ---")

# Python 是动态类型，不需要 val/var。
# 约定俗成：全大写表示常量 (类似于 Kotlin 的 const val)
PI = 3.14

# 普通变量 (类似于 Kotlin 的 var)
name = "Python"
age = 30

# Python 3.6+ 支持类型提示 (Type Hints)，这对 Kotlin 开发者非常友好！
# 注意：这仅仅是“提示”，Python 解释器在运行时不会强制检查它。
city: str = "Beijing"
is_developer: bool = True  # 首字母大写：True, False, None (Kotlin 中是 true, false, null)

# 字符串插值 (f-string，类似于 Kotlin 的 "$name is $age")
print(f"My name is {name}, I am {age} years old.")


# ==========================================
# 2. 空值处理 (Nullability)
# ==========================================
print("\n--- 2. Nullability ---")

# Python 中用 None 表示空值 (等同于 Kotlin 的 null)
# Python 没有 Kotlin 的 `?` 安全调用和 `?:` Elvis 操作符
address: Optional[str] = None # 等同于 Kotlin 的 String?

if address is None: # 推荐用 is 判断 None
    print("Address is not provided")

# 模拟 Kotlin 的 Elvis 操作符: val res = address ?: "Default"
res = address or "Default Address"
print(f"Resolved address: {res}")


# ==========================================
# 3. 集合 (Collections)
# ==========================================
print("\n--- 3. Collections ---")

# 列表 List (等同于 Kotlin 的 MutableList)
fruits: List[str] = ["apple", "banana", "cherry"]
fruits.append("orange") # 相当于 fruits.add()
print(f"First fruit: {fruits[0]}") # 索引访问一致

# 字典 Dictionary (等同于 Kotlin 的 MutableMap)
user_map: Dict[str, int] = {"Alice": 25, "Bob": 30}
user_map["Charlie"] = 35
print(f"Alice's age: {user_map.get('Alice')}") # 安全获取，找不到返回 None

# 元组 Tuple (Python 特有，可以理解为不可变的 List，常用来返回多个值)
coordinates = (10.0, 20.0)


# ==========================================
# 4. 控制流 (Control Flow)
# ==========================================
print("\n--- 4. Control Flow ---")

# 重点：Python 没有大括号 {}，完全依赖缩进 (Indentation) 和冒号 (:)
score = 85

if score >= 90:
    print("A")
elif score >= 80:       # 注意：是 elif，不是 else if
    print("B")
else:
    print("C")

# For 循环 (相当于 Kotlin 的 for (fruit in fruits))
for fruit in fruits:
    print(f"Iterating: {fruit}")

# 范围循环 (相当于 Kotlin 的 for (i in 0 until 5))
for i in range(5):
    print(f"Index: {i}")

# 列表推导式 (List Comprehension) - Python 的精髓！
# 相当于 Kotlin 的 fruits.map { it.uppercase() }.filter { it.startswith("A") }
a_fruits = [f.upper() for f in fruits if f.startswith("a")]
print(f"A fruits: {a_fruits}")


# ==========================================
# 5. 函数与 Lambda (Functions)
# ==========================================
print("\n--- 5. Functions ---")

# 用 def 定义函数 (相当于 Kotlin 的 fun)
def greet(person: str, greeting: str = "Hello") -> str:
    """这是函数的文档字符串 (Docstring)，写在这里。"""
    return f"{greeting}, {person}!"

print(greet("Kotlin Dev"))
print(greet(greeting="Hi", person="Python")) # 命名参数，和 Kotlin 一样

# Lambda 表达式
# Python 的 lambda 只能写单行表达式，没有 Kotlin 的花括号 lambda 强大
multiply = lambda x, y: x * y
print(f"Lambda result: {multiply(3, 4)}")


# ==========================================
# 6. 类与对象 (Classes & Objects)
# ==========================================
print("\n--- 6. Classes ---")

# 普通类
class Animal:
    # 构造函数 (相当于 Kotlin 的 init 块和主构造函数)
    # self 等同于 Kotlin 的 this，但在 Python 中必须显式作为第一个参数写出来！
    def __init__(self, name: str):
        self.name = name # 定义实例属性

    def speak(self):
        print(f"{self.name} makes a sound.")

# 继承 (把父类写在括号里)
class Dog(Animal):
    def speak(self):
        print(f"{self.name} barks!") # 覆写方法不需要 override 关键字

dog = Dog("Buddy") # 实例化对象不需要 new 关键字
dog.speak()

# 数据类 Data Class (Python 3.7+ 引入，简直就是为 Kotlin 开发者量身定制的！)
# 行为和 Kotlin 的 data class 几乎一模一样 (自动生成 __init__, __repr__, __eq__)
@dataclass
class User:
    id: int
    username: str
    is_active: bool = True

user1 = User(1, "john_doe")
print(user1) # 输出: User(id=1, username='john_doe', is_active=True)


# ==========================================
# 7. 异常处理 (Exceptions)
# ==========================================
print("\n--- 7. Exceptions ---")

# 类似于 Kotlin 的 try-catch-finally
try:
    result = 10 / 0
except ZeroDivisionError as e: # catch (e: ZeroDivisionError)
    print(f"Caught an error: {e}")
finally:
    print("This always runs.")