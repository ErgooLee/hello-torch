import os
import pandas as pd
import torch

dir_path = os.path.join(os.getcwd(), 'data')
os.makedirs(dir_path, exist_ok=True)
data_file = os.path.join(dir_path, 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
print(outputs)

inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
print(inputs)

x, y = torch.tensor(inputs.values, dtype=torch.float32), torch.tensor(outputs.values, dtype=torch.float32)
print(x)
print(y)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(X)
print(X + a)
print((X + a).shape)
print(X.sum())

X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(X.sum(dim=0))
print(X.sum(dim=1))
print(X.sum(dim=2))

print(X.sum(dim=0, keepdim=True))

print(X / X.sum(dim=0, keepdim=True))

A = torch.arange(9, dtype=torch.float32).reshape(3, 3)
x = torch.arange(3, dtype=torch.float32)
print(torch.mv(A, x))

A = torch.arange(9, dtype=torch.float32).reshape(3, 3)
x = torch.arange(3, dtype=torch.float32).reshape(-1, 1)
print(torch.mm(A, x))

A = torch.arange(9, dtype=torch.float32).reshape(3, 3)
x = torch.arange(3, dtype=torch.float32).reshape(3, 1)
print(torch.mm(A, x))
# print(torch.mv(A, x)) error

print(x.norm())

print(A.norm())