import matplotlib.pyplot as plt
import numpy as num
import csv
from sklearn.linear_model import Ridge

def adjustedR(y, pred, a):
    mean = num.mean(pred)
    return 1 - ((RSS(y, pred) / (len(y) - a - 1)) / (TSS(y) / (len(y) - 1)))
def RSS(y, ypred):
    rss = 0
    for i in range(len(y)):
        rss += num.square(y[i] - ypred[i])
    return rss
def TSS(y):
    tss = 0
    for i in range(len(y)):
        tss += num.square(y[i] - num.average(y))
    return tss
def r2(y, ypred):
    return 1 - (RSS(y, ypred) / TSS(y))
def coef(main, y):
    maint = main.T
    return num.linalg.pinv(maint @ main) @ maint @ y

x1 = num.array([])
x2 = num.array([])
x3 = num.array([])
x4 = num.array([])
x5 = num.array([])
x6 = num.array([])
result = num.array([])
with open('data.csv') as inputcsv:
    csv_list = list(csv.reader(inputcsv))
for row in csv_list:
    if row != csv_list[0]:
        x1 = num.append(x1, float(row[1]))
        x2 = num.append(x2, float(row[2]))
        x3 = num.append(x3, float(row[3]))
        x4 = num.append(x4, float(row[4]))
        x5 = num.append(x5, float(row[5]))
        x6 = num.append(x6, float(row[6]))
        if row[7] == '':
            continue
        result = num.append(result, float(row[7]))
main = num.ones(len(x1))
main = num.column_stack((main, x1, x2, x3, x4, x5, x6))
Y = result
test=main[100:]
x= main[0:100]
cf1 = coef(x, Y)
coefs = num.array([])
alpha=num.arange(0, 40, 1)
for index in alpha:
    for xin in range(len(x)):
        ridge = Ridge(normalize=True,alpha=index)
        ridge.fit(x, Y)
    coefs=num.append(coefs,ridge.coef_)
coefs=num.reshape(coefs,(40,7))
x1 = num.array([])
x2 = num.array([])
x3 = num.array([])
x4 = num.array([])
x5 = num.array([])
x6 = num.array([])
for row in coefs:
    x1 = num.append(x1, row[1])
    x2 = num.append(x2, row[2])
    x3 = num.append(x3, row[3])
    x4 = num.append(x4, row[4])
    x5 = num.append(x5, row[5])
    x6 = num.append(x6, row[6])
plt.figure()
plt.plot(alpha, x1, c='b')
plt.plot(alpha, x2, c='g')
plt.plot(alpha, x3, c='r')
plt.plot(alpha, x4, c='c')
plt.plot(alpha, x5, c='k')
plt.plot(alpha, x6, c='y')
plt.show()

