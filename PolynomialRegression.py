import numpy as num
import csv

x1 = num.array([])
x3 = num.array([])
x5 = num.array([])
x6 = num.array([])
result = num.array([])

with open('data.csv') as inputcsv:
    csv_list = list(csv.reader(inputcsv))
for row in csv_list:
    if row != csv_list[0]:
        x1 = num.append(x1, float(row[1]))
        x3 = num.append(x3, float(row[3]))
        x5 = num.append(x5, float(row[5]))
        x6 = num.append(x6, float(row[6]))
        if row[7] == '':
            continue
        result = num.append(result, float(row[7]))

ones = num.ones(len(x3))
ones2 = num.ones(100)
main = num.column_stack((ones, x1, x3,  x5, x6))
Y = result

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
    return num.linalg.pinv(maint@main)@maint@y

x1 = x1[0:100]
x3 = x3[0:100]
x5 = x5[0:100]
x6 = x6[0:100]
X = num.column_stack((ones2, x3*x1, x5*x1, x6*x1, x1*x1, x3*x3, x3*x5, x5*x5, x3*x6, x5*x6,x6*x6))

cf = coef(X, Y)
yest = X@cf
rsq = r2(Y, yest)
adjR = adjustedR(yest, Y, 11)
print("R^2", rsq)
print("Adjusted R^2", adjR)


