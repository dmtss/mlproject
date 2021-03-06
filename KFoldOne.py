import numpy as num
import csv

x1=num.array([])
x3=num.array([])
x5=num.array([])
x6=num.array([])
result=num.array([])

def MSE(ypred,y):
    numr=0
    for i in range (0, len(y), 1):
        numr += (ypred[i]-y[i])*(ypred[i]-y[i])
    return numr/len(y)
def adjustedR(y, pred, a):
    mean=num.mean(pred)
    return 1-((RSS(y,pred)/(len(y)-a-1))/(TSS(y)/(len(y)-1)))
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
        return 1-(RSS(y, ypred)/TSS(y))
def coef(main, y):
    maint=main.T
    return num.linalg.pinv(maint@main)@maint@y

with open('data.csv') as inputcsv:
  csv_list=list(csv.reader(inputcsv))
for row in csv_list:
    if row != csv_list[0]:
        x1=num.append(x1, float(row[1]))
        x3=num.append(x3, float(row[3]))
        x5=num.append(x5, float(row[5]))
        x6=num.append(x6, float(row[6]))
        if row[7] == '':
            continue
        result=num.append(result, float(row[7]))

main=num.ones(len(x1))
main=num.column_stack((main,x1,x3,x5,x6))
Y = result
x= main[0:100]

rsquares= num.array([])
adjr= num.array([])

k=10
foldsize = int(len(x)/k)

for i in range(0, len(x), foldsize):
    x_test = x[i:i + foldsize]
    y_test = Y[i:i + foldsize]
    x_train = num.delete(x, range(i, i + foldsize), 0)
    y_train = num.delete(Y, range(i, i + foldsize), 0)
    cef = coef(x_train, y_train)
    cvhat= x_test@cef
    for i in range(len(cvhat)):
        if cvhat[i] < 0:
            cvhat[i] = 0
    rsquares = num.append(rsquares, r2(y_test,cvhat))
    adjr = num.append(adjr, adjustedR(y_test, cvhat, 4))

print("R^2: ",rsquares.max())
print("Adjusted R^2: ",adjr.max())