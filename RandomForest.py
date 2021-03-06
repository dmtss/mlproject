import numpy as num
import csv
from sklearn.ensemble import RandomForestRegressor as forest

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

x1=  num.array([])
x3 = num.array([])
x5 = num.array([])
x6 = num.array([])
Y = num.array([])
rsq = num.array([])
adjRsq = num.array([])
ones2 = num.ones(100)

with open("data.csv") as f:
    mylist = list(csv.reader(f))
for row in mylist:
    if row != mylist[0]:
        x1 = num.append(x1, float(row[1]))
        x3 = num.append(x3, float(row[3]))
        x5 = num.append(x5, float(row[5]))
        x6 = num.append(x6, float(row[6]))
        if row[7] == '':
            continue
        Y = num.append(Y, int(row[7]))

x1=x1[0:100]
x3= x3[0:100]
x5= x5[0:100]
x6= x6[0:100]
X = num.column_stack((ones2, x3*x1, x5*x1, x6*x1, x1*x1, x3*x3, x3*x5, x5*x5,  x3*x6, x5*x6, x6*x6))
x_test = X[60:80]
y_test = Y[60:80]
x_train = num.delete(X, range(60, 80), 0)
y_train = num.delete(Y, range(60, 80), 0)
for i in range(1, 220):
    forg = forest(max_depth=None, n_estimators=i, max_features="sqrt")
    forg.fit(x_train, y_train)
    predict = forg.predict(x_test)
    rsq = num.append(rsq, r2(y_test, predict))
    adjRsq = num.append(adjRsq, adjustedR(y_test, predict, 11))
print('R^2', num.max(rsq))
print('Adjusted R^2: ' , num.max(adjRsq),"\n\n\n")
print([ "{:0.2f}".format(x) for x in predict])