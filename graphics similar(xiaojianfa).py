# 第一类型组跑出来1.3.5.78，那么1.3.5.7,8就将不作为中心点进行跑数据，但会出现在别的类型组里
import numpy as np
import pandas as pd
from scipy.stats import pearsonr # 皮尔逊相关系数
np.seterr(invalid='ignore')

# 读数据
grapdatanew1 = pd.read_table('C:\\Users\\Administrator\\Desktop\\python\\1wdata.txt',header = None,sep = ',')
moni240file = pd.read_excel('C:\\Users\\Administrator\\Desktop\\python\\data30pingjun.xlsx',header = None)

# 取1000条 数据
grapdatanew = grapdatanew1.head(240000)
grapdatanew.columns = np.arange(0,len(grapdatanew.iloc[0]),1)
grapdatanew.columns = ['id','shuju','rank']

# 行转列
grapdatafile = grapdatanew.pivot(index = 'rank',columns = 'id',values= 'shuju')
grapdatafile.columns = np.arange(0,len(grapdatafile.iloc[0]),1) #索引重塑
grapdatafile.index = np.arange(0,len(grapdatafile.iloc[:,0]),1)

# 将左右连接表格
grapdatafilenew = pd.concat((moni240file,grapdatafile),axis = 1,ignore_index= True)

# 将每列数据取n点的平均值并对索引重新打标签
grapdata = grapdatafilenew.iloc[:,1:].groupby(grapdatafilenew.iloc[:,0]).mean()
grapdata.columns = np.arange(0,len(grapdata.iloc[0]),1)
grapdata.index = np.arange(0,len(grapdata.iloc[:,0]),1)

# 将组内数据完全一样和不完全一样的组索引放在不同的类别中
a = []
b = []
for i in range(len(grapdata.iloc[0])):
    aa = grapdata.iloc[:,i].max() - grapdata.iloc[:,i].min()
    if aa > 0:
        a.append(i)
    else:
        b.append(i)

# 取组内数据不完全一样的数据组，并用平均值填充缺失值
grapdata = grapdata.loc[:,a].fillna(grapdata.mean())
grapdata.columns = np.arange(0,len(grapdata.iloc[0]),1)
grapdata.index = np.arange(0,len(grapdata.iloc[:,0]),1)

grapdatafile = grapdatafile.iloc[:,a].fillna(grapdatafile.mean())
grapdatafile.columns = np.arange(0,len(grapdatafile.iloc[0]),1)
grapdatafile.index = np.arange(0,len(grapdatafile.iloc[:,0]),1)
# 用皮尔逊相关系数这个封装好的函数，取出强相关的数据组
c = []
for i in range(len(grapdata.iloc[0,:])):
    if i not in c:
        c.append(i)
        for j in range(len(grapdata.iloc[0,:])):
            if j != i:
                aa = pearsonr(grapdata.iloc[:,i],grapdata.iloc[:,j])
                if aa[1] < 0.05 and aa[0] > 0.9:
                    c.append(j)
        c.append('\n')
'''
# 手写函数

# 求乘积之和
from math import sqrt
def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab

# 计算皮尔逊相关系数
def corrcoef(x, y):
    n = len(x)
    sum1 = sum(x) # 求和
    sum2 = sum(y)
    sumofxy = multipl(x, y) # 求乘积之和
    sumofx2 = sum([pow(i, 2) for i in x]) # 求平方和
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n)) # 计算皮尔逊相关系数
    if den > 0:
        result = num / den
    else:
        result = 0
    return result

c = []
for i in range(len(grapdata.iloc[0,:])):
    if i not in c:
        c.append(i)
        for j in range(len(grapdata.iloc[0,:])):
            if j != i:
                aa = corrcoef(grapdata.iloc[:,i],grapdata.iloc[:,j])
                if aa > 0.9:
                    c.append(j)
        c.append('\n')
'''
# 将lst的list格式转换成多维数组格式
d = [[]]
index = 0
for i in c:
    if i == '\n':
        index += 1
        d.append([])
    else:
        d[index].append(i)

# 将lists二维数组里面的小于1的给排重
e = []
for i in d:
    if len(i) > 1:
        e.append(i)
e.sort()
print(grapdata.shape)
'''
# 两组交集数据的清洗优化
f = []
for i in range(len(e)):
    if i not in f:
        f.append(i)
        for j in range(len(e)):
            if i < j:
                if len(set(e[i]) - set(e[j])) < len(set(e[i])):
                    f.append(j)
        f.append('\n')

g = [[]]
index = 0
for i in f:
    if i == '\n':
        index += 1
        g.append([])
    else:
        g[index].append(i)

h = []
k = []
for i in g:
    if len(i) == 1:
        h.append(i)
    elif len(i) > 1:
        k.append(i)
l = []
for i in k:
    for j in range(len(i)):
        aa = i[0].intersection(i[j])
                
        i[j]
print(h,k)


'''
# 将结果进行输出
with open('C:/Users/Administrator/Desktop/similarclassifydata/similarclassifylists.txt', 'w') as file1: # 输出各类型组，未排掉单项
    file1.write(str(d))
with open('C:/Users/Administrator/Desktop/similarclassifydata/similarclassify.txt', 'w') as file2: # 输出各类型组
    file2.write(str(e))
grapdatafile.to_csv('C:/Users/Administrator/Desktop/similarclassifydata/grapdatafile240.csv') # 240点数据输出
grapdata.to_csv('C:/Users/Administrator/Desktop/similarclassifydata/grapdatapingjun.csv') # 16点数据输出


