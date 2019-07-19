# 第一类型组跑出来1.3.5.78，1.3.5.78不从队列里剔除掉，别的类型组里也会出现这些小分组
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import pearsonr # 皮尔逊相关系数
np.seterr(invalid='ignore')

# 读数据
grapdatanew1 = pd.read_csv('C:\\Users\\Administrator\\Desktop\\python\\10000.csv',header = None,sep = ',')
moni240file = pd.read_excel('C:\\Users\\Administrator\\Desktop\\python\\data240.xlsx',header = None)

# 取1000条 数据
grapdatanew2 = grapdatanew1.head(24000)
grapdatanew2.columns = np.arange(0,len(grapdatanew2.iloc[0]),1)
grapdatanew = grapdatanew2.iloc[:,1:]
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
l = []
m = []
for i in range(len(grapdata.iloc[0])):
    aa = grapdata.iloc[:,i].max() - grapdata.iloc[:,i].min()
    if aa > 0:
        l.append(i)
    else:
        m.append(i)

# 取组内数据不完全一样的数据组，并用平均值填充缺失值
grapdata = grapdata.loc[:,l].fillna(grapdata.mean())
grapdata.columns = np.arange(0,len(grapdata.iloc[0]),1)
grapdata.index = np.arange(0,len(grapdata.iloc[:,0]),1)

grapdatafile = grapdatafile.iloc[:,a].fillna(grapdatafile.mean())
grapdatafile.columns = np.arange(0,len(grapdatafile.iloc[0]),1)
grapdatafile.index = np.arange(0,len(grapdatafile.iloc[:,0]),1)
# 用皮尔逊相关系数这个封装好的函数，取出强相关的数据组
lst=[]
for i in range(len(grapdata.iloc[0,:])):
    lst.append(i)
    for j in range(len(grapdata.iloc[0,:])):
        if j != i:
            aa = pearsonr(grapdata.iloc[:,i],grapdata.iloc[:,j])
            if aa[1] < 0.05 and aa[0] > 0.9:
                lst.append(j)
    lst.append('\n')

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

lst=[]
for i in range(len(grapdata.iloc[0,:])):
    lst.append(i)
    for j in range(len(grapdata.iloc[0,:])):
        if j != i:
            aa = corrcoef(grapdata.iloc[:,i],grapdata.iloc[:,j])
            if aa > 0.9:
                lst.append(j)
    lst.append('\n')
'''

# 将lst的list格式转换成多维数组格式
lists = [[]]
index = 0
for i in lst:
    if i == '\n':
        index += 1
        lists.append([])
    else:
        lists[index].append(i)

'''
# 将二级的数组内数据进行排序 如[10,9]变为[9,10]
for i in range(len(lists)):
    lists[i] = sorted(lists[i])

# 对一维数组内的二级数组进行排重。
# lists = list(set([tuple(t) for t in lists])) #列表表达式方法，和下面的结果一样
s = []
for t in lists:
    t = tuple(t)
    s.append(t)
    snew = list(set(s)) #set是集合，可以作为去重使用

# 将去重后的set结果转换为list格式，并且对二维数组进行排序
# lists = [list(v) for v in lists] #列表表达式方法，和下面的结果一样
ss = []
for v in snew:
    v =list(v)
    if len(v) > 1:
        ss.append(v)
ss.sort()
'''

# 将对一维数组内的二级数组进行排重
## 将每组数据内一样的索引方法放一个集合里
m = []
for i in range(len(lists)):
    if i not in m:
        m.append(i)
        for j in range(len(lists)):
            if i < j:
                if len(set(lists[i]) - set(lists[j])) == 0 and len(set(lists[j]) - set(lists[i])) == 0:
                    m.append(j)
        m.append('\n')

##  将lst的list格式转换成多维数组格式
n = [[]]
index = 0
for i in m:
    if i == '\n':
        index += 1
        n.append([])
    else:
        n[index].append(i)

## 取每一个集合里的第一个索引，达到去重效果
listss = []
for i in n:
    if len(i) >= 1:
        listss.append(i[0])

# 提取去重后的二维数组
llists = np.array(lists)[list(listss)].tolist()

# 将每一组内数据没用相似图形的给排除
ss = []
for i in llists:
    if len(i) > 1:
        ss.append(i)

# 对二维数组进行排序
ss.sort()

'''
# 随意调用任意组数据，画图
def matp(num,i):
    x = np.arange(0,len(grapdata.iloc[:,0]))
    y = grapdata.iloc[:,i]
#    func = interpolate.interp1d(x, y, kind='cubic') # 实现函数
    plt.rcParams['font.family'] = 'SimHei'  # 设置黑体
    plt.rcParams['font.size'] = '16'  # 设置字号
    plt.figure(figsize = (18,6),dpi = 80)
    plt.title('{0}点曲线图'.format(num))
    plt.xlabel('{0}个点'.format(num))
    plt.xlim(0,len(grapdata.iloc[:,0])-1)
    plt.xticks(np.linspace(0,len(grapdata.iloc[:,0])-1,16))
    plt.ylabel('value值')
    plt.ylim(grapdata.iloc[:,i].min(),grapdata.iloc[:,i].max())
    plt.yticks(np.linspace(grapdata.iloc[:,i].min(),grapdata.iloc[:,i].max(),8))
    plt.plot(x,y,label="$grapdata.iloc[:,i]$",color="red",linewidth=2)
    return plt.show()
print(matp(16,1))

'''

def digui(ss):
    l = []
    for i in range(len(ss)):
        if i not in l:
            l.append(i)
            for j in range(len(ss)):
                if i < j:
                    if len(set(ss[i]) - set(ss[j])) < len(set(ss[i])):
                        l.append(j)
            l.append('\n')

    n = [[]]
    index = 0
    for i in l:
        if i == '\n':
            index += 1
            n.append([])
        else:
            n[index].append(i)

    m = []
    for i in n:
        m.append(np.array(ss)[i].tolist())
        m.append('\n')

    a = [[]]
    index = 0
    for i in m:
        if i == '\n':
            index += 1
            a.append([])
        else:
            a[index].append(i)

    b = []
    for i in a:
        for j in i:
            for h in j:
                b.append(h)
            b.append('\n')

    c = [[]]
    index = 0
    for i in b:
        if i == '\n':
            index += 1
            c.append([])
        else:
            c[index].append(i)

    d = []
    for i in range(len(c)):
        if len(c[i]) > 0:
            for j in range(len(c[i])):
                for k in range(len(c[i][j])):
                    d.append(c[i][j][k])
            d.append('\n')

    e = [[]]
    index = 0
    for i in d:
        if i == '\n':
            index += 1
            e.append([])
        else:
            e[index].append(i)

    o = []
    for i in e:
        if len(i) > 0:
            i = set(i)
            o.append(sorted(list(i)))

    if len(o) == len(ss):
        return o
    else:
        ss = o
        return digui(ss)
sss = digui(ss)

ll = []
for i in sss:
    bb = 2
    ll.append(set(i))
    for j in range(len(i)):
        for k in range(len(i)):
            if j < k:
                aa = pearsonr(grapdata.iloc[:,i[j]], grapdata.iloc[:,i[k]])
                if aa[0] < bb:
                    bb = aa[0]
                    cc = set(list([i[j],i[k]]))
    ll.append(bb)
    ll.append(cc)
    ll.append('\n')

ee = [[]]
index = 0
for i in ll:
    if i == '\n':
        index += 1
        ee.append([])
    else:
        ee[index].append(i)

# 将结果进行输出
with open('C:/Users/Administrator/Desktop/similarclassifydata/similarclassify.txt', 'w') as file1: # 输出各类型组
    file1.write(str(ss))
with open('C:/Users/Administrator/Desktop/similarclassifydata/similarclassifyjihe.txt', 'w') as file2: # 输出合并后各类型组
    file2.write(str(sss))
with open('C:/Users/Administrator/Desktop/similarclassifydata/similarclassifymin.txt', 'w') as file3: # 输出合并后各类型组最小的系数
    file3.write(str(ee))
grapdatafile.to_csv('C:/Users/Administrator/Desktop/similarclassifydata/grapdatafile240.csv') # 240点数据输出
grapdata.to_csv('C:/Users/Administrator/Desktop/similarclassifydata/grapdata16.csv') # 16点数据输出
