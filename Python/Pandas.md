# Series

导入模块

```python
import numpy as np
import pandas as pd
```

## 创建Series

```python
s = pd.Series(data, index=index)
```

其中，`data`支持字典、多维数组、标量。

### 字典

```python
d = {'b': 1, 'a': 2, 'u':3}
s = pd.Series(d)
```

可以将字典的`key`与`value`直接转递Series，也可以赋予`index`，来改变`Series`的顺序。

```python
s = pd.Series(d, index=['a', 'e', 'i', 'o', 'u'])
```

会将字典中的`key`与`value`与`index`进行匹配，并按照`index`的顺序进行存储。`index`中没有的`key`会赋予`NaN`。

字典中有但是`index`中没有的`key`会被丢掉。

### 多维数组

`data`是多维数组时，`index`必须与`data`的长度一致。没有`index`时，会创建数值型索引即`[0, 1, 2, ..., len(data) - 1]`。

```python
s1 = pd.Series(np.random.rand(5), index=['a', 'e', 'i', 'o', 'u'])
s2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'e', 'i', 'o', 'u'])
```

```python
s1 = pd.Series(np.random.rand(5))
s2 = pd.Series([1, 2, 3, 4, 5])
```

上面一行有字母标签，下面一行是数字标签。

**值得注意的是，Pandas中的`index`可以重复，第一行中有两个a是不会出问题的。**

### 标量

当`data`是标量是，必须提供索引，索引的长度即是`Series`的长度。

```python
s = pd.Series(5, index=['a', 'e', 'i', 'o', 'u'])
```

## 操作Series

```python
s = pd.Series([3, 1, 4, 7, 5], index=['a', 'e', 'i', 'o', 'u'])

s[0]  # 取值，输出data
# 3

s[:3]  # 切片操作，输出index和data
# a    3
# e    1
# i    4

s[s > s.median()]  # 条件输出
# o    7
# u    5
# dtype: int64
#  s.median()表示data的中位数， 输出为4.0

s[[4, 3, 1]]  # 按照4，3，1的顺序输出
# u    5
# o    7
# e    1
# dtype: int64

np.exp(s[:3])  # 可以当作列表来输入
# a    20.085537
# e     2.718282
# i    54.598150
# dtype: float64
```

Seires和NumPy的数组一样支持`dtype`，大部分情况下数据类型相同。

```python
s,dtype  # 获取dtype
# dtype('int64')

s.array  # 提取Series数组，是一种扩展数组
# <PandasArray>
# [3, 1, 4, 7, 5]
# Length: 5, dtype: int64

s.to_numpy()  # 转化为NumPy数组
# array([3, 1, 4, 7, 5], dtype=int64)
```

## 字典Series

Series类与字典相似，可以用索引标签提取或设置值。

```python
s['u']  # 提取key为'u'的value
# 5

s['u'] = 12  # 设置值
# s['u'] = 12

'u' in s  # 查找是否存在
# True

'f' in s  
# False

s['f']  # 引用不存在的值会保存
# KeyError

s,get('u')  # 利用get来访问可以防止报错
# 12

s.get('f')  # 访问不存在的标签默认返回None
# 这里什么都没有

s.get('f', 1)  # 可以设置默认值，不会改变s
# 1
```

## 矢量操作

Series和NumPy的数组一样，可以进行矢量操作。

```python
s + s  # 对应相加，相乘也行
# a     6
# e     2
# i     8
# o    14
# u    24
```

Series会根据标签自动对齐，没有的标签会将结果标记为`NaN`。

```python
s[1:] + s[:-1]  # 包含不同标签的Series相加
# a     NaN
# e     2.0
# i     8.0
# o    14.0
# u     NaN
# dtype: float64
```

## 名称属性

Series支持`name`属性。

```python
s.name  # 自动分配时，名称为None
# None

s2 = s.rename("pipi")  # 利用rename创建新的对象，s2与s指向不同的对象
# a     3
# e     1
# i     4
# o     7
# u    12
# Name: pipi, dtype: int64
```

# DataFrame

DataFrame是由多种类型的列组成的二位标签数组结构，是最常用的Pandas对象。和Series类似，支持多种输入结构。

## 创建DataFrame

可以利用Series字典或字典生成DataFrame。

```python
s1 = pd.Series([3, 1, 4, 7, 5], index=['a', 'e', 'i', 'o', 'u'])  # 一个Series
s2 = pd.Series([2, 4, 6, 9, 0], index=['a', 'b', 'c', 'd', 'e'])  # 一个Series
d = {"one": s1, "two": s2}  # 一个嵌套字典，d的类型是dict

df = pd.DataFrame(d)  # 利用嵌套字典生成DataFrame，七行两列

df = pd.DataFrame(d, index=['a', 'e'])  # 两行两列，行标签顺序与index相同

df = pd.DataFrame(d, columns=['two', 'three'])  # 五行两列，列标签为three的都是NaN

for i in df.index:  # 访问df的行标签
    print(i)

df.columns  # 访问df的列标签
```

同样也可以使用多维数组字典、列表字典生成DataFrame。

```python
d = {"one": [1, 2, 4, 5], "two": [9, 8, 7, 6]}  # 多维数组字典，需要保证长度一致

pd.DataFrame(d)  # 四行两列

pd.DataFrame(d, index=['a', 'e', 'i', 'o'])  # 四行两列有行标签

pd.DataFrame(d, index=['a', 'e', 'i', 'o', 'u'])  # 不匹配导致报错
```

或者使用列表字典，生成的结果与上文相同。

```python
d = [{'one': 1.0, 'two': 4.0}, {'one': 2.0, 'two': 3.0},{'one': 3.0, 'two': 2.0},{'one': 4.0, 'two': 1.0}]  # 列表字典

pd.DataFrame(d)  # 四行两列，其他操作也相同
```

利用元组字典可以生成多层嵌套的DataFrame。

```python
pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
              ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
              ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
              ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
              ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})

# 结果是一个多层嵌套的DataFrame
```

## 操作DataFrame

DataFrame 就像带索引的 Series 字典，提取、设置、删除列的操作与字典类似。

```python
d = {"one": [11, 12, 14, 15], "two": [29, 28, 27, 26], "three": [31, 32, 33, 34], "four": [44, 45, 46, 47]}
df = pd.DataFrame(d)

df['one']  # 提取列

df['three'] = df['one'] * df['two']  # 修改列

df['flag'] = df['one'] > 2  # 新增一列布尔

del df['two']  # 删除列

three = df.pop('three')  # 删除three列，并将他赋予three

df['foo'] = 'bar'  # 新增一列标量，以广播的方式填充

df['z'] = [1, 2, 3]  # 新增一列数组时，需要长度相同，否则报错

df['z'] = [1, 2, 3, 4]  # 不报错

s = pd.Series([1, 2, 3])
df['x'] = s  # 插入Series时，允许长度不一致，index不同的均为NaN

df.insert(1, 'bar', df['one'])  # 可以在指定位置insert新列
```

利用`assign()`创建新列

```python
d = {"one": [11, 12, 14, 15], "two": [29, 28, 27, 26], "three": [31, 32, 33, 34], "four": [44, 45, 46, 47]}
df = pd.DataFrame(d)

df.assign(five = lambda x:x['one'] + 1)  # 创建新列，是一个新的对象，不会对df进行更改。

df.query('two>26')  # 按照列进行筛选，输出三行四列

df.assign(five=lambda x: x['one'] + x['two'], six=lambda x: x['two'] + x['five'])  # 顺序执行，可以在第二次操作中执行第一次操作生成的列

df.assign(five=lambda x: x['one'] + x['two'])
  .assign(six=lambda x: x['two'] + x['three'])  # 对于不同的python版本可以使用这种方法，但是不像上行可以顺序执行
```

## 索引与选择

| 选择列           | `df[col]`       |  Series   |
| :--------------- | --------------- | :-------: |
| 用标签选择行     | `df.loc[label]` |  Series   |
| 用整数位置选择行 | `df.iloc[loc]`  |  Series   |
| 行切片           | `df[5:10]`      | DataFrame |
| 用布尔向量选择行 | `df[bool_vec]`  | DataFrame |

## 数据运算

DataFrame 对象可以自动对齐**列与索引（行标签）**的数据。与上文一样，生成的结果是列和行标签的并集。

```python
df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
df1 + df2  # 自动按列与行标签对齐，没有值的地方变成NaN

df1 - df1.iloc[0]  # DataFrame 和 Series 之间执行操作时，默认操作是在 DataFrame 的列上对齐 Series 的索引，按行执行广播操作
```

时间序列是特例，DataFrame 索引包含日期时，按列广播。

如果要执行其他的按列广播的减去一个`Series`的操作，需要使用`sub`。

```python
df1.sub(df1['A'], axis=0)  # 减去A列，并按列广播
```

标量操作和布尔操作和其他数据结构一样，每个元素单独运算。

```python
df1.T  # 获得df1的转置
```

