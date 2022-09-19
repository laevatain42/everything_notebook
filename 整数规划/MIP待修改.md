#  MIP Problem

##  Stone Problem - 分石头问题

给定n个石头，分成两堆石头，使两堆石头重量相等。

$$
\text{s.t.} \\
\sum_{i=1}^na_ix_i=0 \\
x_i=\{-1,1\},\forall i
$$

## Knapsack Problem - 背包问题

背包容量给定，装载价值最多的商品。

$$
max \sum_{j=1}^n c_jx_j \\
\text{s.t.} \\
\sum_{j=1}^n a_jx_j\le b \\
x_j = \{0, 1\},j=1...n
$$

## Assignment Problem - 指派问题

指派问题与线性松弛后的线性规划问题是等价的，因为其约束矩阵是全单模矩阵。

$$
min \sum_{i=1}^n\sum_{j=1}^n c_{ij}x_{ij} \\
\text{s.t.} \\
\sum_{j=1}^n x_{ij} = 1 \\
\sum_{i=1}^n x_{ij} = 1 \\
x_{ij}=
	\begin{cases}
		1, & \text{if proson i do job j}\\
		0, & other
	\end{cases}
$$

## Set Covering Problem - 集合覆盖问题

给定一定区域，需要建立一系列中心，来提供一系列服务：急救中心，火警中心等，需要满足一定的服务能力。每个地方都被覆盖。
$$
min \sum_{j=1}^n c_j x_j \\
\text{s.t.} \\
\sum_{j=1}^na_{ij}x_j\ge1;\forall i\\
a_{ij} =
	\begin{cases}
		1, &\text{如果中心j可以服务i点}\\
		0, & other
	\end{cases}
\\
x_j = 
	\begin{cases}
		1, & \text{在j点建立中心}\\
		0，& \text{在j点不建立中心}
	\end{cases}
$$

## Facility Location Problem - 选址问题

建立分销中心；服务中心等

i 为需求点；j为 中心点；$y_{ij}$为i需求点由j供给的比例；$x_j$为j中心点是否建立；$f_j$为j的建立成本。
$$
max\ \sum_i\sum_jc_{ij}y_{ij}-\sum_jf_jx_j \\
\text{s.t.} \\
\sum_jy_{ij} = 1m \\
y_{ij}\le x_j \\
x\in\{0, 1\}^n
$$

## Traveling Salesm Problem - 旅行商问题

对于每个节点：进去一次；出去一次；保证没有环

保证无环有两种方法：

1. 选取一个子集，该子集到其补集必有至少一条路径
2. 选取一个子集，其内部的路径少于节点数（子集无环）

只找一个可行解，为图论中的哈密尔顿圈

$$
x_{ij}=
	\begin{cases}
	1, & \text{从i旅行到j} \\
	0 & other
	\end{cases}\\
\sum_{i\ne j}x_{ij} = 1 \\
\sum_{j\ne i}x_{ij} = 1 \\ 
\sum_{i \in S}\sum_{j \in S}x_{ij} \le |S|-1
$$


## Vehicle Routing Problem - 车辆路径问题

车辆覆盖问题与TSP问题类似，不过要考虑车辆负责的节点。

## Generalized Assignment Problem - 广义指派问题

m个machine，n个jobs，不同的机器做不同的作业效率不同
$$
min \sum\sum c_{ij}x_{ij} \\
\text{s.t.} \\
\sum_j a_{ij}x_{ij} \le b_i \\
\sum_i x_{ij} = 1 \\
x_{ij} \in \{0, 1\}
$$

## Bin Packing Problem - 装箱问题
装箱问题在考虑空间是，需要考虑每种物体的拜访，具体的建模方式较为复杂，不列举了。


## Minimum Cost Network Flows - 最小费用网络流

*G = (V,A)*，$x_{ij}$表示i流向j
$$
min \sum_{(i,j) \in A}c_{ij}x_{ij} \\

\text{s.t.} \\

\sum_{k\in V_i^+} x_{ik} - \sum_{k \in V_i^-}x_{ki} = b_i, i\in V \\
0 \le x_{ij} \le h_{ij}
$$
起点$ b_i = 1$, 中间$ b_i = 0$, 终点$ b_i = -1$。

x_ij取整数与分数的结果相同，因为A矩阵全单模。

不仅最小费用网络流问题是全单模的，包括最大流，最小割的问题，所有网络问题都是全单模的。

## Maximun Flow Problem - 最大流最小割问题

构建从终点到起点的一条人工弧，使人工弧的容量最大，约束变为每个节点的流出和流入相等。
$$
max \ x_{ts} \\
\text{s.t.} \\
\sum_{k\in V_i^+} x_{ik} - \sum_{k \in V_i^-}x_{ki} = 0, i\in V \\
0 \le x_{ij} \le h_{ij}
$$

A全单模，引进y，（A，I）全单模，h整数，因此它也可以松弛，还可以写出对偶问题。	

对偶问题（是一个最小割问题）：
$$
min \sum_{(i,j) \in A} h_{ij}w_{ij} \\
\text{s.t.} \\
u_i-u_j+w_{ij} \ge 0, (i,j) \in A\\
u_t - u_s\ge1,w_{ij}\ge0,(i,j) \in A
$$
有节点数+弧数的对偶变量：对偶问题的矩阵也是全单模的。

最大流问题的对偶问题是一个最小割问题。

Ford-Fulkerson Algorithm可以求解。

## Shortest Path - 最短路问题

P问题，强多项式时间内求解。（如果没有负值弧）

是最小费用网络流的特例。

网络中的最短路问题，矩阵A是全单模的，因此其松弛问题的线性规划的解与整数规划的解相同。

定理：z是最短路当且仅当：

$$
\lambda_s = 0;\lambda_t=z;\lambda_j-\lambda_i\le c_{ij}
$$
$\lambda_i$是对应的节点$i$的平衡方程的对偶乘子。上述的三个式子即为对偶方程。
$$
(D)\ max\  \lambda_t-\lambda_s \\
\text{s.t.} \\
\lambda_j-\lambda_i\le c_{ij},(i,j)\in A
$$

上述用到了全单模性和对偶，其也是迪杰斯特拉算法的基础。

## Maximum Cut - 最大切割问题

给定一个无向图，将图分为两组，使两组之间的点的弧的权重最大。

表达异或；
$$
x_1 = \pm 1;x_2 = \pm 1 \\
1-x_1x_2 \ \text{表达整数变量的异或}
$$
x1x2在一组得0，不在一组得1

SDP可以求解上述问题

## Minimum spanning tree - 最小支撑树问题

树是无环路的图，等价定义：

1. 一个图是树，当且仅当任意两个节点都只有一条路径
2. 一个图是树，当且仅当每一条边都是桥
3. 一个图是树，当且仅当有V个节点和V-1条弧

最小支撑树（MST）：从一个图中找到一棵树，保证全联通，且权重最小。

求解算法：Kruskal’s Algorithm

## Minimum Striner Tree - 最小Striner数问题

NP-hard问题。

最小Striner树：允许新增节点的最小支撑树。

已被证明的猜想：最大可以增加2/根号3倍。

又被证明上述的证明中有一条引理是错的。因此还是开放的。

## Portfolio Optimization - 资产组合问题

资产组合时，要同时考虑回报和风险。

目标函数是二次函数。

# MIP Theory

## Linear Program - 线性规划

### Linear Program - 线性规划

标准型：
$$
\text{min} \ c^Tx \\
\text{s.t.}\\
\begin{aligned}
\  Ax &=b \\
			   x &\ge 0
\end{aligned}
$$
通常假设A行满秩。

线性规划的可行域是一个多面体，多面体的顶点(*extreme point*)均不可表现为其他可行解的凸组合。

线性函数的目标函数为定值时，表现为一个超平面，将超平面按其法向量移动，即将离开可行域时获得最优解。

### Simplex Method - 单纯形法

可行域：S，A行满秩，令$A=(B,N)$，其中B满秩，大小mxm。

令$x = (x_B, x_N)^T$，$x_B$为基变量，$x_N$为非基变量

有
$$
A = (B,N)\\
x = (x_B, x_N)^T\\
Ax = b\ =>(B,N)(x_B, x_N)^T \\
Bx_B + Nx_N = b\ =>x_B=B^{-1}b;x_N=0
$$
$(B^{-1},0)^T$为一个基解，如果基解同时是一个可行解，称之为基可行解。基可行解必为一个顶点。

几何上相邻的极点，在代数上表示为仅有一个基变量不同

线性规划基本定理，有顶点必有最优解，必有一个最优解是极点。

**检验数**

对于一个基可行解，选取任意一个解，可以用该基可行解和(B,N)表示其目标函数，有
$$
c^Tx=c^T \hat{x} + c_N^T-c_B^TB^{-1}x_N\\
r_N=c^T_N-c^T_BB^{-1}N
$$
$r_N$称为reduced cost，如果$r_N$>=0，则$c^Tx>c^Tx^*$,$x^*$已经是最优解。

注意在求r_N时，讲基变量对应的$c^T$列化为0。

### Dual of Linear Program - 线性规划问题的对偶

$$
P: min\ f(x)\\
\text{s.t.}\\
h(x)=0\\
x \in X
$$

这里h(x)是一个难约束，X是一个简单约束

构造对偶函数
$$
d(\lambda) = \ min_x\ d(x)=f(x)+\lambda h(x)
$$
对偶函数给出一个LB，有
$$
d(\lambda) \le f(x)
$$
求最高的低界的问题，即为对偶问题：
$$
D:max\ d(\lambda)
$$
D问题称为原问题的对偶。
$$
\begin{aligned}
(LP) \ & min\ c^Tx\\
& s.t.\ Ax=b,x\ge0\\
(LD) \ & max\ b^T\lambda\\
& s.t.A^T\lambda\le c
\end{aligned}
$$

## Easy Problem

假设一个图*G=(V,E)*，V个点，E条边。称一个算法对求解问题是有效的，如果他的复杂度是O($E^p$)。

Separation Problem：对于一个组合优化问题，如果给定一个点x，x不在可行域的凸包之中，就可以找到一个割平面，将x与可行域分隔开，称为**可分离性**。

Easy Problem4个性质：

1. 有一个有效的问题可以求解P。
2. 该问题有强对偶的问题。
3. 有效的分离性：存在一个有效的算法，找到分离问题的割平面。
4. 可以有效表示问题的可行域的凸包。

## Totally Unimodularity - 全单模

### 全单模定义

对MIP问题的松弛LP问题，何时取到相同的解？即：松弛LP问题的顶点都是整数点。

LP问题对应的多面体的订单都是整数点<=>基本可行解都是整数点。
$$
\begin{aligned}
(IP) \ & min\ c^Tx\\
& s.t.\ Ax\le b,x\ge0, x\text{整数}\\
(LP) \ & min\ c^Tx\\
& s.t.Ax\le b,x\ge0
\end{aligned}
$$
**线性规划可行域的顶点都是整数点<=>线性规划的最优解与整数规划相同**

基可行解<=>顶点
$$
Ax+y=b,x\ge0,y\ge0 \\
(A,I)(x,y)^T=b \\
(A,I)\  \text{构造}\ (B,N) \\
x^* = B^{-1}b\\
B^{-1}=|B|^{-1}B^* \\\text{$B^*$是伴随矩阵}
$$
有B的行列式是1，-1时，x*必为整数。

若最优时的基B是1，-1，则顶点是整数

因为最优的基初始不知道，即B不知道，则我们可以令：

当A的所有非奇异方阵的模都是1，-1时，满足最优的基是1，-1

A全单模（TU）：A的所有子方阵的模是1，-1，0。

一个线性规划问题的所有顶点都是整数点，说明其矩阵式全单模的。

### 全单模判断

A全单模 => 每个元素均为1，-1，0。

A全单模<=>A^T全单模

A全单模<=>(A,I)全单模

**充分条件**：

1. 每个元素均为1，-1，0

2. 每一列（或每一行，因转置不影响全单模）最多有两个非零元素。

3. 可以按行分为M1M2；对于有两个非零元素的列，M1中该列和与M2中该列和相等：

   M1：1，M2：1或

   M1：0，M2：-1，1

   又可表示为：当两个非零元素符号相同，在一组里，符号相反，在另一组里。

证明：不是全单模，则存在子矩阵不为1，-1，0。对于最小的该子矩阵，对其进行行变换，会与充分条件3相悖。

矩阵A全单模且b整数<=>整数规划与线性规划相同最优解。

## P & NP



# Basic Algorithms

## Branch-and-Bound Method - 分支定界算法

分支定界，也称为：Partical Enumeration（部分枚举法）

对比Total Enumeration（穷举法）更快。

分支时，求解线性松弛问题（因为好求）。

**剪枝规则：**

1. 找到了一个解是整数解，不继续分支

2. 找到一个不可行界，不继续分支

3. 某一支的下界（线性松弛问题的解）大于某一整数解的解，说明该支的解都差，不继续分支，舍去。

   即超过界。

EG1：背包问题

背包问题松弛之后永远只有一个分数变量，可证。

借的过程中，可以将分数部分固定为整数进行求解。

分支时先分解更好的那边。

**分枝定界使用过程中面临的问题：**

1. How to Branch：求解一个松弛问题之后，对哪个变量进行分枝。

   对分数变量，取分数变量的上整数和下整数，获得两个范围。

   选取离整数点最远的变量进行Branch：这样分，子问题与父问题的差别最大。

2. Whick Node to select 选择哪一支继续计算：

   目标：找到好的可行解，获得好的上下界，减少计算量，加快时间。

   - Best First：选择目标函数最小的进行分支
     - 缺点：不能热启动，上一个问题和下一个问题没有练习。对内存的管理麻烦。需要更大的存储空间。
   - Depth-First：深度优先（在还没有找到整数解的时候，可以用深度优先
     - 优点：比较快的求解到一个整数解；需要保存的节点信息更少，set-up cost更小。
     - 缺点：初始下界不好，会导致计算许多不需要计算的枝
   - Hybrid Strategie：混合
     - 先用Deep First找到一个整数点，获得LB，然后用Best First进行计算
   - Best Esimate：最好估计

## Dijkstra Algorithm - 迪杰斯特拉算法 - 最短路问题

最短路的一部分也必然是最短路：最短路上有点*k*，则*s*到*k*也必然是最短的。

该算法要求没有负权边。

算法过程：

将节点分为两组：已有最短路的节点集S，未求得最短路的节点集U，按照最短路径长度递增的顺序将节点加入S

1. 初始时，S中只包含初始点，从U中找到距离初始点最近的一个点（要求相邻），加入S
2. 更新U到初始点的距离，从U中找到距离S中点最近的一个点（要求相邻），加入S
3. 直到搜索到终点。

是一个从终点不断扩散的过程。每次侵染最近的点。

算法的复杂度：O(|V|^2)

## Ford-Fulkerson Algorithm - 增广路径法 - 最大流问题

也被称为Augmentation Path Algorithm，增广路径法。

迭代算法：

1. 先找到可行解，将可行解去掉，构造一个新的图
2. 在新图中找到一个可行解，再找到一个增广路径，再去掉，构造一个新的图
3. 直到无法再找到增广路径

## **Kruskal’s Algorithm** - 最小支撑树

1. 选取所有边中权重最小的，标红。

2. 选择次小的边，如果形成环，不管，再找，直到不找到环，标红。
3. 循环直到找完。

复杂度：$O(E^2)$

## Dynamic Programming - 动态规划

为什么动态规划节省了计算时间：储存了历史的解。

用来做多阶段决策问题。

### 最优性原理

Principle of optimality： 一个多阶段决策问题的每个阶段都是最优的。

不是所有的多阶段问题都满足上述principle。但是大部分都满足。

最优性原理应用于问题，而不是算法。

States：状态（nodes for which values need to be calculated）

Stages：阶段（steps which define the ordering）比如k

可以应用的几种问题：

### 最短路问题

到达v点的最短路 = min{到达v点相邻点i的最短路 + iv的直线距离}

m = |E|，n=|V|

计算d(v)的复杂度为O(m)，m为边数。

记k为到最多有几条边到达点j：
$$
D_k(j)=min\{D_{k-1}(j)，min_{i \in V^-(j)}[D_{k-1}(i) + c_{ij}]\}
$$
$D_k(j)$，经过最多k条弧。

算法流程

1. k=1 计算只通过1条弧到达所有点的最短路
2. 开始循环，状态转移方程如上。

k从1到n-1，不超过顶点个数。注：有n-1个弧时，所有包含起点和终点的树都计算过了。

故：动态规划的求解复杂度O(mn)

P.S. Dijkstra算法作为贪婪算法，和动态规划都可以求解最短路问题。他们之间的区别在于，贪婪算法可以求解一阶马尔可夫链问题，动态规划可以求解需要存储多阶段状态的多阶马尔科夫链。

### 0-1背包问题

阶段：装在物品的数量，从1开始，表示前k件物品（正因是前k件，可以节约计算量，同时也是阶段决策。）

状态：容量。

第一阶段：只能拿去第一件物品，容量逐渐扩展。

依次循环，知道容量到达给定上限。

### 整数背包问题 ----- 还没学 --- 第四讲

### 最优批量问题 ----- 还没学 --- 第四讲

# Advanced Algorithms

# Nonlinear IP

