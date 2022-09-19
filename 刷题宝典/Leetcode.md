# 树

## 概述

树是一种有向无环图，具有$N$个节点和$N-1$条边。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
```

## 树的遍历

- 理解和区分树的遍历方法

- 能够运用递归方法解决树的为前序遍历、中序遍历和后序遍历问题

- 能用运用迭代方法解决树的为前序遍历、中序遍历和后序遍历问题

- 能用运用广度优先搜索解决树的层序遍历问题

### 前序遍历

前序遍历首先访问根节点，然后遍历左子树，最后遍历右子树。

```python
## 递归解法：深度搜索
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        def dfs(r):
            if not r:
                return
            nonlocal ans # 表明不是本地标量，下面根左右
            ans.append(r.val)
            dfs(r.left)
            dfs(r.right)
        dfs(root)
        return(ans)

## 遍历解法：
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:  # stack非空
            node = stack.pop()
            if node:
                ans.append(node.val)  # 当前节点加入结果
                if node.right:
                    stack.append(node.right)  # 右节点入栈
                if node.left:
                    stack.append(node.left)  # 左节点入栈
        return ans
```

### 中序遍历

中序遍历是先遍历左子树，然后访问根节点，然后遍历右子树。

```python
## 递归解法：深度搜索
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        def dfs(r):
            if not r:
                return
            nonlocal ans # 表明不是本地标量，下面左根右
            dfs(r.left)
            ans.append(r.val)
            dfs(r.right)
        dfs(root)
        return(ans)
## 遍历，和前后不同，优先找到最左下角的子树
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = []
        ans = []
        node = root
        while stack or node:  # stack非空
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            ans.append(node.val)
            node = node.right
        return ans
```

### 后序遍历

后序遍历是先遍历左子树，然后遍历右子树，最后访问树的根节点。

```python
## 递归解法：深度搜索
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        def dfs(r):
            if not r:
                return
            nonlocal ans # 表明不是本地标量，下面左右根
            dfs(r.left)
            dfs(r.right)
            ans.append(r.val)
        dfs(root)
        return(ans)
    
## 遍历，利用前序进行翻转
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:  # stack非空
            node = stack.pop()
            if node:
                ans.append(node.val)  # 当前节点加入结果
                if node.left:
                    stack.append(node.left)  # 左节点入栈
                if node.right:
                    stack.append(node.right)  # 右节点入栈
        return ans[::-1]
```

值得注意的是，当你删除树中的节点时，删除过程将按照后序遍历的顺序进行。 也就是说，当你删除一个节点时，你将首先删除它的左节点和它的右边的节点，然后再删除节点本身。

另外，后序在数学表达中被广泛使用。 编写程序来解析后缀表示法更为容易。 

### 层序遍历

层序遍历就是逐层遍历树结构。

广度优先搜索是一种广泛运用在树或图这类数据结构中，遍历或搜索的算法。 该算法从一个根节点开始，首先访问节点本身。 然后遍历它的相邻节点，其次遍历它的二级邻节点、三级邻节点，以此类推。

当我们在树中进行广度优先搜索时，我们访问的节点的顺序是按照层序遍历顺序的。

```python
## 遍历，利用队列来进行访问，虽然根本没用到
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return[]
        ans = []
        queue = [root]
        while queue:
            childs = []
            temp = []
            for node in queue:
                temp.append(node.val)
                if node.left:
                    childs.append(node.left)
                if node.right:
                    childs.append(node.right)
            ans.append(temp)
            queue = childs
        return ans
```

## 利用递归解决问题

在前面的章节中，我们已经介绍了如何利用递归求解树的遍历。 递归是解决树的相关问题最有效和最常用的方法之一。

我们知道，树可以以递归的方式定义为一个节点（根节点），它包括一个值和一个指向其他节点指针的列表。 递归是树的特性之一。 因此，许多树问题可以通过递归的方式来解决。 对于每个递归层级，我们只能关注单个节点内的问题，并通过递归调用函数来解决其子节点问题。

通常，我们可以通过 “自顶向下” 或 “自底向上” 的递归来解决树问题。

### 自顶向下

“自顶向下” 意味着在每个递归层级，我们将首先访问节点来计算一些值，并在递归调用函数时将这些值传递到子节点。 所以 “自顶向下” 的解决方案可以被认为是一种前序遍历。 具体来说，递归函数 `top_down(root, params) `的原理是这样的：

```python
1. return specific value for null node
2. update the answer if needed                      // answer <-- params
3. left_ans = top_down(root.left, left_params)		// left_params <-- root.val, params
4. right_ans = top_down(root.right, right_params)	// right_params <-- root.val, params
5. return the answer if needed                      // answer <-- left_ans, right_ans
```

自顶向下意味着每层都解决一次问题，并把该层的结果变成参数传递给下一层。

### 自底向上

“自底向上” 是另一种递归方法。 在每个递归层次上，我们首先对所有子节点递归地调用函数，然后根据返回值和根节点本身的值得到答案。 这个过程可以看作是后序遍历的一种。 通常， “自底向上” 的递归函数 `bottom_up(root)` 为如下所示：

```python
1. return specific value for null node
2. left_ans = bottom_up(root.left)			// call function recursively for left child
3. right_ans = bottom_up(root.right)		// call function recursively for right child
4. return answers                           // answer <-- left_ans, right_ans, root.val
```

自底向上意味着每层都向上返回一层结果，该层结合下一层的结果计算这一层的结果并向上返回。

### 建议

了解递归并利用递归解决问题并不容易。

当遇到树问题时，请先思考一下两个问题：

- 你能确定一些参数，从该节点自身解决出发寻找答案吗？
- 你可以使用这些参数和节点本身的值来决定什么应该是传递给它子节点的参数吗？

*如果答案都是肯定的，那么请尝试使用 “自顶向下” 的递归来解决此问题。*

- 对于树中的任意一个节点，如果你知道它子节点的答案，你能计算出该节点的答案吗？ 

*如果答案是肯定的，那么 “自底向上” 的递归可能是一个不错的解决方法。*

# 图

## 概述

总而言之好像是咕咕咕了