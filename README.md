# Implementation-templates

# Segment Tree Implementation

class NumArray:
    #the optimized approach is by using segment tree

    def __init__(self, nums: List[int]):
        self.n=len(nums)
        # self.arr=nums
        self.seg=[0]*(4*self.n+1)
        self.build(nums,0,0,self.n-1)
        #here the build will have parameters arr,ind,start,end

    #now create a helper function to build the segment tree
    def build(self,nums,ind,start,end):
        #base case if we have reached the leaf node of the tree
        if(start==end):
            self.seg[ind]=nums[start]
            return
        #here we have not reached the end index so find the mid and find the left and right substree
        mid=start+(end-start)//2
        self.build(nums,2*ind+1,start,mid)
        self.build(nums,2*ind+2,mid+1,end)
        #now we need to find the sum of the left and the right subtrees
        self.seg[ind]=self.seg[2*ind+1]+self.seg[2*ind+2]
        #here it is the sum of the left and the right subtree,the left one is 2*i+1 and right one is 2*i+2

    #now create a helperfunction to update
    def _update(self,ind,start,end,target_ind,val):
        #here we need to update the element at target index to value
        if start==target_ind==end:
            self.seg[ind]=val
            return self.seg[ind]
        #check for the range if out of range then directly return the answer
        if(start>target_ind or end<target_ind):
            return self.seg[ind]
        mid=start+(end-start)//2
        self._update(2*ind+1,start,mid,target_ind,val)
        self._update(2*ind+2,mid+1,end,target_ind,val)
        self.seg[ind]=self.seg[2*ind+1]+self.seg[2*ind+2]
        return self.seg[ind]
    
    #now create a helper function to perform the query
    def query(self,ind,start,end,qstar,qend):
        #check whether the node range is there is query range or not
        if(start>=qstar and end<=qend):
            return self.seg[ind]
        #as it is out of range
        if(start>qend or end<qstar):
            return 0
        #not inside the range completly
        mid=start+(end-start)//2
        left=self.query(2*ind+1,start,mid,qstar,qend)
        right=self.query(2*ind+2,mid+1,end,qstar,qend)
        return left+right

        
    def update(self, index: int, val: int) -> None:
        return self._update(0,0,self.n-1,index,val)
        

    def sumRange(self, left: int, right: int) -> int:
        return self.query(0,0,self.n-1,left,right)




# Sweep line algorithm 

## used when we are given intervals and we need to find the intersection points or checking the cases of double booking triple booking. Examples like calendar 1 ,2,3
        You are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a double booking.

A double booking happens when two events have some non-empty intersection (i.e., some moment is common to both events.).

The event can be represented as a pair of integers start and end that represents a booking on the half-open interval [start, end), the range of real numbers x such that start <= x < end.

Implement the MyCalendar class:

MyCalendar() Initializes the calendar object.
boolean book(int start, int end) Returns true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.
 

Example 1:

Input
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
Output
[null, true, false, true]

Explanation
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False, It can not be booked because time 15 is already booked by another event.
myCalendar.book(20, 30); // return True, The event can be booked, as the first event takes every time less than 20, but not including 20.

#if the end is not inclusive  then take the end as it is and if it is inclusive then take the index as end+1 and reduce the end+1 and increase start+1

## This is my calender 1
class MyCalendar:

    def __init__(self):
        self.hmap=defaultdict(int)
        

    def book(self, start: int, end: int) -> bool:
        #increase the start by 1 and end+1 but here we dont have to incliude end
        self.hmap[start]+=1
        self.hmap[end]-=1
        su=0
        #nowthe next step is to iterate over the hmap
        for key,value in sorted(self.hmap.items()):
            su+=value
            if(su>1):
                #here reseting the value using line sweep algorithm
                self.hmap[start]-=1
                self.hmap[end]+=1
                return False
        return True
        





# My calendar 2  here we dont have to do triple booking
class MyCalendarTwo:

    def __init__(self):
        self.hmap=defaultdict(int)
        

    def book(self, start: int, end: int) -> bool:
        #here we dont have to do triple booking
        #here also the start interval is inclusive but the end is not
        self.hmap[start]+=1
        self.hmap[end]-=1
        su=0
        for key,value in sorted(self.hmap.items()):
            su+=value
            if(su>2):
                #means there is triple booking reset the values
                self.hmap[start]-=1
                self.hmap[end]+=1
                return False
        return True

        



## My calender 3 

class MyCalendarThree:

    def __init__(self):
        #we can solve this using the line sweep algorithm
        self.hmap=defaultdict(int)
        

    def book(self, startTime: int, endTime: int) -> int:
        #here the end time is not inclusive
        self.hmap[startTime]+=1
        self.hmap[endTime]-=1
        su=0
        #sweep line will work in case of ordered map so it is important to sort the hmap
        k=0
        for key,value in sorted(self.hmap.items()):
            su+=value
            k=max(k,su)
        return k


        



# Implementation  of Trie


class Node:
    def __init__(self):
        self.links=[None]*26 #an array of 26 charcaters
        self.flag=False

    def containsKey(self,ch):
        return self.links[ord(ch)-ord("a")]!=None
    
    def put(self,ch,node):
        self.links[ord(ch)-ord("a")]=node
    def get(self,ch):
        return self.links[ord(ch)-ord("a")]
    
    def setEnd(self):
        self.flag=True
    
    def isEnd(self):
        return self.flag
class Trie:

    def __init__(self):
        self.root=Node()
        
    def insert(self, word: str) -> None:
        node=self.root
        #we need to insert the characters in a trie
        for i in range(0,len(word)):
            if not (node.containsKey(word[i])):
                node.put(word[i],Node())
            #now the next step is to move to the reference node
            node=node.get(word[i])
        node.setEnd() #as the word ends the flag is set to true
        

    def search(self, word: str) -> bool:
        node=self.root
        #traverse through the given word
        for i in range(0,len(word)):
            if(not node.containsKey(word[i])):
                return False
            node=node.get(word[i])
        return node.isEnd()

        

    def startsWith(self, prefix: str) -> bool:
        #here we need to find if the word starts with the prefix or not
        node=self.root
        for i in range(0,len(prefix)):
            if(not node.containsKey(prefix[i])):
                return False
            #move to the reference
            node=node.get(prefix[i])
        return True

# Implementation of DSU

class DisjointSet:

    def __init__(self,n):
        self.parent=[i for i in range(n+1)]
        self.rank=[0 for i in range(n+1)]

    def findparent(self,node):
        if(node==self.parent[node]):
            return node
        self.parent[node]=self.findparent(self.parent[node])
        return self.parent[node]

    def unionbyrank(self,u,v):
        upu=self.findparent(u)
        upv=self.findparent(v)
        if(self.rank[upu]<self.rank[upv]):
            self.parent[upu]=upv
        elif(self.rank[upv]<self.rank[upu]):
            self.parent[upv]=upu
        else:
            self.rank[upu]+=1
            self.parent[upv]=upu


## Best time to buy and sell stocks (DP ON STOCKS)

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.




class Solution:

    def maxProfit(self, prices: List[int]) -> int:
        #here you need to return the max profit you can achieve
        n=len(prices)
        dp=[[-1 for i in range(2)]for j in range(n+1)]
        return self.helper(0,1,prices,n,dp)

    def helper(self,ind,buy,prices,n,dp):
        if(ind==n):
            return 0
        if(dp[ind][buy]!=-1):
            return dp[ind][buy]
        if(buy):
            profit=max(-prices[ind]+self.helper(ind+1,0,prices,n,dp),
            0+self.helper(ind+1,1,prices,n,dp))
        else:
            #here is the case of selling so here we put money in the market
            profit=max(prices[ind]+self.helper(ind+1,1,prices,n,dp),
            0+self.helper(ind+1,0,prices,n,dp))
        dp[ind][buy]=profit
        return dp[ind][buy]


## Morris Traversal helps us to perform tree traversal in O(1) space and it is the most optimized way to perform tree traversals

# Inorder Traversal through Morris Traversal



class Solution:

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    
        #most optimized approach is by using morris traversal where the time complexity will be O(N) but space will be O(1)
        curr=root
        ans=[]
        #keep iterating until the curr!=None
        while(curr!=None):
            leftNode=curr.left
            if(leftNode==None):
                #matlab right mei move karna hai
                ans.append(curr.val)
                curr=curr.right
            else:
                #toh isme se uska rightmost node nikalo
                rightmostNode=self.findright(leftNode,curr)
                if(rightmostNode.right==None):
                    #matlab thread nai hai to thread banao
                    rightmostNode.right=curr
                    curr=curr.left
                else:
                    #means thread already exists so we need to break that thread
                    rightmostNode.right=None
                    ans.append(curr.val)
                    curr=curr.right
        return ans

    #now creating a helper function to find the rightmost most
    def findright(self,leftNode,curr):
        while(leftNode.right!=None and leftNode.right!=curr):
            #matlab leftNode.right exists and it is not equal to the curr node
            #move the leftNode to the right
            leftNode=leftNode.right
        return leftNode


# Preorder traversal using Morris Traversal


class Solution:

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        #the optimized approach is by using morris traversal
        #preorder is root->left->right
        ans=[]
        curr=root
        while(curr!=None):
            leftNode=curr.left
            if(leftNode==None):
                ans.append(curr.val)
                curr=curr.right
            else:
                rightmostnode=self.findright(leftNode,curr)
                if(rightmostnode.right==None):
                    #means we need to create a thread
                    ans.append(curr.val)
                    rightmostnode.right=curr
                    curr=curr.left
                else:
                    #here break the thread
                    rightmostnode.right=None
                    curr=curr.right
        return ans

    def findright(self,leftNode,curr):
        while(leftNode.right!=None and leftNode.right!=curr):
            leftNode=leftNode.right
        return leftNode
            






    #     #root->left->right
    #     ans=[]
    #     self.helper(root,ans)
    #     return ans





    # def helper(self,root,ans):
    #     if root is None:
    #         return ans
    #     ans.append(root.val)
    #     self.helper(root.left,ans)
    #     self.helper(root.right,ans)
    #     return ans




# Diagonal Traversal of a Binary Tree(Clockwise)

Given a Binary Tree, print the diagonal traversal of the binary tree.

Consider lines of slope -1 passing between nodes. Given a Binary Tree, print all diagonal elements in a binary tree belonging to same line.
If the diagonal element are present in two different subtress then left subtree diagonal element should be taken first and then right subtree. 

Example 1:

Input :
            8
         /     \
        3      10
      /   \      \
     1     6     14
         /   \   /
        4     7 13
Output : 8 10 14 3 6 7 13 1 4


'''
# Node Class:

class Node:

    def _init_(self,val):
        self.data = val
        self.left = None
        self.right = None
'''
#Complete the function below

from collections import deque

class Solution:

    def diagonal(self,root):
    
        #:param root: root of the given tree.
        #return: print out the diagonal traversal,  no need to print new line
        #code here
        #we will solve it using bfs logic 
        queue=deque()
        queue.append(root)
        ans=[] #this will store the diagonal traversal
        #keep iterating until the queue is empty
        while(queue):
            n=len(queue)
            temp=[]
            for i in range(0,n):
                node=queue.popleft()
                while(node!=None):
                    temp.append(node.data)
                    if(node.left!=None):
                        #matlab abhi left wala hai to usse queue mei daaldo
                        queue.append(node.left)
                    node=node.right
            ans.extend(temp)
        return ans



# Diagonal Traversal of a Binary Tree(Anticlockwise)

from collections import deque

class Solution:

    def diagonal(self,root):
        #:param root: root of the given tree.
        #return: print out the diagonal traversal,  no need to print new line
        #code here
        #we will solve it using bfs logic 
        queue=deque()
        queue.append(root)
        ans=[] #this will store the diagonal traversal
        #keep iterating until the queue is empty
        while(queue):
            n=len(queue)
            temp=[]
            for i in range(0,n):
                node=queue.popleft()
                while(node!=None):
                    temp.append(node.data)
                    if(node.right!=None):
                        #matlab abhi left wala hai to usse queue mei daaldo
                        queue.append(node.right)
                    node=node.left
            ans.extend(temp)
        return ans



#  Redundant Connection 1 And 2 it is based on Union Find Logic


In this problem, a tree is an undirected graph that is connected and has no cycles.

You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.

Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.




class DisjointSet:

    def __init__(self,n):
        self.parent=[i for i in range(n+1)]
        self.rank=[0 for i in range(n+1)]
    #now the next step is to find the parent of the node
    def findparent(self,node):
        if(node==self.parent[node]):
            return node
        self.parent[node]=self.findparent(self.parent[node])
        return self.parent[node]
    #now the next one is to find the union by rank
    def unionbyrank(self,u,v):
        #find the ultimate parent of u and v
        upu=self.findparent(u)
        upv=self.findparent(v)
        if(self.rank[upu]<self.rank[upv]):
            self.parent[upu]=upv
        elif(self.rank[upv]<self.rank[upu]):
            self.parent[upv]=upu
        else:
            self.parent[upv]=upu
            self.rank[upu]+=1

        
class Solution:

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        #here we will use the logic of disjointSet union and the nodes which have the same parent can be removed
        ans=[]
        n=len(edges)
        ds=DisjointSet(n)
        for u,v in edges:
            if(ds.findparent(u)==ds.findparent(v)):
                ans.extend([u,v])
            ds.unionbyrank(u,v)
        if(len(ans)==0):
            ans.extend([-1,-1])
        return ans



## Catalan Numbers 



C0=1

C1=1

C2=C0C1+C1C0

C3=C0C2+C1C1+C2C0


C4=C0C3+C1C2+C2C1+C3C0  #these are the formulas for catalan number


class Solution:

  def nthcatalan(self,n):
  
      #first create a dp array
      dp=[0 for i in range(n+1)]
      dp[0]=1
      dp[1]=1
      for i in range(2,len(dp)):
          dp[i]+=d[j]*dp[i-j-1]
      return dp[n]


## No of digit one (Based onthe concept of digit dp)

Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

 

Example 1:

Input: n = 13
Output: 6
Example 2:

Input: n = 0
Output: 0

# Digit dp is mainly applied where we are provided with certain ranges and then either we need to count the digits equal to 1 3 so one or some other condtion.There will be repeating subproblems



class Solution:

    def countDigitOne(self, n: int) -> int:
    
        #here we are given the range from 0 till n and we need to return the count of 1s in it this is the problem of digit dp
        dp=[[[-1 for i in range(12)]for j in range(2)]for k in range(12)]
        ans=str(n)
        return self.solve(ans,0,1,0,dp)
        

    def solve(self,s,ind,flag,count,dp):
        if(ind==len(s)):
            return count
        #if we have already got the answer then return that
        if(dp[ind][flag][count]!=-1):
            return dp[ind][flag][count]
        #flag 1 means restricted and 0 means not restricted
        if(flag==1):
            limit=ord(s[ind])-ord("0")
        else:
            limit=9
        #now traverse through the limits
        ans=0
        for i in range(0,limit+1):
            updatecount = count + (1 if i == 1 else 0)
            ans += self.solve(s,ind+1,flag &(i == ord(s[ind]) - ord("0")), updatecount, dp)
        dp[ind][flag][count]=ans
        return dp[ind][flag][count]



## Delete nodes and return forest

Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest. You may return the result in any order.

 

Example 1:


Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]
Example 2:

Input: root = [1,2,4,null,3], to_delete = [3]
Output: [[1,2,4]]




class Solution:

    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
    
        #use the logic of dfs to solve this first move to the left part and then move to the right part
        hset=set()
        result=[]
        #put the elements of to_delete in hset
        for ele in to_delete:
            hset.add(ele)
        self.helper(root,hset,result)
        if(root.val not in hset):
            result.append(root)
        return result 

    #now the next step is to create a helper function that will delete
    def helper(self,root,hset,result):
        #base case
        if root is None:
            return None
        #now we will first move to the left tree and then to the right tree
        root.left=self.helper(root.left,hset,result)
        root.right=self.helper(root.right,hset,result)
        if(root.val in hset):
            #so we need to put the left and right node in the result and delete the root.val
            if(root.left!=None):
                result.append(root.left)
            if(root.right!=None):
                result.append(root.right)
            return None
        else:
            return root

## Trapping rain water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

 Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.



class Solution:

    def trap(self, height: List[int]) -> int:
        n=len(height)
        leftgreater=self.greaterleft(height,n)
        rightgreater=self.greaterright(height,n)
        width=1
        su=0 #at last we need to return the sum of the heights
        for i in range(0,n):
            h=min(leftgreater[i],rightgreater[i])-height[i]
            #now after this we need to add the area to the su
            su+=(width*h)
        return su

    #now creating two helper functiom to findgreater on left and greater on right
    def greaterleft(self,height,n):
        #the first index element doesn't have anyone on left
        left=[0 for i in range(n)]
        left[0]=height[0]
        #now filling the other half 
        for i in range(1,n):
            left[i]=max(left[i-1],height[i])
        return left 

    #now similarly create a helper function to find the maximum on right
    def greaterright(self,height,n):
        right=[0 for i in range(n)]
        right[n-1]=height[n-1] #as there are no elements 
        for i in range(n-2,-1,-1):
            right[i]=max(right[i+1],height[i])
        return right

# Stack Implementation using Arrays 

The intution here is that we will create an array of size as asked in the question and use a top variable with a value assigned and when we need to push the element in the stack we will increment the top by 1 and put the element in stack[top]=x and while popping we reduce the top by top-1



class MyStack:

    def __init__(self):
        self.arr=[-1 for i in range(1000)]
        self.top=-1
    
    #Function to push an integer into the stack.
    def push(self,data):
        #add code here
        self.top+=1
        self.arr[self.top]=data
    
    #Function to remove an item from top of the stack.
    def pop(self):
        #add code here
        if(self.top==-1):
            return -1
        x=self.arr[self.top]
        self.top-=1
        return x

## Implement queue using arrays 

Here we will take two pointers called as start and end while pushing we will move the end pointer and while popping we will modify the start pointer and then check according to the currsize and the original size



class MyQueue:
    def __init__(self):
        self.currsize=0
        self.size=100005
        self.arr=[-1 for i in range(self.size)]
        self.start=-1
        self.end=-1
        
        
    
    #Function to push an element x in a queue.
    def push(self, x):
        if(self.currsize==self.size):
            #matlabpush nai karsakte hai 
            return -1
        if(self.currsize==0):
            self.start=0
            self.end=0
        else:
            self.end=(self.end+1)%self.size
        self.arr[self.end]=x
        self.currsize+=1
        return 0
            
        
    #Function to pop an element from queue and return that element.
    def pop(self): 
        #this helps us to remove theelement from the queue
        if(self.currsize==0):
            return -1
        ele=self.arr[self.start]
        if(self.currsize==1):
            self.start=-1
            self.end=-1
        else:
            self.start=(self.start+1)%self.size
        self.currsize-=1
        return ele

## NGE TO THE RIGHT WITH TWO ARRAYS

The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

 

Example 1:

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]



class Solution:

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #use hmap and stack to solve this 
        #use the same logic as nearest greater to right 
        n=len(nums1)
        m=len(nums2)
        ngr=[-1 for i in range(m)]
        stack=[]
        hmap=defaultdict(int)
        #traverse in the reverse direction
        for i in range(m-1,-1,-1):
            while(len(stack)>0 and stack[-1]<=nums2[i]):
                #matlab smaller element mila hai toh isko pop karna hai
                stack.pop()
            #yahan wo case hai jahan greater mila hoga
            if(len(stack)==0):
                ngr[i]=-1
            else:
                ngr[i]=stack[-1]
            stack.append(nums2[i])
        #now iterate through the nums2 array and store the indexes
        for i in range(0,m):
            hmap[nums2[i]]=i #here we store the element and the indexes
        #now find the specific elements from nums1
        ans=[]
        for i in range(0,n):
            ele=nums1[i]
            val=hmap[ele]
            ans.append(ngr[val])
        return ans

        



        
            
        
        
    
        
        
            
       
        
        
        


        
        



        



