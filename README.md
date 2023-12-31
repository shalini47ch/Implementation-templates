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
        
        


        
        



        



