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
        
        



