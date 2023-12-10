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
        


## Your MyCalendar object will be instantiated and called as such:
##  obj = MyCalendar()
##  param_1 = obj.book(start,end)


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

        


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)

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


        


## Your MyCalendarThree object will be instantiated and called as such:
## obj = MyCalendarThree()
## param_1 = obj.book(startTime,endTime)


        



