import random
import math
def get_randomNumber(size):
    lists = []
    for _ in range(size):
        lists.append(random.randint(0,100))
    return lists

#调整堆
def adjust_heap(lists,i,size):
    left_child = 2*i
    right_child = 2*i + 1
    max = i
    if i< size/2:
        if left_child <size and lists[left_child] > lists[max]:
            max = left_child

        if right_child <size and lists[right_child] >lists[max]:
            max = right_child

        if max != i:
            lists[max],lists[i] = lists[i],lists[max]
            adjust_heap(lists,max,size)

#创建堆
def build_heap(lists,size):
    for i in list(range(0,int(size/2) +1))[::-1]:
        adjust_heap(lists,i,size)

def heap_sort(lists):
    size = len(lists)
    build_heap(lists,size)
    for i in list(range(0,size))[::-1]:
        lists[0],lists[i] = lists[i],lists[0]
        adjust_heap(lists,0,i)
    return lists

if __name__ == '__main__':
    a = get_randomNumber(10)
    print("排序之前:%s" % a)

    b = heap_sort(a)
    print("排序之后:%s" % b)


