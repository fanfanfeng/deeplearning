
#
def quickSort(arr):
    if len(arr) <=1:
        return  arr

    privot = arr[int(len(arr)/2)]

    left = [x for x in arr if x < privot]
    midle = [x for x in arr if x == privot]
    right = [x for x in arr if x > privot]

    return  quickSort(left) + midle + quickSort(right)

if __name__ == '__main__':
    arr = [3,6,8,10,1,2,1]
    print(quickSort(arr))