'''
2024-1-25
The number of files in IFS is 728, this script is to find which day was missed
'''
import os

list1 = os.listdir('/data5/2019swh/liuxl_sailing/IFS')
list2 = os.listdir('/data5/2019swh/liuxl_sailing/GFS')
list3 = []

for i in list1:
    if i in list2:
        print('yes')
    else:
        print('no, the miss file is {}'.format(i))
        list3.append(i)

#print(len(list1))
#print(list2)