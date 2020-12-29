import os
import numpy as np



path="taxi_log_2008_by_id"  #待读取的文件夹
path_list=os.listdir(path)
#path_list.sort() #对读取的路径进行排序
count=0
temp=np.array([])

data_temp = np.loadtxt(open(os.path.join(path,path_list[0]),"rb"),dtype=str, delimiter=",", skiprows=0)
print(data_temp.shape)

# data_temp1 = np.loadtxt(open("taxi_log_2008_by_id\\1024.txt","rb"),dtype=str, delimiter=",", skiprows=0)
# print(data_temp1.shape)


for filename in path_list[1:]:
    count+=1
    data_name=os.path.join(path,filename)
    #print(data_name)

    method1_data = np.loadtxt(open(data_name, "rb"),dtype=str, delimiter=",", skiprows=0).reshape(-1,4)

    if method1_data.shape[0]>=1:
        data_temp = np.append(data_temp, method1_data, axis=0)



a=data_temp[:,2:].astype(float)
print(a)
np.savetxt('F:\\taxi.csv',a,delimiter=',')




print(count)

