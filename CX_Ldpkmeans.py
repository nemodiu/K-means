import numpy as np


def bernoulli(probability):
    arr_ber = np.random.rand(1, 1)
    if arr_ber<=probability:
        return 1
    else:
        return 0


def perturb(unperturb_list,privacy_cost):
    probility = np.exp(privacy_cost) / (np.exp(privacy_cost) + 1)
    print(probility)
    perturb_list = []
    for x in unperturb_list:
        if bernoulli(probility):
            perturb_list.append(x)
        else:
            perturb_list.append(1-x)


    return np.array(perturb_list).reshape(unperturb_list.shape)


def aggregeate(unaggregeate_list,privacy_cost):
    probility = 2 / (np.exp(privacy_cost) + 1)
    print(probility)
    list=np.sum(unaggregeate_list,axis=0)
    num=unaggregeate_list.size
    aggregeate_list=[]
    for x in list:
        a=(x-(0.5*probility*num))/(1-probility)
        aggregeate_list.append(a)

    print(aggregeate_list)
    return aggregeate_list









Epsilon=1

list1=np.ones((10000, 1))

list=np.append(list1,np.zeros((50000,1)))


perturb_list=perturb(list,Epsilon)
print(np.sum(perturb_list))

values=np.array([x for x in range(0,16)]).reshape(-1,4)
print(values)
aggregeate(values,Epsilon)