import torch





if __name__ == '__main__':
    # test = {1:Image(id=1,qvec=np.array([0.9,0.006,-0.097,-0.03]),tvev=np.array([5.73230413, -2.03552609,  1.48107704])),
    # 35:Image(id=2,qvec=np.array([0.29,0.2006,-0.2097,-0.203]),tvev=np.array([25.73230413, -22.03552609,  21.48107704]))}
    # # 总而言之enumerate就是枚举的意思，把元素一个个列举出来，第一个是什么，第二个是什么，所以他返回的是元素以及对应的索引。
    # for idx, key in enumerate(test):
    #     print(idx)
    scales = torch.log(torch.sqrt(torch.tensor([0.0003])))
    print(scales)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


