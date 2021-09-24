import numpy as np

if __name__ == '__main__':
    s = -np.ones((1,1,5,1,1))
    print(s)
    print(s.shape)
    print(s)
    s.resize(1,1,2,1,1)

    print("**********\n")
    print(s)
    print(s[0].shape)
    print(s[0][0].shape)
    # s[0].reshape(1,1,1,1,1)
    # s = s.reshape((1,1,1,1,1))
    # print(s)