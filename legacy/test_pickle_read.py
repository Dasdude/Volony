import cPickle as pickle
import numpy as np
from test_pickle_write import Ds
if __name__ == '__main__':
    a = pickle.load(open('test_dump.p','rb'))
    print a
    # for frame_num in range(100):
    #     a_dict = {frame_num:Ds(frame_num%8,frame_num%4,frame_num%2)}

