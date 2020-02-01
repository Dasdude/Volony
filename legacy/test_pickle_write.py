import cPickle as pickle
import numpy as np
import json
class Ds:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
def test_jason_write():
    a_dict = {}
    with open('test_json.json','w') as outfile:
        for frame_num in range(100):
            a_dict = {'type':'json',frame_num:Ds(frame_num%10,frame_num%5,frame_num%2)}
            json.dump(a_dict,outfile)

def test_pickle_write():
    for frame_num in range(10):
        with open('test_pickle.p', 'ab') as outfile:
            a_dict = {'frame_num':frame_num,frame_num:Ds(frame_num%10,frame_num%5,frame_num%2)}
            pickle.dump(a_dict,outfile,0)
def test_pickle_read():
    objs = {}
    with open('test_pickle.p','rb') as outfile:
        while 1:
            try:
                a = pickle.load(outfile)
                objs[a['frame_num']]=a
                print a
            except EOFError:
                break
    return objs
if __name__ == '__main__':
    with open('test_pickle.p','wb') as file:
        file.flush()
    test_pickle_write()
    a = test_pickle_read()
    print(a)


