import numpy as np

def convert_num_base(number_array,base):
    """

    :type number_array: numpy.array
    """
    representation = np.zeros_like(number_array)
    residual = np.expand_dims(number_array,-1)
    representation_list = []
    while np.any(residual)>0:
        digit = residual%base
        residual = residual//base
        representation_list+=[digit]
    representation_list.reverse()
    representation= np.concatenate(representation_list,-1)
    return representation

if __name__ == '__main__':
    a = np.array([[2,3],[4,5]])
    print convert_num_base(a,4)