from activation import Tanh
from gate import AddGate, MultiplyGate
import numpy as np
mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):


        print('U dimension : ', np.array(U).shape)
        print('Single timestamp dimension : ', np.array(x).shape)
        self.mulu = mulGate.forward(U, x)
        print('U * X   : ', np.array(self.mulu).shape)
        print('-------------------------------------------------')
        print('W dimension : ', np.array(W).shape)
        print('Previous dimension : ', np.array(prev_s).shape)
        self.mulw = mulGate.forward(W, prev_s)
        print('W * previous  : ', np.array(self.mulw).shape)
        print('-------------------------------------------------')
        self.add = addGate.forward(self.mulw, self.mulu)
        print('(U*X)  + (W * previous)  : ', np.array(self.add).shape)
        self.s = activation.forward(self.add)
        print('-------------------------------------------------')
        print('(U*X)  + (W * previous) after tanH  : ', np.array(self.s).shape)
        self.mulv = mulGate.forward(V, self.s)
        print('-------------------------------------------------')
        print('V dimension : ', np.array(V).shape)
        print('( V * ((U*X)  + (W * previous)) )  : ', np.array(self.mulv).shape)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)