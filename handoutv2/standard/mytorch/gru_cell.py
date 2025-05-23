import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        h_t = np.zeros_like(h_prev_t)
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        #store:  r, z, n
        self.r = self.r_act.forward(self.Wrx @ self.x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ self.x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act.forward(self.Wnx @ self.x + self.bnx + (self.r * (self.Wnh @ h_prev_t + self.bnh)))
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        x_t = self.x.reshape((self.d, 1)) #input dim = input size = self.d
        h_t_prev = self.hidden.reshape((self.h, 1)) # hidden dim = hidden size = self.h
        # 2) Transpose all calculated dWs...
        # print(delta.shape) #2,
        # 3) Compute all of the derivatives
        def pp(name: str, var):
            print(name + " is shape " + str(var.shape))
        # print(self.d) #5
        dldh = delta
        # pp("dldh", dldh) #2,
        dldz = dldh * (self.hidden - self.n).T
        # pp("dldz", dldz) #2,
        dldn = dldh * (np.ones_like(self.z) - self.z).T
        # pp("dldn", dldn) #2,
        dldn_act = self.h_act.backward(dldn)
        # pp("dldn_a", dldn_act) #2,
        self.dWnx = (dldn_act * x_t).T
        self.dbnx = dldn_act
        dldr = dldn_act.reshape(self.h,1) * (self.Wnh @ h_t_prev + self.bnh.reshape(self.h, 1))
        dldr = dldr.reshape(self.h,)
        # pp("dldr", dldr)
        
        # assert dldr.shape == (self.h,)
        # pp("Wnh", self.Wnh)
        # pp("htprev", h_t_prev)
        # pp("bnh", self.bnh)
        # pp("dldr", dldr)
        self.dWnh = (dldn_act * (self.r.T * h_t_prev)).T
        self.dbnh = dldn_act * (self.r.T)
        
        dldz_act = self.z_act.backward(dldz)
        self.dWzx = (dldz_act * x_t).T
        self.dbzx = dldz_act
        self.dWzh = (dldz_act * h_t_prev).T
        self.dbzh = dldz_act

        dldr_act = self.r_act.backward(dldr)
        assert dldr_act.shape == (self.h,)
        self.dWrx = (dldr_act * x_t).T
        self.dbrx = dldr_act
        self.dWrh = (dldr_act * h_t_prev).T
        self.dbrh = dldr_act
        
        #SANITY CHECK SIZE
        assert self.dWrx.shape == (self.h, self.d)
        assert self.dWzx.shape == (self.h, self.d)
        assert self.dWnx.shape == (self.h, self.d)
        assert self.dWrh.shape == (self.h, self.h)
        assert self.dWzh.shape == (self.h, self.h)
        assert self.dWnh.shape == (self.h, self.h)
        assert self.dbrx.shape == (self.h,)
        assert self.dbzx.shape == (self.h,)
        assert self.dbnx.shape == (self.h,)
        assert self.dbrh.shape == (self.h,)
        assert self.dbzh.shape == (self.h,)
        assert self.dbnh.shape == (self.h,)
        
        dx = dldn_act @ self.Wnx + dldz_act @ self.Wzx + dldr_act @ self.Wrx
        # self.n = self.h_act.forward((self.r * (self.Wnh @ h_prev_t)))
        dh_prev_t = dldh * self.z + dldn_act * self.r @ self.Wnh + dldz_act @ self.Wzh + dldr_act @ self.Wrh
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
