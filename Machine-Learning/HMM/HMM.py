import numpy as np

class HiddenMarkovModel:
    def __init__(self):
        pass

    def forward(self, Q, V, A, B, O, PI):
        '''
        Q: set of states
        V: set of observation 
        A: state transition matrix
        B: observation probability matrix
        O: observation sequence

        n: length of state sequence
        m: length of observation sequence
        '''
        n = len(Q)
        m = len(O)
