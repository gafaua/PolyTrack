from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise

import numpy as np


points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=3)

def hx(x):
    return x[[0,3]]

def fx(x, dt):
    r"""
    State transition function

    x has the form [x, dx/dt, d2x/dt2, y, dy/dt, d2y/dt2]
    dt is the delta time, amount of time between state transitions
    """
    dt2 = (dt*dt)/2
    F = np.array([[1, dt, dt2, 0,  0,   0],
                  [0,  1,  dt, 0,  0,   0],
                  [0,  0,   1, 0,  0,   0],
                  [0,  0,   0, 1, dt, dt2],
                  [0,  0,   0, 0,  1,  dt],
                  [0,  0,   0, 0,  0,   1]])
    return F @ x

class UKF_Tracker(object):
    def __init__(self, pos, dt):
        self.ukf = UKF(dim_x=6, dim_z=2, dt=dt, hx=hx, fx=fx ,points=points)

        # Standard deviation confidence for centers prediction TODO check if value is ok
        pos_std = 20

        # Measurement noise matrix 2x2 for measurements of x and y positions
        self.ukf.R = np.diag([pos_std**2, pos_std**2])
        # Process noise matrix, TODO check var value is ok, possibility to use Q_continuous_white_noise()
        self.ukf.Q = Q_discrete_white_noise(dim=3, dt=dt, block_size=2, var=0.01)
        #self.ukf.Q = Q_continuous_white_noise(dim=3,block_size=2)

        # Init
        self.ukf.x = np.array([pos[0], 0, 0., pos[1], 0, 0.]) # Initial state
        self.ukf.P = np.diag([10**2, 20**2, 20**2, 10**2, 20**2, 20**2])  # Covariance matrix, at first we assume that all variables are independant, thus using a diagonal matrix

        self.state_history = []

    def predict(self, dt=None):
        self.ukf.predict(dt=dt)
        # TODO maybe add some checks to the covariance matrix to verify uncertainty 
        # about the current state
        state = self.ukf.x
        
        self.state_history.append(state)
        return state

    def update_match(self, pos, dt=None):
        self.ukf.predict(dt=None)
        self.ukf.update(pos)
        state = self.ukf.x
        
        self.state_history.append(state)
        return state
    