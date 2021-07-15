from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np


points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=3) # TODO change those parameters

def hx(x):
    return x[[0,3]]

def fx(x, dt):
    r"""
    State transition function

    x has the form [x, dx/dt, d2x/dt2, y, dy/dt, d2y/dt2]
    dt is the delta time, amount of time between state transitions
    """
    dt2 = (dt**2)/2
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
        
        self.ukf.R = np.diag() # TODO add measurement noise matrix 2x2 for measurements of x and y positions
        self.ukf.Q = np.diag() # TODO add process noise matrix , using Q_discrete_white_noise()?

        # Init
        self.ukf.x = np.array([pos[0], avg_speed_x, 0., pos[1], avg_speed_y, 0.]) # TODO add initial state
        self.ukf.P = np.diag([])  # TODO create covariance matrix, at first we assume that all variables are independant, thus using a diagonal matrix

    def predict(self, dt=None):
        self.ukf.predict(dt=dt)
        
        
    
