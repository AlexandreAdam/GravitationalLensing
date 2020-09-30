import numpy as np
"""
beta: Source plane angular coordinate
theta: Image plane coordinate
kappa: Projected surface mass density, in units of the critical density Sigma
"""

class Theta:
    """
    Object to store theta x and y coordinates [theta_x, theta_y]
    """
    def __init__(self, theta_x, theta_y):
        self._x = theta_x
        self._y = theta_y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

def coordinates(N: int) -> Theta:
    """
    Generate the image plane coordinates
    """
    x = np.arange(0, N) - N // 2 - (N%2 - 1)//2
    xx, yy = np.meshgrid(x, x)
    out = Theta(xx, yy)
    return out

def alpha(theta: Theta, kappa: np.ndarray) -> np.ndarray:
    """
    Given a meshgrid of N pixel, this takes O(N^4) operations to compute
    """
    return



def lens_equation(theta: Theta, alpha: np.ndarray) -> np.ndarray:
    # Returns beta
    return theta - alpha
