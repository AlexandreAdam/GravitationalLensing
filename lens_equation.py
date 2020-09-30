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
        self.x = theta_x
        self.y = theta_y

    def rescale(self, value):
        self.x *= value
        self.y *= value



class Kappa:
    """
    Object to store the mass function and underlying coordinates
    """
    def __init__(self, profile, coordinates: Theta):
        """
        Profile should be a function of the form
            Theta_x, Theta_y -> kappa
        Where kappa is a mesh of real numbers
        """
        self.profile = profile
        self.coordinates = coordinates
        self._kappa = profile(coordinates)

    @property
    def kappa(self):
        return self._kappa

    # TODO work on this setter, it is not that simple.. Its gonna be ok for
    # SIS profile, unless I unload the Sigma integral to profile...
    @kappa.setter
    def kappa(self, coordinates, profile=None):
        if profile is None:
            pi = self.profile
        else:
            pi = profile
        print(f"Recalculating projected surface mass density using given "
        "coordinates and profile {pi.__name__}")
        self._kappa = pi(coordinates)


def sis_profile(theta):
    """
    Singular Isothermal Sphere
    """
    return 1 / 2 / (np.sqrt(theta.x**2 + theta.y**2) + 1e-6)

def coordinates(N: int) -> Theta:
    """
    Generate the image plane coordinates
    """
    x = np.arange(0, N, dtype=np.float32) - N // 2 - (N%2 - 1)/2
    xx, yy = np.meshgrid(x, x)
    theta = Theta(xx, yy)
    return theta

def _alpha(theta: Theta, kappa: Kappa) -> np.ndarray:
    """
    Given a meshgrid of N pixel, this takes O(N^4) operations to compute
    """
    theta = np.stack([theta.x, theta.y], axis=-1)
    kappa = kappa.kappa[..., np.newaxis, np.newaxis, np.newaxis]
    N = theta.shape[0] # number of pixels
    assert N <= 32, "Shape is too large, N <= 32"
    theta_prime = np.stack([theta] * N, axis=-1)
    theta_prime = np.stack([theta_prime] * (N - 1), axis=-1) # add 2 dimensions
    # to integrate over theta_prime, we make N(N - 1) copies of theta and shift
    # them by 1, such that each copies has a different theta_{ij} in a given position.
    for i in range(0, N):
        for j in range(1, N):
            theta_prime[..., i, j-1] = np.roll(theta_prime[..., i, j-1], j, axis=1)
        theta_prime[..., i, j-1] = np.roll(theta_prime[..., i, j-1], i, axis=0)

    theta = theta[..., np.newaxis, np.newaxis] + 1e-6 # prepare for broadcast
    norm = np.sqrt(theta**2 + theta_prime**2)
    alpha = np.sum(kappa * (theta - theta_prime) / norm, axis=(3, 4)) # return a 2d tensor
    return alpha


def lens_equation(theta: Theta, alpha: np.ndarray) -> np.ndarray:
    # Returns beta
    theta = np.stack([theta.x, theta.y], axis=-1)
    return theta - alpha


if __name__ == "__main__":
    from visualization import lens_and_source
    import matplotlib.pyplot as plt
    N = 12  # 32 is about the maximum the double integral can handle
    theta = coordinates(N)
    theta.rescale(10)
    kappa = Kappa(sis_profile, theta)
    alpha = _alpha(theta, kappa)
    beta = lens_equation(theta, alpha)
    im = plt.imshow(kappa.kappa)
    plt.colorbar(im)
    plt.show()
    im = plt.imshow(alpha[..., 0]**2 + alpha[..., 1]**2)
    plt.colorbar(im)
    plt.show()
