import numpy as np
"""
beta: Source plane angular coordinate
theta: Image plane coordinate
kappa: Projected surface mass density, in units of the critical density Sigma
"""


def sis_profile(theta, x0=5, y0=5):
    """
    Singular Isothermal Sphere kappa = 1/2x, but in vector form here
    """
    xi = np.stack([np.abs(theta[..., 0] - x0), np.abs(theta[..., 1] - y0)], axis=-1)
    return  xi / 2 / (xi[..., 0]**2 + xi[..., 1]**2 + 1e-6)[..., np.newaxis]


def point_mass(theta, x0=7, y0=0):
    out = np.zeros_like(theta)
    x_index = np.unravel_index(np.argmin(np.abs(theta[..., 0] - x0)), theta[..., 0].shape)[1]
    y_index = np.unravel_index(np.argmin(np.abs(theta[..., 1] - y0)), theta[..., 1].shape)[0]
    out[y_index, x_index, :] += 1
    return out


def coordinates(N: int):
    """
    Generate the image plane coordinates
    """
    x = np.arange(0, N, dtype=np.float32) - N // 2 - (N%2 - 1)/2
    xx, yy = np.meshgrid(x, x)
    theta = np.stack([xx, yy], axis=-1)
    return theta


def alpha_integral(theta, kappa):
    #TODO make this memory efficient
    N = theta.shape[0] # number of pixels
    assert N <= 32, "Shape is too large, N <= 32"
    theta_prime = np.stack([theta] * N, axis=-1)
    theta_prime = np.stack([theta_prime] * N, axis=-1) # add 2 dimensions
    kappa = np.stack([kappa] * N, axis=-1)
    kappa = np.stack([kappa] * N, axis=-1)
    for i in range(0, N):
        for j in range(0, N):
            theta_prime[..., i, j] = np.roll(theta_prime[..., i, j], j, axis=1)
            theta_prime[..., i, j] = np.roll(theta_prime[..., i, j], i, axis=0)
            kappa[..., i, j] = np.roll(kappa[..., i, j], j, axis=1)
            kappa[..., i, j] = np.roll(kappa[..., i, j], i, axis=0)

    theta = theta[..., np.newaxis, np.newaxis] # prepare for broadcast
    norm = ((theta[..., 0, :, :] - theta_prime[..., 0, :, :])**2 + \
                    (theta[..., 1, :, :] - theta_prime[..., 1, :, :])**2)[..., np.newaxis, :, :]
    alpha = np.sum(kappa * (theta - theta_prime) / (norm + 1e-6), axis=(3, 4)) # return a 2d tensor
    return alpha


def lens_equation(theta, alpha):
    # Returns beta
    return theta - alpha


if __name__ == "__main__":
    from visualization import lens_and_source
    import matplotlib.pyplot as plt
    N = 32  # 32 is about the maximum the double integral can handle
    theta = coordinates(N)
    kappa = sis_profile(theta)
    # kappa = point_mass(theta)
    alpha = alpha_integral(theta, kappa)
    beta = lens_equation(theta, alpha)
    plt.plot(beta[..., 0], beta[..., 1], "k.")
    # im = plt.imshow(kappa[..., 1]**2 + kappa[..., 0]**2)
    # plt.colorbar(im)
    # plt.show()
    # alpha = alpha[..., 1]**2 + alpha[..., 0]**2 # scalar function of position
    # im = plt.imshow(alpha)
    # plt.colorbar(im)
    # plt.show()
    # im = plt.imshow(beta[..., 1]**2 + beta[..., 0]**2)
    # plt.colorbar(im)
    plt.show()
