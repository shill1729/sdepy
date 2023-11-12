# This module has sympy functions for dealing with SDEs (on manifolds).
import sympy as sp
import numpy as np
from sdepy.symcalc import matrix_divergence, hessian


def metric_tensor(chart, x):
    """
    Compute the metric tensor given a diffeomorphic chart from low to high dimension.

    :param chart:
    :param x:
    :return:
    """
    j = sp.simplify(chart.jacobian(x))
    g = j.T * j
    return sp.simplify(g)


def manifold_divergence(a: sp.Matrix, p: sp.Matrix, volume_measure):
    """
    Compute the manifold divergence of a matrix (row-wise) at a point given the volume measure
    :param a:
    :param p:
    :param detg:
    :return:
    """

    drift = sp.simplify(matrix_divergence(sp.simplify(volume_measure * a), p))
    drift = sp.simplify(drift / volume_measure)
    return drift


def local_bm_coefficients(g: sp.Matrix, p):
    """
    Compute the SDE coefficients of a Brownian motion in a local chart of a manifold

    :param g: metric tensor
    :param p: point
    :return:
    """
    # 1. Compute the diffusion coefficient
    ginv = sp.simplify(g.inv())
    diffusion = sp.simplify(ginv.cholesky(hermitian=False))
    # 2. Compute the drift
    detg = g.det()
    sqrt_detg = sp.sqrt(detg)
    manifold_div = manifold_divergence(ginv, p, sqrt_detg)
    drift = sp.simplify(manifold_div / 2)
    return drift, diffusion


def infinitesimal_generator(f, p: sp.Matrix, drift: sp.Matrix, diffusion: sp.Matrix):
    """
    Compute the infinitesimal generator of a SDE

    :param f: test function
    :param p: point to evaluate at
    :param drift: drift coefficient
    :param diffusion: diffusion coefficient

    :return:
    """
    h = sp.Matrix([f])
    hess_f, grad_f = hessian(h, p, return_grad=True)
    first_order_term = drift.T * grad_f
    first_order_term = sp.simplify(first_order_term)[0, 0]
    cov = sp.simplify(diffusion * diffusion.T)
    quadratic_variation_term = sp.simplify(cov * hess_f)
    second_order_term = quadratic_variation_term.trace() / 2
    inf_gen = first_order_term + second_order_term
    inf_gen = sp.simplify(inf_gen)
    return inf_gen


def adjoint_generator(f, p: sp.Matrix, drift: sp.Matrix, diffusion: sp.Matrix):
    """
    
    :param f:
    :param p:
    :param drift:
    :param diffusion:
    :return:
    """
    Sigma = sp.simplify(diffusion * diffusion.T)
    second_order_term = sp.simplify(matrix_divergence(Sigma*f, p)/2)
    flux = sp.simplify(-drift * f + second_order_term)
    adjoint = matrix_divergence(flux, p)
    return adjoint


def sympy_to_numpy_coefficients(mu, sigma, p):
    """
    Convert sympy SDE coefficients to numpy SDE coefficients as functions of p

    :param mu:
    :param sigma:
    :param p:
    :return:
    """
    d = mu.shape[0]
    return lambda x: sp.lambdify([p], mu)(x).reshape(d), sp.lambdify([p], sigma)


# Creating surfaces via the parameterizations
def surf_param(coord, chart, grid, aux=None, p=None):
    """ Compute a mesh of a surface via parameterization. The argument
    'grid' must be a tuple of arrays returned from 'np.mgrid' which the user
    must supply themselves, since boundaries and resolutions are use-case dependent.
    The tuple returned can be unpacked and passed to plot_surface

    (Parameters):
    coord: sympy object defining parameters
    chart: sympy object defining the coordinate transformation
    grid: tuple of the arrays returned from np.mgrid[...]
    aux: sympy Matrix for auxiliary parameters in the metric tensor
    p: numpy array for the numerical values of any auxiliary parameters in the equations

    returns tuple (x,y,z), (x,y) or (x)
    """
    d = len(grid)
    m = grid[0].shape[0]
    N = chart.shape[0]
    if aux is None:
        chart_np = sp.lambdify([coord], chart)
    else:
        chart_np = sp.lambdify([coord, aux], chart)

    xx = np.zeros((N, grid[0].shape[0], grid[0].shape[1]))
    for i in range(m):
        for j in range(m):
            w = np.zeros(d)
            for l in range(d):
                w[l] = grid[l][i, j]
            for l in range(N):
                if aux is None:
                    xx[l][i, j] = chart_np(w)[l, 0]
                else:
                    xx[l][i, j] = chart_np(w, p)[l, 0]
    if N == 3:
        x = xx[0]
        y = xx[1]
        z = xx[2]
        return x, y, z
    elif N == 2:
        x = xx[0]
        y = xx[1]
        return x, y
    else:
        return xx


def lift_path(xt, f, m=3):
    """

    :param xt: local path
    :param f: diffeomorphism lifting to the manifold
    :param m: higher dimension
    :return:
    """
    ntime = xt.shape[0] - 1
    # We need to lift the motion back to the ambient space
    yt = np.zeros((ntime + 1, m))
    for i in range(ntime + 1):
        yt[i, :] = f(xt[i, :]).reshape(m)
    return yt


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from solvers import euler_maruyama
    # sympy input
    theta, phi = sp.symbols("theta phi", real=True, positive=True)
    x = sp.sin(theta) * sp.cos(phi)
    y = sp.sin(theta) * sp.sin(phi)
    z = sp.cos(theta)
    xi = sp.Matrix([x, y, z])
    coord = sp.Matrix([theta, phi])
    g = metric_tensor(xi, coord)

    # Path input
    x0 = np.array([1.5, 3.14])
    tn = 1
    seed = 17
    n = 9000

    # TODO: make everything below a function: and with multiple paths?
    mu, sigma = local_bm_coefficients(g, coord)
    f = sp.Function("f")(*coord)
    inf_gen = infinitesimal_generator(f, coord, mu, sigma)
    adj_gen = adjoint_generator(f, coord, mu, sigma)
    mu_np, sigma_np = sympy_to_numpy_coefficients(mu, sigma, coord)
    harmonic_test = infinitesimal_generator(sp.sin(theta), coord, mu, sigma)
    fokker_planck_test = adjoint_generator(sp.sin(theta), coord, mu, sigma)
    print("Metric tensor")
    print(g)
    print("Local drift")
    print(mu)
    print("Local diffusion")
    print(sigma)
    print("Infinitesimal generator")
    print(inf_gen)
    print("Harmonic test")
    print(harmonic_test)
    print("Fokker Planck RHS")
    print(adj_gen)
    print("Fokker Planck Test")
    print(fokker_planck_test)

    xt = euler_maruyama(x0, tn, mu_np, sigma_np, n, seed=seed)
    yt = lift_path(xt, sp.lambdify([coord], xi))
    # Surface grid
    grid1 = np.linspace(0, np.pi, 100)
    grid2 = np.linspace(0, 2 * np.pi, 100)
    grid = np.meshgrid(grid1, grid2, indexing="ij")
    x1, x2, x3 = surf_param(coord, xi, grid)

    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], color="black")
    ax.plot_surface(x1, x2, x3, cmap="viridis", alpha=0.5)
    plt.show()
