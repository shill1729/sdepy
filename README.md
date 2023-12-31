# sdepy
<!-- badges: start -->
<!-- badges: end -->
This is a package for aiding in studying SDEs.


## Installation

You can install the package via from github on windows/mac in command line with:

``` 
python -m pip install git+https://github.com/shill1729/sdepy.git
```

## Features
The package has functions/classes for

### Symbolic computations:
- Infinitesimal generators
- Adjoint generators
- SDE coefficients for Riemannian Brownian motion in local coordinates
- SDE coefficients for Riemannian Brownian motion in embedded in $\mathbb{R}^D$

### SDE solvers:
- Euler-Maruyama
- Milstein (TODO)
- Runge-Kutta (TODO)

### Parabolic PDE Solvers:
- Monte-Carlo parabolic PDE solvers (via Feynman-Kac formula)
- Implicit finite difference solvers for parabolic PDEs

### Hyperbolic PDE solvers:
- Implicit finite difference solver for the telegrapher type
- Monte-Carlo solvers for the telegrapher PDE



## Symbolic computation of local SDE coefficients of Riemannian Brownian motion
The user can supply coordinates $x,y$ a chart,
$$\phi(x,y) = (x_1(x,y), x_2(x,y), x_3(x,y))^T,$$
and the app will compute the metric tensor $g = D\phi^T D\phi$ and the coefficients for 
an SDE defining Brownian motion locally up to the first exit time of the chart:
$$dZ_t = \frac12 \nabla_g \cdot g^{-1}(Z_t) dt + \sqrt{g^{-1}(Z_t)}dB_t,$$
where $g^{-1}$ is the inverse metric tensor, the diffusion coefficient is its unique square root,
and $\nabla_g \cdot A$ is the manifold-divergence applied row-wise to the matrix $A$. The manifold
divergence of a vector field $f$ is $\nabla_g \cdot f = (1/\sqrt{\det g}) \nabla \cdot (\sqrt{\det g} f)$
where $\nabla \cdot h$ is the ordinary Euclidean divergence of the vector field $h$.
```python
from sdepy.symsde import *

# sympy input
t, x, y = sp.symbols("t x y", real=True)
xi = sp.Matrix([x, y])
coord = sp.Matrix([x, y])
g = metric_tensor(xi, coord)

# Path input
x0 = np.array([1.5, 3.14])
tn = 1
seed = 17
n = 9000

mu, sigma = local_bm_coefficients(g, coord)
f = sp.Function("f")(*coord)
inf_gen = infinitesimal_generator(f, coord, mu, sigma)
adj_gen = adjoint_generator(f, coord, mu, sigma)
print("Metric tensor")
print(g)
print("Local drift")
print(mu)
print("Local diffusion")
print(sigma)
print("Infinitesimal generator")
print(inf_gen)
print("Fokker Planck RHS")
print(adj_gen)

# Do any substitutions here
mu_np, sigma_np = sympy_to_numpy_coefficients(mu, sigma, coord)
xt = euler_maruyama(x0, tn, mu_np, sigma_np, n, seed=seed)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot3D(xt[:, 0], xt[:, 1], color="black")
plt.show()
```

## Symbolic computation of SDE coefficients of RBM embedded in Euclidean space

Many every day surfaces can be written as the zeros of some smooth function:
1. Sphere: $f(x,y,z)=x^2+y^2+z^2-1$
2. Ellipsoid $f(x,y,z)=(x/a)^2+(y/b)^2+(z/c)^2-1$
3. Paraboloid: $f(x,y,z)=(x/a)^2+(y/b)^2-z$
4. Hyperbolic Paraboloid: $f(x,y,z)=(y/b)^2-(x/a)^2-z$
5. Hyperboloid $f(x,y,z)=(x/a)^2+(y/b)^2-(z/c)^2-1$
6. Cylinder $f(x,y,z)=x^2+y^2-1$
7. Torus $f(x,y,z)=(\sqrt{x^2+y^2}-R)^2+z^2-r^2$

The orthogonal projection method to generate Brownian motion is as follows. Given a surface $M$ that can be written as
$$M = f^{-1}(\{0\})$$
for some smooth $f:\mathbb{R}^3\to \mathbb{R}$, we can define the normal vector
$$n(x)=\frac{\nabla f(x)}{\|\nabla f(x)\|},$$
and then the orthogonal projection to the tangent subspace at $x$ on $\Sigma$,
$$P(x)=I-n(x)n(x)^T,$$
where matrix multiplication is being used.

Then the Stratonovich SDE for BM on $\Sigma$ is
$$\partial X_t = P(X_t)\partial B_t$$
and the Ito SDE is
$$d X_t = c(X_t)n(X_t) dt + P(X_t) dB_t,$$
where $c=-\frac 12 \nabla \cdot n$ is the mean curvature of the surface in the direction of
the normal (chosen to face "inward").

More generally, for intersections of hypersurfaces we have
$$dX = N^T(c+q)dt + P(X)dB,$$
where the rows of $N$ are the $K=D-d$ normal vectors, $c$ is a vector of size $K$ whose
components are the mean curvatures in the direction of the normals, and $q$ is some
nasty term: $q^r = \mathop{\text{Tr}}(N D[n_r] N^T)$.


## Feynman-Kac formula
The function $u\in C^{1,2}([0, T]\times \mathbb{R}^n, \mathbb{R})$ solves
$$\frac{\partial u}{\partial t} + \mu(x)^T \nabla_x u(t, x)+ \frac12 \mathop{\text{Tr}}(\Sigma(x) \nabla_x^2 u(t,x))=0$$
with terminal condition $u(T, x) = h(x)$, if and only if
$$u(t,x) = \mathbb{E}(h(X_T) | X_t=x).$$
This can be extended to include "running-costs". Indeed, $u\in C^{1,2}([0, T]\times \mathbb{R}^d, \mathbb{R})$ solves
$$\frac{\partial u}{\partial t} + \mu(t, x)^T \nabla_x u(t,x)+\frac12 \mathop{\text{Tr}}(\Sigma(t,x) 
\nabla_x^2 u(t,x))+f(t,x)=0$$
with terminal condition $u(T,x)=h(x)$, if and only if
$$u(t,x) = \mathbb{E}\left[\int_t^T f(s, X_s)ds +h(X_T) | X_t=x\right]$$
where $\nabla_x$ is the gradient of a scalar function with repsect to $x$ and $\nabla_x^2$ is the Hessian 
of a scalar with respect to $x$. Here, $(X_t)_{t\geq 0}$ solves the SDE
$$dX_t = \mu(t, X_t) dt+\sigma(t, X_t)dB_t$$
where $B$ is a standard Brownian motion in $\mathbb{R}^m$, $\mu:[0,T]\times \mathbb{R}^d\to \mathbb{R}^d$ and 
$\sigma: [0, T]\times 
\mathbb{R}^d\to \mathbb{R}^{d\times m}$ and finally $\Sigma(t,x)= \sigma(t,x)\sigma(t,x)^T$ is the infinitesimal 
covariance.

For solutions to the SDE to exist, we only require locally Lipschitz continuity of $\mu$ and $\sigma$ in $x$. Here, 
the terminal cost $h$ and the running cost are allowed to be non-smooth. This is a highly efficient solver in large 
dimension $d\gg 1$ and small time-scales $0< T \ll 1$ because of its very nature--the solution at a single point 
$(t,x)$ is grid-free, and we can use a small amount of time-steps in the SDE solver when $[0,T]$ is 
small without losing accuracy.

### Example:
The below example computes the proportion of time that planar Brownian motion
spends in the unit-disk, i.e. we compute
$$u(t,x)=\mathbb{E}\left[\frac{1}{T}\int_t^T \mathbb{1}_{\|X_u\| \leq 1} du|X_t=x \right].$$

In terms of the above notation, the terminal cost is $h(x)=0$, and the running cost 
is $f(t, x)=\mathbb{1}_{ \|x\| \leq 1}/T$. 

```python
# A template for 2d Fenyman-Kac problems (solving PDEs with MC estimates of SDEs)
from sdepy.sdes import SDE
import numpy as np
tn = 0.1
ntime = 5
npaths = 50
noise_dim = None
x0 = np.array([1., 1.])
a = -1.5
b = 1.5
c = -1.5
d = 1.5
space_grid_size = 20
time_grid_size = 5
grid_bds = [a, b, c, d]
grid_sizes = [space_grid_size, time_grid_size]


def mu(t, x):
    return np.zeros(2)


def sigma(t, x):
    return np.eye(2)


def f(t, x):
    # return np.abs(x) < 1.
    return (np.linalg.norm(x, axis=1) < 1.)/tn


def h(x):
    return 0.


# For 2-d PDE estimation
sde = SDE(mu, sigma)
sde.feynman_kac_2d(f, h, x0, tn, grid_bds, grid_sizes, ntime, npaths, noise_dim)
```

![Feynman Kac 2D](Images/fk_ex_2d.png)







