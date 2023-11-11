# sdepy
This is a package for aiding in studying SDEs.

The user can supply coordinates $x,y$ a chart,
$$\phi(x,y) = (x_1(x,y), x_2(x,y), x_3(x,y))^T,$$
and the app will compute the metric tensor $g = D\phi^T D\phi$ and the coefficients for 
an SDE defining Brownian motion locally up to the first exit time of the chart:
$$dZ_t = \frac12 \nabla_g \cdot g^{-1}(Z_t) dt + \sqrt{g^{-1}(Z_t)}dB_t,$$
where $g^{-1}$ is the inverse metric tensor, the diffusion coefficient is its unique square root,
and $\nabla_g \cdot A$ is the manifold-divergence applied row-wise to the matrix $A$. The manifold
divergence of a vector field $f$ is $\nabla_g \cdot f = (1/\sqrt{\det g}) \nabla \cdot (\sqrt{\det g} f)$
where $\nabla \cdot h$ is the ordinary Euclidean divergence of the vector field $h$.





