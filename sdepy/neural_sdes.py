# A reimplementation of the Euler-Scheme Gaussian first-order Markov MLE approximation for
# learning SDEs using neural networks.
# The idea is simple: given an ensemble of sample paths, observed in discrete time with even time-steps,
# we maximize the log-likelihood of the corresponding Gaussian Markov process with drift and covariance
# parameterized by neural networks.
import torch

from nncoefs import EulerMaruyamaNeuralNetwork


def gaussian_nll(z, Sigma):
    """
    Gaussian negative log likelihood of a N(0, Sigma) random variable

    :param z: tensor of shape (N, n+1, d)
    :param Sigma: tensor of shape (N, n+1, d, d)
    :return: tensor of (N * (n+1), )
    """
    Sigma_inv = torch.linalg.inv(Sigma)
    quadratic_form = torch.einsum('Ntk, Ntkl, Ntl -> Nt', z, Sigma_inv, z)  # Efficient batched matrix-vector product
    log_det = torch.log(torch.linalg.det(Sigma))
    return (0.5 * quadratic_form + 0.5 * log_det).reshape(-1).sum()


class NeuralSDE(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, activation, noise_dim, *args, **kwargs):
        """
        Neural SDE

        :param state_dim: int, the dimension of the system
        :param hidden_dim: list of ints, Hidden dimension of the neural networks
        :param num_layers: int, Number of inner layers of the neural networks
        :param activation: torch.Activation or string, Activation function of the neural networks
        :param noise_dim: int, he dimension of the driving Brownian motion
        :param args: additional arguments for nn.Module
        :param kwargs: additional key word arguments for nn.Module
        """
        super().__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.drift_net = EulerMaruyamaNeuralNetwork(state_dim, state_dim, hidden_dim, num_layers, activation,
                                                    "Identity")
        self.diffusion_net = EulerMaruyamaNeuralNetwork(state_dim, state_dim * noise_dim, hidden_dim, num_layers,
                                                        activation, "Identity")
        self.covariance_net = EulerMaruyamaNeuralNetwork(state_dim, state_dim ** 2, hidden_dim, num_layers, activation,
                                                         "Identity")

    def diffusion(self, x):
        A = self.covariance_net(x).view(self.state_dim, self.state_dim)
        return torch.linalg.cholesky(A)

    def loss(self, x, h):
        """
        The loss function is the negative log-likelihood (up to some constants) of the Euler-Maruyama approximation.

        :param x: tensor of shape (N, n+1, d), ensemble of sample paths
        :param h: the time step
        :return: float
        """
        # Get the current values
        x1 = x[:, :-1, :]
        # The next values
        x2 = x[:, 1:, :]
        # The drifts
        drift = h * self.drift_net(x1)

        # Parameterizing the diffusion matrix instead of the covariance is more stable with Cholesky decomps
        # Sigma = h * self.covariance_net(x1).view(x.size(0), x.size(1)-1, self.state_dim, self.state_dim)

        # The diffusion and covariance
        sigma = self.diffusion_net(x1).view(x.size(0), x.size(1) - 1, self.state_dim, self.noise_dim)
        Sigma = h * torch.einsum('Ntik,Ntjk->Ntij', sigma, sigma)  # Efficient batched outer product
        z = x2 - x1 - drift
        loss = gaussian_nll(z, Sigma)
        return loss

    def fit(self, ensemble, lr, epochs, printfreq=1000, h=1 / 252, weight_decay=0.) -> None:
        """
        Train the autoencoder on a point-cloud.

        :param h:
        :param ensemble: the training data, expected to be of (N, n+1, D)
        :param lr, the learning rate
        :param epochs: the number of training epochs
        :param printfreq: print frequency of the training loss

        :return: None
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs + 1):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            total_loss = self.loss(ensemble, h)
            # Stepping through optimizer
            total_loss.backward()
            optimizer.step()
            if epoch % printfreq == 0:
                print('Epoch: {}: Train-Loss: {}'.format(epoch, total_loss.item()))

    def mu_fit(self, t, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.drift_net(x).detach().numpy()

    def sigma_fit(self, t, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.diffusion_net(x).view((self.state_dim, self.noise_dim)).detach().numpy()


if __name__ == "__main__":
    # An example of our Neural SDE being fit to planar motions.
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sdepy.sdes import SDE

    x00 = [0.5]
    x0 = torch.tensor(x00, dtype=torch.float32)
    x0np = np.array(x00)
    tn = 5.
    ntime = 10000
    ntrain = 5000
    npaths = 1  # number of sample paths in data ensemble
    npaths_fit = 5  # number of sample paths to generated under fitted model
    seed = 17
    lr = 0.001
    weight_decay = 0.  # Weight decay improves [32, 16] hidden dim fit by a lot!
    epochs = 10000
    hidden_dim = [1, 1]
    num_layers = 2
    noise_dim = 1
    act = "Tanh"
    printfreq = 1000
    state_dim = x0.size()[0]
    h = tn / ntime
    torch.manual_seed(seed)

    # Ground truth coefficients: One dimensional example: Mean-Reverting Process
    def mu(t, x):
        return 1.1 * (0.9 - x)


    def sigma(t, x):
        return 0.15 * np.sqrt(x)


    # Generating ensemble data
    sde = SDE(mu, sigma)
    ensemble = sde.sample_ensemble(x0np, tn, ntime, npaths, noise_dim=noise_dim)
    ensemble = torch.tensor(ensemble, dtype=torch.float32)
    training_ensemble = torch.zeros((npaths, ntrain, state_dim))
    test_ensemble = torch.zeros((npaths, ntime - ntrain + 1, state_dim))
    for j in range(npaths):
        training_ensemble[j, :, :] = ensemble[j, :ntrain, :]
        test_ensemble[j, :, :] = ensemble[j, ntrain:, :]

    # Neural SDE model to fit to ensemble data
    nsde = NeuralSDE(state_dim, hidden_dim, num_layers, act, noise_dim)
    nsde.fit(training_ensemble, lr, epochs, printfreq, h, weight_decay)
    print("NLL on Test Ensemble = " + str(nsde.loss(test_ensemble, h)))

    # Throw to an SDE object for convenient simulations of ensembles
    sde_fit = SDE(nsde.mu_fit, nsde.sigma_fit)
    ensemble_fit = sde_fit.sample_ensemble(x0np, tn, ntime, npaths=npaths_fit)
    ensemble = ensemble.detach().numpy()

    # Plot ensembles
    fig = plt.figure()
    t = np.linspace(0, tn, ntime + 1)

    for i in range(npaths):
        if state_dim == 2:
            plt.plot(ensemble[i, :, 0], ensemble[i, :, 1], c="black", alpha=0.5)
        elif state_dim == 1:
            plt.plot(t, ensemble[i, :, 0], c="black", alpha=0.5)
    for i in range(npaths_fit):
        if state_dim == 2:
            plt.plot(ensemble_fit[i, :, 0], ensemble_fit[i, :, 1], c="blue", alpha=0.5)
        elif state_dim == 1:
            plt.plot(t, ensemble_fit[i, :, 0], c="blue", alpha=0.5)

    # Creating custom legend entries
    true_line = plt.Line2D([], [], color='black', label='True')
    model_line = plt.Line2D([], [], color='blue', label='Model')
    # Adding the legend to the plot
    plt.legend(handles=[true_line, model_line])
    plt.show()
