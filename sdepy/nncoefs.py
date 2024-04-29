# An implementation of feed-forward neural networks whose inputs are ensembles of sample paths.
# This specific class is to be used for Neural SDEs that are trained by performing MLE for
# a random ensemble of discretized sample paths via some stochastic-integration scheme.
#
# For Euler-Maruyama, this is equivalent to a Gaussian Markov Process approximation to the SDE.
# For Milstein, it is a little more complicated but I have seen at least one paper where it was attempted. The
# complication arises from the square of a normal, hence Chi-Square distributions appear.
import torch.nn as nn
import torch


class EulerMaruyamaNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, num_layers, activation, final_act, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self._affine_list = nn.ModuleList()
        self.mse = nn.MSELoss()
        self.activation = getattr(nn, activation)() if isinstance(activation, str) else activation
        self.final_activation = getattr(nn, final_act)() if isinstance(final_act, str) else final_act
        self._initialize_nets()

    def _initialize_nets(self):
        self._affine_list.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(1, self.num_layers):
            self._affine_list.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
        self._affine_list.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.activation(self._affine_list[i](x))
        return self.final_activation(self._affine_list[-1](x))

    def jacobian_network(self, x):
        """
        Compute the Jacobian matrix of the network with respect to its inputs.
        This function will be adjusted to properly handle input dimensions and batching.
        """
        # Ensure x requires gradient to compute Jacobians
        x.requires_grad_(True)

        # Prepare a tensor to store Jacobians for each sample and each time step
        jacobian = torch.zeros(x.shape[0], x.shape[1], self.output_dim, self.input_dim, dtype=x.dtype, device=x.device)

        # Compute the network output
        y = self.forward(x)

        # Loop over each output dimension to compute partial derivatives with respect to every input dimension
        for i in range(self.output_dim):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, :, i] = 1  # Set to compute derivative w.r.t. the current output dimension

            # Compute gradients for the entire batch and time steps at once
            gradients, = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)

            # Assign gradients to the corresponding output dimension in the Jacobian tensor
            jacobian[:, :, i, :] = gradients

        return jacobian


if __name__ == "__main__":
    # Initialize the network.
    # Let's assume we want 3 layers with 4 neurons each and a simple structure.
    fn = EulerMaruyamaNeuralNetwork(3, 2, [4, 4, 4], 3, 'Tanh', 'Identity')

    # Create a sample input tensor of shape (N, n+1, D)
    # Let's assume N=2 (two batch samples), n+1=3 (three time steps), and D=3 (dimension of each sample)
    x = torch.tensor([[[2, 1, 3], [1, 1, 4], [0, 1, 2]],
                      [[1, 2, 3], [2, 0, 1], [3, 3, 3]]], dtype=torch.float32, requires_grad=True)

    # Display the input
    print("Input Tensor (N, n+1, D):")
    print(x)
    print(x.size())

    # Forward pass
    print("\nOutput from the network:")
    print(fn(x))
    print(fn(x).size())

    # Compute and print the Jacobian
    print("\nJacobian matrix of the network with respect to its inputs:")
    jacobian = fn.jacobian_network(x)
    print(jacobian)

    # Print the size of the Jacobian to confirm shape
    print("\nShape of the Jacobian matrix:")
    print(jacobian.size())
