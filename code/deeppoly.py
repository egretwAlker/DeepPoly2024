import torch
from skip_block import SkipBlock
import torch.nn.functional as F
from typing import Optional


DEBUG = False
EPS = 1e-6
INIT_COEF = 4


def eliding_w(input_dim: int, target: int) -> torch.Tensor:
    """
    Generate a weight matrix for last layer eliding.

    Args:
        input_dim (int): The dimensionality of the input.
        target (int): The target index to elide weights.

    Returns:
        w (torch.Tensor): The weight matrix for eliding weights.
    """
    w = torch.zeros(input_dim-1, input_dim)
    w[:target, :target] = torch.eye(target)
    w[target:, target + 1:] = torch.eye(input_dim - target - 1)
    w[:, target] = -1
    return w


def skip_application(skip_dim: int) -> torch.Tensor:
    """
    Generate a Linear weight matrix for skip connections.

    Args:
        skip_dim (int): The dimensionality of the skip connections.
    
    Returns:
        w (torch.Tensor): The Linear weight matrix for skip connections.
    """
    w = torch.zeros(skip_dim, 2*skip_dim)
    w[:, :skip_dim] = w[:, skip_dim:skip_dim*2] = torch.eye(skip_dim)
    return w


def dp_verify(model: torch.nn.Module, input_lower_bound: torch.Tensor, input_upper_bound: torch.Tensor, true_label: int, n_labels: int) -> bool:
    """
    The main pipeline for DeepPoly verification.

    Args:
        model (torch.nn.Module): The model to be verified.
        input_lower_bound (torch.Tensor): The lower bound of the input.
        input_upper_bound (torch.Tensor): The upper bound of the input.
        true_label (int): The expected output of the model.
        n_labels (int): The number of labels in the model.
    
    Returns:
        result (bool): True if the model satisfies DeepPoly verification, False otherwise.
    """
    inshape = input_lower_bound.shape
    input_lower_bound = input_lower_bound.unsqueeze(0)
    input_upper_bound = input_upper_bound.unsqueeze(0)
    shallowuni = Deeppoly(input_lower_bound, input_upper_bound)
    shallowuni.forward(model)
    if shallowuni.append_and_verify(true_label):
        return True
    if shallowuni.alpha_buffer_idx == 0:
        return False

    if DEBUG:
        print("Start learning slopes")
        torch.autograd.set_detect_anomaly(True)

    lowest_upper_bounds = shallowuni.upper_bounds[-1].min(0).values
    optimizer = torch.optim.Adagrad(shallowuni.alpha_buffer, lr=1)
    for epoch in range(2000):
        optimizer.zero_grad()
        shallowuni.rewind()
        shallowuni.forward(model)
        if shallowuni.append_and_verify(true_label):
            return True
        lowest_upper_bounds = torch.min(lowest_upper_bounds, shallowuni.upper_bounds[-1].min(0).values)
        if (lowest_upper_bounds <= -EPS).all():
            return True
        loss = shallowuni.upper_bounds[-1][:, lowest_upper_bounds >= -EPS].flatten().sum()
        loss.backward()
        optimizer.step()
        # print(loss, lowest_upper_bounds)
    return False
    


def backsubstitute(W_l0: torch.Tensor, W_u0: torch.Tensor, W_l: torch.Tensor, W_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the linear constraints according to the given two linear constraints.

    Args:
        W_l0 (torch.Tensor): The former lower bound for the linear constraints.
        W_u0 (torch.Tensor): The former upper bound for the linear constraints.
        W_l (torch.Tensor): The current lower bound for the linear constraints.
        W_u (torch.Tensor): The current upper bound for the linear constraints.
    
    Returns:
        W_l_tilde (torch.Tensor): The updated lower bound for the linear constraints.
        W_u_tilde (torch.Tensor): The updated upper bound for the linear constraints.
    """
    if DEBUG:
        assert W_l0[:, -1, -1] == W_u0[:, -1, -1] == 1
        assert W_l[:, -1, -1] == W_u[:, -1, -1] == 1

    W_l_pos = W_l.clamp_min(0)
    W_l_neg = W_l.clamp_max(0)
    W_u_pos = W_u.clamp_min(0)
    W_u_neg = W_u.clamp_max(0)

    W_l_tilde = W_l0 @ W_l_pos + W_u0 @ W_l_neg
    W_u_tilde = W_u0 @ W_u_pos + W_l0 @ W_u_neg

    return W_l_tilde, W_u_tilde


class Deeppoly():
    def __init__(self, init_lower_bound: torch.Tensor, init_upper_bound: torch.Tensor):
        """
        Initialize the DeepPoly object with the initial numerical boundaries.

        Args:
            init_lowe_bound (torch.Tensor): The initial lower bound for the numerical boundaries.
            init_upper_bound (torch.Tensor): The initial upper bound for the numerical boundaries.
        
        Returns:
            None
        """

        # numerical boundaries
        self.lower_bounds: list[torch.Tensor] = [init_lower_bound]
        self.upper_bounds: list[torch.Tensor] = [init_upper_bound]
        # the lock mechanism will be buggy if there's skip in skip, which is not the case for our nets
        # the lock mechanism is to ensure that the linear layer absorbed does not cause problem for skip
        self.lock_idx : Optional[int] = None
        # linear constraint expressions can be stored as matrices multipliers (x' \leq Ax + b)
        # x' <> x (A| 0)
        #         (b| 1)
        self.lower_const_weights: list[torch.Tensor] = []
        self.upper_const_weights: list[torch.Tensor] = []

        self.alpha_buffer: list[torch.nn.Parameter] = []
        self.alpha_buffer_idx: int = 0

        self.conv_buffer: list[torch.Tensor] = []
        self.conv_buffer_idx: int = 0

        self.forward_net = {
            torch.nn.Sequential: self._sequential,
            SkipBlock: self._skip,
            torch.nn.ReLU6: self._relu6,
        }

        self.forward_layer = {
            torch.nn.Conv2d: self._conv2d,
            torch.nn.Linear: self._linear,
            torch.nn.ReLU: self._relu,
            torch.nn.Flatten: self._flatten,
        }


    def rewind(self):
        """
        Clear the cache except the alpha buffer.

        Args:
            None
        
        Returns:
            None
        """
        self.lower_bounds = self.lower_bounds[:1]
        self.upper_bounds = self.upper_bounds[:1]
        self.lower_const_weights = []
        self.upper_const_weights = []
        self.lock_idx = None
        self.alpha_buffer_idx = 0
        self.conv_buffer_idx = 0


    def lock(self):
        """
        Set the idx of the original tensor of the skip net.

        Args:
            None
        
        Returns:
            None
        """

        self.lock_idx = len(self.upper_bounds) - 1


    def unlock(self):
        self.lock_idx = None


    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add the last constant dimension for the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            x (torch.Tensor): The input tensor with the last constant dimension.
        """
        return torch.cat((x.flatten(start_dim=1), torch.ones(x.shape[0], 1)), dim=1).unsqueeze(1)


    def _get_output(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Delete the last constant dimension for the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            shape (tuple[int,...]): The shape of the input tensor.
        
        Returns
            x (torch.Tensor): The input tensor without the last constant dimension.
        """
        return x[:, :, :-1].view(shape)


    def _get_bounds(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor, W_l: torch.Tensor, W_u: torch.Tensor, outshape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the new bounds

        Args:
            lower_bound (torch.Tensor): The lower bound for the numerical boundaries.
            upper_bound (torch.Tensor): The upper bound for the numerical boundaries.
            W_l (torch.Tensor): The constraint weights for the lower bound.
            W_u (torch.Tensor): The constraint weights for the upper bound.
        """
        lower_bound = self._prepare_input(lower_bound)
        upper_bound = self._prepare_input(upper_bound)
        W_l_pos = W_l.clamp_min(0)
        W_l_neg = W_l.clamp_max(0)
        W_u_pos = W_u.clamp_min(0)
        W_u_neg = W_u.clamp_max(0)
        return self._get_output(lower_bound @ W_l_pos + upper_bound @ W_l_neg, outshape), \
                self._get_output(upper_bound @ W_u_pos + lower_bound @ W_u_neg, outshape)


    def forward(self, model: torch.nn.Module) -> None:
        """
        The forward process of DeepPoly. Get the box bounds and related constraints weights.

        Args:
            model (torch.nn.Module): The model to be processed.

        Returns:
            None
        
        Raises:
            NotImplementedError: If the model type is not supported.
        """
        for param in model.parameters():
            param.requires_grad = False
        if type(model) in self.forward_net:
            self.forward_net[type(model)](model)
        else:
            raise NotImplementedError(f"{type(model)} is not supported")


    def append_constraints(self, bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Add the new lower bound, upper bound, lower constraint weights, and upper constraint weights to the attributes.

        Args:
            bounds (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The lower bound, upper bound, lower constraint weights, and upper constraint weights.

        Returns:
            None
        """
        l_bound, u_bound, l_const_weight, u_const_weight = bounds
        if DEBUG:
            bounds = self._get_bounds(self.lower_bounds[-1], self.upper_bounds[-1], l_const_weight, u_const_weight, l_bound.shape)
            assert (bounds[0] > l_bound + 1e-4).sum() == 0, (bounds[0] - l_bound).max()
            assert (bounds[1] < u_bound - 1e-4).sum() == 0, (bounds[1] - u_bound).min()
        if DEBUG:
            assert (l_bound > u_bound).sum() == 0
        idx = len(self.upper_bounds)-1
        if idx > 0 and self.lock_idx != idx and torch.allclose(self.lower_const_weights[-1], self.upper_const_weights[-1]):
            W_l, W_u = backsubstitute(self.lower_const_weights[-1], self.upper_const_weights[-1], l_const_weight, u_const_weight)
            new_bounds = self._get_bounds(self.lower_bounds[-2], self.upper_bounds[-2], W_l, W_u, l_bound.shape)
            self.lower_bounds[-1] = torch.max(new_bounds[0], l_bound)
            self.upper_bounds[-1] = torch.min(new_bounds[1], u_bound)
            self.lower_const_weights[-1] = W_l
            self.upper_const_weights[-1] = W_u
        else:
            self.lower_bounds.append(l_bound)
            self.upper_bounds.append(u_bound)
            self.lower_const_weights.append(l_const_weight)
            self.upper_const_weights.append(u_const_weight)
        self._upd_from_back()


    def append_and_verify(self, true_output: int) -> bool:
        """
        Append Elide layer and verify the output.

        Args:
            true_output (int): The true 
            
        Returns:
            result (bool): True if the output is verified, False otherwise
        """
        last_layer = eliding_w(self.upper_bounds[-1].shape[1], true_output)
        self.append_constraints(self._affine(self.lower_bounds[-1], self.upper_bounds[-1], last_layer))
        return (self.upper_bounds[-1] <= -EPS).all(-1).any()


    def _upd_from_back(self):
        """
        Back Substitution for the lower and upper bounds.

        Args:
            None

        Returns:
            None
        """
        W_l = self.lower_const_weights[-1]
        W_u = self.upper_const_weights[-1]
        out_shape = self.lower_bounds[-1].shape
        for i in range(len(self.upper_bounds)-3, -1, -1):
            W_l, W_u = backsubstitute(self.lower_const_weights[i], self.upper_const_weights[i], W_l, W_u)
            if DEBUG:
                assert (self.lower_bounds[-1] > self.upper_bounds[-1]).sum() == 0, (self.lower_bounds[-1] - self.upper_bounds[-1]).max()
            new_bounds = self._get_bounds(self.lower_bounds[i], self.upper_bounds[i], W_l, W_u, out_shape)
            self.lower_bounds[-1] = torch.max(new_bounds[0], self.lower_bounds[-1])
            self.upper_bounds[-1] = torch.min(new_bounds[1], self.upper_bounds[-1])
            if DEBUG:
                assert (self.lower_bounds[-1] > self.upper_bounds[-1]).sum() == 0, (self.lower_bounds[-1] - self.upper_bounds[-1]).max()


    def _sequential(self, net: torch.nn.Sequential) -> None:
        """
        Forward process for sequential layers.

        Args:
            net (torch.nn.Sequential): Sequential layers

        Returns:
            None
        """
        for i, module in enumerate(net.children()):
            if type(module) in self.forward_layer:
                self.append_constraints(self.forward_layer[type(module)](self.lower_bounds[-1], self.upper_bounds[-1], module))
                if DEBUG:
                    assert (self.lower_bounds[-1] > self.upper_bounds[-1]).sum() == 0
            else:
                self.forward_net[type(module)](module)


    def _relu(self, l_bound: torch.Tensor, u_bound: torch.Tensor, layer: torch.nn.ReLU) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process for ReLU layers.

        Args:
            l_bound (torch.Tensor): Lower bounds
            u_bound (torch.Tensor): Upper bounds
            layer (torch.nn.ReLU): ReLU layer
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Lower and upper bounds, and weights for lower and upper bounds
        """
        n_batch = u_bound.shape[0]
        l_flatten = l_bound.flatten(start_dim=1)
        u_flatten = u_bound.flatten(start_dim=1)
        l_new = l_bound.clamp(min=0)
        u_new = u_bound.clamp(min=0)

        above_diag = torch.ones_like(u_flatten) * (l_flatten >= 0)
        above_diag = torch.concat([above_diag, torch.zeros(n_batch, 1)], dim=1)
        u_above_weight = torch.diag_embed(above_diag)
        l_above_weight = torch.diag_embed(above_diag)

        crossing = (u_flatten > 0) & (l_flatten < 0)

        upper_slope = torch.zeros_like(u_flatten)
        upper_slope[crossing] = u_flatten[crossing] / (u_flatten[crossing] - l_flatten[crossing])

        upper_slope = torch.concat([upper_slope, torch.zeros(n_batch, 1)], dim=1)
        u_cross_weight = torch.diag_embed(upper_slope)
        u_cross_weight[:, -1, :-1] = -l_flatten * upper_slope[:, :-1]

        l_cross_weight = torch.zeros_like(l_above_weight)
        if self.alpha_buffer_idx == len(self.alpha_buffer):
            lower_slope = torch.nn.Parameter(INIT_COEF * torch.where(u_flatten.abs() > l_flatten.abs(), 1.0, -1.0))
            self.alpha_buffer.append(lower_slope)
        else:
            lower_slope = self.alpha_buffer[self.alpha_buffer_idx]

        l_cross_weight[:, :-1, :-1] = torch.diag_embed(torch.where(crossing, torch.sigmoid(lower_slope), 0.0))
        self.alpha_buffer_idx += 1

        l_weight = l_above_weight + l_cross_weight
        u_weight = u_above_weight + u_cross_weight
        l_weight[:, -1, -1] = u_weight[:, -1, -1] = 1
        if DEBUG:
            bounds = self._get_bounds(l_bound, u_bound, l_weight, u_weight, l_new.shape)
            return *bounds, l_weight, u_weight
        return l_new, u_new, l_weight, u_weight


    def _relu6(self, module: torch.nn.ReLU6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process for ReLU6 Layer.

        Args:
            module (torch.nn.ReLU6): The ReLU6 layer.
        
        Returns:
            None
        """
        def six_minus(t: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            l_bound, u_bound = 6 - t[1], 6 - t[0]
            l_const_weight = torch.zeros_like(t[2])
            l_const_weight[:, :-1, :-1] = -t[3][:, :-1, :-1]
            l_const_weight[:, -1, :-1] = 6-t[3][:, -1, :-1]
            l_const_weight[:, -1, -1] = 1
            u_const_weight = torch.zeros_like(t[3])
            u_const_weight[:, :-1, :-1] = -t[2][:, :-1, :-1]
            u_const_weight[:, -1, :-1] = 6-t[2][:, -1, :-1]
            u_const_weight[:, -1, -1] = 1
            return l_bound, u_bound, l_const_weight, u_const_weight

        self.append_constraints(six_minus(self._relu(self.lower_bounds[-1], self.upper_bounds[-1], None)))
        self.append_constraints(six_minus(self._relu(self.lower_bounds[-1], self.upper_bounds[-1], None)))


    def _conv2d(
        self, l_bound: torch.Tensor, u_bound: torch.Tensor, layer: torch.nn.Conv2d
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process for Conv2d Layer.

        Args:
            l_bound (torch.Tensor): Lower bounds
            u_bound (torch.Tensor): Upper bounds
            layer (torch.nn.Conv2d): The Conv2d layer
        
        Returns:
            l_new (torch.Tensor): New lower bounds
            u_new (torch.Tensor): New upper bounds
            l_weight (torch.Tensor): Lower bounds constraint weights
            u_weight (torch.Tensor): Upper bounds constraint weights
        """
        if DEBUG:
            assert (l_bound > u_bound).sum() == 0
        # shadow the setting and weight of conv2d layer
        w_kernel = layer.weight
        b = layer.bias
        stride = layer.stride
        padding = layer.padding
        batch_size, in_channel, in_height, in_width = u_bound.shape

        # Masking positive and negative weights separately
        w_kernel_postive = w_kernel.clamp(min=0)
        w_kernel_negative = w_kernel.clamp(max=0)
        if DEBUG:
            assert torch.allclose(
                F.conv2d(l_bound, w_kernel, b, stride, padding),
                layer(l_bound),
            ), (F.conv2d(l_bound, w_kernel, b, stride, padding) - layer(l_bound)).abs().max()

        l_new = F.conv2d(
            l_bound, w_kernel_postive, b, stride, padding
        ) + F.conv2d(u_bound, w_kernel_negative, None, stride, padding)
        u_new = F.conv2d(
            u_bound, w_kernel_postive, b, stride, padding
        ) + F.conv2d(l_bound, w_kernel_negative, None, stride, padding)
        if DEBUG:
            assert (l_new > u_new).sum() == 0

        _, out_channel, out_height, out_width = u_new.shape

        # calculate the linear constraint
        if self.conv_buffer_idx == len(self.conv_buffer):
            conv_mat = torch.zeros(
                in_channel * in_height * in_width + 1,
                out_channel * out_height * out_width + 1,
            )
            input = torch.eye(in_channel * in_height * in_width).reshape(-1, in_channel, in_height, in_width)
            conv_mat[:-1, :-1] = layer(input).flatten(start_dim=1)
            conv_mat[-1, :-1] = b.repeat(out_height * out_width, 1).T.flatten()
            conv_mat[:-1, :-1] -= conv_mat[-1, None, :-1]
            conv_mat[-1, -1] = 1
            self.conv_buffer.append(conv_mat)
        else:
            conv_mat = self.conv_buffer[self.conv_buffer_idx]
        self.conv_buffer_idx += 1
        conv_mat = conv_mat.repeat(batch_size, 1, 1)
        if DEBUG:
            input = u_bound
            output = layer(l_bound)
            output_prime = self._get_output(self._prepare_input(l_bound) @ conv_mat, output.shape)
            assert torch.allclose(output, output_prime, atol=1), (output - output_prime).abs().max()
            l_new_prime = self._get_output(self._prepare_input(l_bound) @ conv_mat.clamp_min(0) + self._prepare_input(u_bound) @ conv_mat.clamp_max(0), l_new.shape)
            u_new_prime = self._get_output(self._prepare_input(u_bound) @ conv_mat.clamp_min(0) + self._prepare_input(l_bound) @ conv_mat.clamp_max(0), u_new.shape)
            assert torch.allclose(l_new, l_new_prime, atol=1), (l_new - l_new_prime).abs().max()
            assert torch.allclose(u_new, u_new_prime, atol=1), (u_new - u_new_prime).abs().max()
        return l_new, u_new, conv_mat, conv_mat


    def _flatten(self, l_bound: torch.Tensor, u_bound: torch.Tensor, layer: torch.nn.Flatten) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process for Flatten Layer.

        Args:
            l_bound: lower bound tensor
            u_bound: upper bound tensor
            layer: Flatten layer
        
        Returns:
            l_new: lower bound tensor after applying the layer
            u_new: upper bound tensor after applying the layer
            l_const_weight: lower bound constraint weights
            u_const_weight: upper bound constraint weights
        """
        n_batch = u_bound.shape[0]
        l_new = l_bound.flatten(start_dim=1)
        u_new = u_bound.flatten(start_dim=1)
        
        l_const_weight = torch.eye(l_new.shape[1]+1).repeat(n_batch, 1, 1)
        u_const_weight = torch.eye(u_new.shape[1]+1).repeat(n_batch, 1, 1)

        return l_new, u_new, l_const_weight, u_const_weight

 
    def _skip(self, net: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process for Skip net.

        Args:
            net (torch.nn.Module): Skip net module
        
        Returns
            None
        """
        self.lock()
        upd_from = len(self.upper_bounds) - 1
        skipdim = self.upper_bounds[-1].shape[1]
        for i, module in enumerate(net.path.children()):
            if type(module) in self.forward_layer:
                self.append_constraints(self.forward_layer[type(module)](self.lower_bounds[-1], self.upper_bounds[-1], module))
            else:
                self.forward_net[type(module)](module)

        def carry(bound0: torch.Tensor, bound1: torch.Tensor, const_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            load initial matrix for bound1 and update the weights.

            Args:
                bound0 (torch.Tensor): lower bound tensor
                bound1 (torch.Tensor): upper bound tensor
                const_weights (torch.Tensor): constraint weights
            
            Returns:
                new_bound (torch.Tensor): updated lower bound tensor
                new_weights (torch.Tensor): updated constraint weights
            """
            new_bound = torch.cat([bound1, bound0[:, -skipdim:]], dim=1)
            new_weights = torch.zeros(const_weights.shape[0], bound0.shape[1]+1, new_bound.shape[1]+1)
            new_weights[:, :const_weights.shape[1]-1, :const_weights.shape[2]-1] = const_weights[:, :-1, :-1]
            new_weights[:, -1, :-skipdim-1] = const_weights[:, -1, :-1]
            new_weights[:, -skipdim-1:, -skipdim-1:] += torch.eye(skipdim+1)
            return new_bound, new_weights
        
        for i in range(upd_from + 1, len(self.upper_bounds)):
            self.lower_bounds[i], self.lower_const_weights[i-1] = carry(self.lower_bounds[i-1], self.lower_bounds[i], self.lower_const_weights[i-1])
            self.upper_bounds[i], self.upper_const_weights[i-1] = carry(self.upper_bounds[i-1], self.upper_bounds[i], self.upper_const_weights[i-1])
        self.unlock()
        self.append_constraints(self._affine(self.lower_bounds[-1], self.upper_bounds[-1], skip_application(skipdim)))

 
    def _linear(self, l_bound: torch.Tensor, u_bound: torch.Tensor, layer: torch.nn.Linear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrapped forward process for Linear Layer.

        Args:
            l_bound (torch.Tensor): lower bound tensor
            u_bound (torch.Tensor): upper bound tensor
            layer (torch.nn.Linear): Linear layer
        
        Returns:
            l_new (torch.Tensor): lower bound tensor after applying the layer
            u_new (torch.Tensor): upper bound tensor after applying the layer
            l_const_weight (torch.Tensor): lower bound constraint weights
            u_const_weight (torch.Tensor): upper bound constraint weights
        """
        return self._affine(l_bound, u_bound, layer.weight, layer.bias)
    

    def _affine(self, l_bound: torch.Tensor, u_bound: torch.Tensor, w: torch.nn.Module, b: Optional[torch.nn.Module] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detailed forward process for Linear Layer.

        Args:
            l_bound (torch.Tensor): lower bound tensor
            u_bound (torch.Tensor): upper bound tensor
            layer (torch.nn.Linear): Linear layer
        
        Returns:
            l_new (torch.Tensor): lower bound tensor after applying the layer
            u_new (torch.Tensor): upper bound tensor after applying the layer
            l_const_weight (torch.Tensor): lower bound constraint weights
            u_const_weight (torch.Tensor): upper bound constraint weights
        """
        n_batch = u_bound.shape[0]
        w = w.T

        w_postive = w.clamp(min=0)
        w_negative = w.clamp(max=0)

        l_new = l_bound @ w_postive + u_bound @ w_negative
        u_new = u_bound @ w_postive + l_bound @ w_negative
        const_weight = torch.zeros(n_batch, w.shape[0]+1, w.shape[1]+1)
        const_weight[:, :-1, :-1] = w
        const_weight[:, -1, -1] = 1

        if b is not None:
            const_weight[:, -1, :-1] = b
            l_new += b
            u_new += b

        return l_new, u_new, const_weight, const_weight