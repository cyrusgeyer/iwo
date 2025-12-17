from torch import nn
import torch
import copy
import math


class IWOModel(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_dim,
        factor_sizes,
        factor_discrete,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = len(factor_sizes)
        models = []
        for i in range(self.num_factors):
            if factor_discrete[i]:
                out_size = factor_sizes[i]
            else:
                out_size = 1
            cfg = copy.deepcopy(model_cfg["default"])
            if model_cfg[f"factor_{i}"] is not None:
                cfg.update(model_cfg[f"factor_{i}"])

            model = ReductionNet(input_dim, out_size, **cfg)
            models.append(model)

        self.model = nn.ModuleList(models)
        self.per_factor_training = model_cfg["per_factor_training"]

    def forward(self, x):
        pred_factors = []
        for k, model in enumerate(self.model):
            if (
                str(self.per_factor_training) == str(k)
                or str(self.per_factor_training) == "all"
            ):
                pred_factors.append(model(x))
            else:
                pred_factors.append(None)
        return pred_factors

    def train_factor(self, k, train=True):
        self.per_factor_training[k] = train

    def get_w(self):
        w_list = []
        for k in range(self.num_factors):
            w_list.append(self.model[k].get_w())
        return w_list


class ReductionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        out_size,
        w_sizes=None,
        num_hidden_layers=2,
        hidden_dim=254,
        batch_norm=True,
        init_w="kaiming_normal_",
        init_layers="kaiming_normal_",
        nonlinearity="leaky_relu",
        nonlinearity_param=0.01,
        first_layer_batch_norm=False,
        first_layer_nonlinearity=False,
        **kwargs,
    ):
        super().__init__()
        self.out_size = out_size

        if isinstance(w_sizes, int):
            assert w_sizes < input_dim
            w_sizes = [
                i for i in reversed(range(1, w_sizes + 1))
            ]  # Reducing one dimension at a time.
        if isinstance(w_sizes, str):
            w_sizes = w_sizes.split(",")
            w_sizes = [int(size) for size in w_sizes]
            assert w_sizes[0] < input_dim
            if any(w_sizes[i] < w_sizes[i + 1] for i in range(len(w_sizes) - 1)):
                raise ValueError("w_sizes is not sorted from biggest to lowest")
        num_lin_layers = len(w_sizes)

        if isinstance(num_hidden_layers, int):
            num_hidden_layers = [num_hidden_layers] * (num_lin_layers + 1)
        if isinstance(num_hidden_layers, str):
            num_hidden_layers = num_hidden_layers.split(",")
            num_hidden_layers = [int(value) for value in num_hidden_layers]
            assert len(num_hidden_layers) == num_lin_layers + 1

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * (num_lin_layers + 1)
        if isinstance(hidden_dim, str):
            hidden_dim = hidden_dim.split(",")
            hidden_dim = [int(value) for value in hidden_dim]
            assert len(hidden_dim) == num_lin_layers + 1

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * (num_lin_layers + 1)
        if isinstance(batch_norm, str):
            batch_norm = batch_norm.split(",")
            batch_norm = [bool(int(value)) for value in batch_norm]
            assert len(batch_norm) == num_lin_layers + 1

        nets = [
            SimpleRegressionNet(
                input_dim,
                out_size,
                num_hidden_layers[0],
                hidden_dim[0],
                batch_norm[0],
                nonlinearity,
                nonlinearity_param,
                init_layers,
                False,
                False,
            )
        ]

        # Note that the first net acts directly on the hidden representation.
        # It therefore requires no w. That's why w_list[i] and
        # num_hidden_layers[i + 1] correspond to the same layer in the LNN.

        w_list = []

        # Subsequent IWO Layers
        column_dim = input_dim
        for i in range(num_lin_layers):
            row_dim = w_sizes[i]
            w = nn.Parameter(torch.eye(row_dim, column_dim))
            apply_init_w(w, init_w, nonlinearity, nonlinearity_param)
            w_list.append(w)
            nets.append(
                SimpleRegressionNet(
                    row_dim,
                    out_size,
                    num_hidden_layers[i + 1],
                    hidden_dim[i + 1],
                    batch_norm[i + 1],
                    nonlinearity,
                    nonlinearity_param,
                    init_layers,
                    first_layer_batch_norm,
                    first_layer_nonlinearity,
                )
            )
            column_dim = row_dim

        # Store the layers in a ParameterList and ModuleList so that they are registered as parameters and modules
        self.w_list_params = nn.ParameterList(w_list)
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        outs = []
        outs.append(self.nets[0](x))
        for i, w in enumerate(self.w_list_params):
            x = torch.einsum("nj,bj->bn", w, x)
            outs.append(self.nets[i + 1](x))
        return outs

    def get_w(self):
        w_list = []
        for w in self.w_list_params:
            w_list.append(w.detach().clone())
        return w_list


class SimpleRegressionNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_size,
        num_hidden_layers,
        hidden_dim,
        batch_norm,
        nonlinearity,
        nonlinearity_param,
        init_layers,
        first_layer_batch_norm,
        first_layer_nonlinearity,
    ):
        super().__init__()

        self.size = out_size

        layers = []
        if first_layer_nonlinearity:
            layers.append(make_nonlinearity(nonlinearity, nonlinearity_param))
        if first_layer_batch_norm:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(make_nonlinearity(nonlinearity, nonlinearity_param))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(1, num_hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            apply_init_w(layer.weight, init_layers, nonlinearity, nonlinearity_param)
            layers.append(layer)
            layers.append(make_nonlinearity(nonlinearity, nonlinearity_param))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_size))
        if out_size > 1:
            layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def make_nonlinearity(nonlinearity, nonlinearity_param):
    if nonlinearity == "leaky_relu":
        return nn.LeakyReLU(nonlinearity_param)
    elif nonlinearity == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError


def apply_init_w(w, init_w, nonlinearity, nonlinearity_param):
    if init_w == "kaiming_uniform_custom":
        gain = torch.nn.init.calculate_gain(nonlinearity, nonlinearity_param)
        fan = w.shape[0] * w.shape[1]
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return w.uniform_(-bound, bound)
    elif init_w == "kaiming_normal":
        torch.nn.init.kaiming_normal_(
            w, a=nonlinearity_param, nonlinearity=nonlinearity
        )
    elif init_w == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(
            w, a=nonlinearity_param, nonlinearity=nonlinearity
        )
    elif init_w == "xavier_normal":
        gain = torch.nn.init.calculate_gain(
            nonlinearity,
            param=nonlinearity_param if nonlinearity == "leaky_relu" else None,
        )
        torch.nn.init.xavier_normal_(w, gain)
    elif init_w == "xavier_uniform":
        gain = torch.nn.init.calculate_gain(
            nonlinearity,
            param=nonlinearity_param if nonlinearity == "leaky_relu" else None,
        )
        torch.nn.init.xavier_uniform_(w, gain)
    elif init_w == "eye":
        pass
    else:
        raise NotImplementedError
