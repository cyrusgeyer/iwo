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
            if self.per_factor_training[k]:
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
        first_dim=None,
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

        # First IWO Layer
        if input_dim == first_dim:
            w = torch.eye(input_dim)
            self.register_buffer("first_w", w)
            self.first_layer_not_param = True
            w_list = []
        else:
            w = nn.Parameter(torch.eye(first_dim, input_dim))
            apply_init_w(w, init_w, nonlinearity, nonlinearity_param)
            w_list = [nn.Parameter(w)]
            self.first_layer_not_param = False

        if isinstance(num_hidden_layers, int):
            num_hidden_layers = [num_hidden_layers] * first_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * first_dim
        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * first_dim
        if isinstance(num_hidden_layers, str):
            num_hidden_layers = num_hidden_layers.split(",")
            num_hidden_layers = [int(value) for value in num_hidden_layers]
        if isinstance(hidden_dim, str):
            hidden_dim = hidden_dim.split(",")
            hidden_dim = [int(value) for value in hidden_dim]

        nets = [
            SimpleRegressionNet(
                first_dim,
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

        # Subsequent IWO Layers
        for i in range(first_dim - 1):
            w = nn.Parameter(torch.eye(first_dim - i - 1, first_dim - i))
            apply_init_w(w, init_w, nonlinearity, nonlinearity_param)
            w_list.append(w)
            nets.append(
                SimpleRegressionNet(
                    first_dim - i - 1,
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

        # Store the layers in a ParameterList and ModuleList so that they are registered as parameters and modules
        self.w_list_params = nn.ParameterList(w_list)
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        outs = []
        if self.first_layer_not_param:
            x = torch.einsum("ij,bj->bi", self.first_w, x)
            outs.append(self.nets[0](x))
            for i, w in enumerate(self.w_list_params):
                x = torch.einsum("nj,bj->bn", w, x)
                outs.append(self.nets[i + 1](x))
            return outs
        for i, w in enumerate(self.w_list_params):
            x = torch.einsum("nj,bj->bn", w, x)
            outs.append(self.nets[i](x))
        return outs

    def get_w(self):
        w_list = []
        if self.first_layer_not_param:
            w_list.append(self.first_w.detach())
        for w in self.w_list_params:
            w_list.append(w.detach())
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
