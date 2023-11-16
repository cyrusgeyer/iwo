import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils.utils import batch_eye


class IWOModel(nn.Module):
    # This convolutional network is equivalent to num_factor independent MLPs
    def __init__(
        self,
        model_cfg,
        input_dim,
        factor_sizes,
        factor_discrete,
    ):
        super().__init__()
        self.num_factors = len(factor_sizes)

        # For now, this impl only supports first_dim same as the input_dim
        assert input_dim == model_cfg["first_dim"]

        self.model = ConvIndependentMLPs(
            self.num_factors,
            factor_sizes,
            factor_discrete,
            model_cfg["first_dim"],
            model_cfg["num_hidden_layers"],
            model_cfg["hidden_dim"],
        )

    def get_w(self):
        w_list = [[] for _ in range(self.num_factors)]
        for k in range(self.num_factors):
            w_list[k].append(None)
            for layer in self.model.iwo_layers:
                w_list[k].append(layer.weights_down.detach()[k])
            # Set the first w to identity
            w_list[k][0] = torch.eye(self.first_dim).to(w_list[k][1].device)
        return w_list

    def forward(self, x):
        return self.model(x)


class ConvIndependentMLPs(nn.Module):
    def __init__(
        self,
        num_factors,
        factor_sizes,
        factor_discrete,
        input_dim,
        num_hidden_layers=1,
        hidden_dim=0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_sizes = factor_sizes
        self.factor_discrete = factor_discrete
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        iwo_layers = []
        for d in range(input_dim, 1, -1):
            iwo_layers.append(IWOLayer(d, d - 1, num_factors, input_dim))

        self.iwo_layers = nn.Sequential(*iwo_layers)

        if self.hidden_dim == 0:
            self.num_hidden_layers = 0
            self.hidden_dim = self.input_dim

        if self.num_hidden_layers == 0:
            self.hidden_dim = self.input_dim

        list_ind_mlps = []
        # input has size: B x D * num_factors x L x 1
        # each independent linear layer will act only on one group of features
        # this is equivalent with having num_factos separate networks
        if self.num_hidden_layers > 0:
            # list_ind_mlps.append(nn.Linear(self.input_dim, self.hidden_dim))
            group_independent_conv = nn.Conv1d(
                in_channels=num_factors * self.input_dim * self.input_dim,
                out_channels=num_factors * self.hidden_dim * self.input_dim,
                kernel_size=1,
                groups=num_factors * self.input_dim,
            )
            list_ind_mlps.append(group_independent_conv)
            list_ind_mlps.append(nn.ReLU(True))
            # nn.init.dirac_(group_independent_conv.weight,groups=num_factors * self.input_dim)

            for i in range(self.num_hidden_layers - 1):
                # list_ind_mlps.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                group_independent_conv = nn.Conv1d(
                    in_channels=num_factors * self.hidden_dim * self.input_dim,
                    out_channels=num_factors * self.hidden_dim * self.input_dim,
                    kernel_size=1,
                    groups=num_factors * self.input_dim,
                )
                list_ind_mlps.append(group_independent_conv)
                list_ind_mlps.append(nn.ReLU(True))
                # nn.init.dirac_(group_independent_conv.weight,groups=num_factors * self.input_dim)

        self.ind_mlps = nn.Sequential(*list_ind_mlps)
        ind_pred = []
        for i in range(num_factors):
            for _ in range(input_dim):
                if self.factor_discrete[i]:
                    predictor = nn.Linear(self.hidden_dim, self.factor_sizes[i])
                else:
                    predictor = nn.Linear(self.hidden_dim, 1)
                ind_pred.append(predictor)

        self.ind_pred = nn.ModuleList(ind_pred)

    def forward(self, x):
        b = x.shape[0]
        # x: B x L
        x = x.unsqueeze(1).repeat(1, self.num_factors, 1)
        # x: B x F x L

        # apply the IWO layers
        out_list = [x.permute(0, 2, 1)]
        _, out_list = self.iwo_layers([x, out_list])

        x = torch.cat(out_list, dim=1)
        x = x.transpose(1, 2).contiguous().view(b, -1).unsqueeze(-1)

        # apply the independent MLPs on each group
        x = self.ind_mlps(x)
        x = x.view(x.shape[0], self.num_factors, -1)

        all_preds = []

        for i in range(self.num_factors):
            preds = []
            for l in range(self.input_dim):
                inp_il = x[:, i, self.hidden_dim * l : self.hidden_dim * (l + 1)]
                pred_il = self.ind_pred[i * self.input_dim + l](inp_il)
                if self.factor_discrete[i]:
                    pred_il = F.softmax(pred_il, dim=-1)
                preds.append(pred_il)
            all_preds.append(preds)

        return all_preds


class IWOLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_factors, L):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        weights_down = batch_eye(num_factors, output_dim, input_dim)
        self.weights_down = nn.Parameter(
            weights_down
        )  # nn.Parameter is a Tensor that's a module parameter.
        torch.nn.init.kaiming_uniform_(self.weights_down)
        weights_up = batch_eye(num_factors, L, output_dim)
        self.weights_up = nn.Parameter(weights_up)
        torch.nn.init.kaiming_uniform_(self.weights_up)

    def forward(self, _input):
        x, out_list = _input
        x_down = torch.einsum(
            "foi,bfi->bfo", self.weights_down, x
        )  # o: output_dim, i: input_dim, f: num_factors, b: batch_size
        x_up = torch.einsum(
            "flo,bfo->bfl", self.weights_up, x_down
        )  # o: output_dim, l: dim for subsequent NN, f: num_factors, b: batch_size
        out_list.append(x_up.permute(0, 2, 1))
        return [x_down, out_list]
