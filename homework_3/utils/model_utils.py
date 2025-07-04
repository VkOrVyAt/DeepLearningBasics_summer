import torch.nn as nn

class MLP_1(nn.Module):
    def __init__(self, num_layers=3, input_size=784, hidden_size=128, output_size=10,
                 use_dropout=False, dropout_rate=0.3, use_batchnorm=False):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Flatten())

        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class MLP(nn.Module):
    """Многослойный персептрон с заданными размерами скрытых слоев."""
    def __init__(self, hidden_sizes: list, input_size: int = 784, output_size: int = 10,
                 use_dropout: bool = False, dropout_rate: float = 0.0, use_bn: bool = False):
        super().__init__()
        layers = []
        in_feats = input_size

        for idx, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_feats, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if use_dropout and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_feats = h

        layers.append(nn.Linear(in_feats, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # ожидаем, что x уже имеет форму (batch, features)
        return self.net(x.view(x.size(0), -1))