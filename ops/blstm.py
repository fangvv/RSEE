import torch
import torch.nn as nn


class BLSTM_IRTENet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_classes = kwargs["num_classes"]

        self.blstm = nn.LSTMCell(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            bias=True
            ) # [2048, 512]

        # NOTE 2: [0, 1] 0 for not exit, 1 for exit
        # self.exit = nn.Linear(self.hidden_dim, 2)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_dim)
        

    
    def forward(self, x, h_c_pair):
        h, c = h_c_pair
        if h.shape[-1] != self.hidden_dim:
            h = self.fc1(h)
        feat, c = self.blstm(x, (h, c))
        # exit_out = self.exit(feat)
        feat = self.fc2(feat)
        return None, feat, c


class STAN_LSTM(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_classes = kwargs["num_classes"]

        self.blstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
        )
        self.exit = nn.Linear(self.hidden_dim * 2, 2)

    def forward(self, x, h_c_pair):
        h, c = h_c_pair
        # x = self.fc(x)
        feat, (h, c) = self.blstm(x, (h, c))
        exit_out = self.exit(feat)
        return exit_out, feat, h, c

class MaxPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: tuple):
        last, curr = x
        out = torch.cat((last.unsqueeze(1), curr.unsqueeze(1)), dim=1)
        out = out.max(dim=1)[0]
        return out


class AvePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: tuple):
        last, curr = x
        out = torch.cat((last.unsqueeze(1), curr.unsqueeze(1)), dim=1)
        out = out.mean(dim=1)
        return out

if __name__ == "__main__":
    from thop import profile
    model = BLSTM_IRTENet(input_dim=2048, hidden_dim=512, num_classes=51)
    x = torch.randn(1, 2048)
    hx, cx = torch.randn(1, 512), torch.randn(1,512)
    flops, params = profile(model, (x, (hx, cx)))
    print(flops / 1e9, params / 1e6)
