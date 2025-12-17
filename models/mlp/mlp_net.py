import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _params_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.apply(_params_init)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)

        out = self.linear2(out)

        return out
