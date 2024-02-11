import torch
import torch.nn as nn

from typing import Optional

class ThreeBodiesModel(nn.Module):

    def __init__(self) -> None:
    
        super().__init__()
        self.input_layer = nn.Linear(in_features=6, out_features=64)
        self.lstm = nn.LSTMCell(input_size=64, hidden_size=64)
        self.output_layer = nn.Linear(in_features=64, out_features=6)

    def forward(self, x: torch.Tensor, steps: Optional[int] = 1000):
        
        h, c = torch.zeros(2, len(x), 64).to(x.device)
        output = []
        if x.ndim == 3:
            for i in range(x.shape[1]):
                h, c = self.lstm(self.input_layer(x[:, i]), (h, c))
                output.append(self.output_layer(h))
        else:
            for i in range(steps):
                h, c = self.lstm(self.input_layer(x), (h, c))
                x = self.output_layer(h)
                output.append(x)

        return torch.stack(output, dim=1)
    
if __name__ == '__main__':
    x = torch.rand(5, 6)
    m = ThreeBodiesModel()
    out = m(x)
    print(out.shape)