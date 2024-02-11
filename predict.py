from generator import setup, show
import torch

from model import ThreeBodiesModel

m = ThreeBodiesModel()
with torch.inference_mode():
    m.load_state_dict(torch.load('models/model.pt', map_location='cpu'))
    out: torch.Tensor = m(setup(1).flatten(-2))[0]
    out = out.view(len(out), 3, 2)
print(out.shape)
show(out)
