from generator import run_euler, setup, show
import numpy as np
import random as rd
import torch

from model import ThreeBodiesModel

rd.seed(123)

s = setup()
x, y = run_euler(s)

m = ThreeBodiesModel()
with torch.inference_mode():
    m.load_state_dict(torch.load('model.pth', map_location='cpu'))
    out: torch.Tensor = m(torch.from_numpy(s).flatten().unsqueeze(0), steps=50)[0]
print(out[-1])
print(x[50], y[50])

