import torch
import random as rd

torch.manual_seed(1)
rd.seed(1)

from typing import Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

STEP = .005
MAX_T = 5
M = torch.ones(3, 1, device=DEVICE)
G = 1
MAX_MAGNITUDE = 3

T = torch.arange(start=0, end=MAX_T, step=STEP)

def setup(n: int) -> torch.Tensor:
    
    theta = torch.rand(n, device=DEVICE) * torch.pi + torch.pi / 2
    r = torch.rand(n, device=DEVICE)

    x1 = torch.stack((torch.ones(n, device=DEVICE), torch.zeros(n, device=DEVICE)), dim=1)
    x2 = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
    x3 = - torch.add(x1, x2)
    
    # [n, 3, 2] -> [n, body_id, (x,y)]
    return torch.stack((x1, x2, x3), dim=1)

def compute_acceleration(positions: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
    
    # positions: [n, 3, 2] -> [n, body_id, (x,y)]
    # acceleration: [n, 3, 3, 2] -> [n, body_id, other_body_id, (x,y)]

    for i in range(3):

        # Get the difference in coordinates between body i and the others (including itself)
        # [n, 3, 2]
        dxy = (positions[:, i].unsqueeze(-1) - positions.mT).mT

        # Get the distances (norm or hypotenuse) between body i and the others
        # [n, 3, 2]
        d = torch.norm(dxy, dim=-1, keepdim=True) ** 3

        # Compute the acceleration towards each body in each direction (x,y)
        acceleration[:, i] = torch.nan_to_num(-dxy * M * G / d)

    # Sum the xs and ys for each body to get the total acceleration
    # [n, 3, 2] -> [n, body_id, (x,y)]
    return torch.sum(acceleration, dim=2)

def clamp_magnitude(v: torch.Tensor) -> torch.Tensor:

    # v: [n, 3, 2]

    # Get all magnitudes [n, 3]
    magnitudes = torch.norm(v, dim=-1)

    # Get the mask of all large magnitudes
    large = magnitudes > MAX_MAGNITUDE
    if large.any():
        # Clamp the magnitude of all large vectors in v
        v[large] = v[large] * MAX_MAGNITUDE / magnitudes[large].unsqueeze(-1)

    # [n, 3, 2]
    return v

def run_euler(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # [n, t, 3, 2] -> [simulation_id, timestep, body_id, (x,y)]
    positions = torch.zeros(n, len(T), 3, 2, device=DEVICE)

    # [n, 3, 2] -> [simulation_id, body_id, (x,y)]
    velocities = torch.zeros(n, 3, 2, device=DEVICE)

    # [n, 3, 3, 2] -> [simulation_id, body_id, other_body_id, (x,y)]
    acceleration = torch.zeros(n, 3, 3, 2, device=DEVICE) 
    
    positions[:, 0] = setup(n)
    
    for i in tqdm(range(len(T) - 1)):

        # Get current accelerations
        total_acceleration = compute_acceleration(positions[:, i], acceleration)
        
        # Update the position with a small step in the direction of the velocity
        positions[:, i+1] = positions[:, i] + velocities * STEP

        # Update the velocities with a small step in the direction of the acceleration
        velocities = clamp_magnitude(velocities + total_acceleration * STEP)
    
    return positions

def animation_show(positions: torch.Tensor) -> None:
    
    # positions -> [t, 3, 2] -> [timestep, body_id, (x,y)]
    positions = positions.cpu()

    fig, ax = plt.subplots(figsize=(8, 8))
    iterator = list(zip(range(3), "gbr"))
    for i, c in iterator:
        ax.scatter(positions[0, i, 0], positions[0, i, 1], color=c)
    lines = [ax.plot(positions[0, i, 0], positions[0, i, 1], color=c)[0] for i, c in iterator]
    ax.grid()
    ax.set_xlim(positions[:, :, 0].min() - 0.2, positions[:, :, 0].max() + 0.2)
    ax.set_ylim(positions[:, :, 1].min() - 0.2, positions[:, :, 1].max() + 0.2)

    def update(frame):
        for i, line in enumerate(lines):
            line.set_xdata(positions[:frame, i, 0])
            line.set_ydata(positions[:frame, i, 1])
        return lines

    ani = FuncAnimation(fig=fig, func=update, frames=len(positions), interval=5)
    plt.show()

def generate_data(n_simulations: int = 2, save_dir: str = 'data/three_bodies_data_2D.pt') -> None:
    torch.save(run_euler(n=n_simulations), save_dir)

def show(positions: torch.Tensor) -> None:

    # positions -> [t, 3, 2] -> [timestep, body_id, (x,y)]
    positions = positions.cpu()

    iterator = list(zip(range(3), "gbr"))
    for i, c in iterator:
        plt.scatter(positions[0, i, 0], positions[0, i, 1], color=c)
        plt.plot(positions[:, i, 0], positions[:, i, 1], color=c)
    plt.grid()
    plt.xlim(positions[:, :, 0].min() - 0.2, positions[:, :, 0].max() + 0.2)
    plt.ylim(positions[:, :, 1].min() - 0.2, positions[:, :, 1].max() + 0.2)
    plt.savefig('show.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    d = run_euler(n=1)[0]
    show(d)
