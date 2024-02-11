import numpy as np
import random as rd

np.seterr(invalid='ignore')
rd.seed(0)

from typing import Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

step = 0.005
max_t = 5
M = np.ones(3)
G = 1
max_magnitude = 3
float_type = np.float32

t = np.arange(start=0, stop=max_t, step=step)

def setup() -> np.ndarray:
    
    theta = rd.random() * np.pi + np.pi / 2
    r = rd.random()

    x1 = (1, 0)
    x2 = (r * np.cos(theta), r * np.sin(theta))
    x3 = -np.add(x1, x2)

    return np.stack((x1, x2, x3), axis=1, dtype=float_type)

def show(x: np.ndarray, y: np.ndarray, filename: Optional[str] = None) -> None:
  
    plt.figure(figsize = (8, 8))
    for i, c in zip(range(3), "gbr"):
        plt.plot(x[:, i], y[:, i], color=c)
        plt.scatter(x[0, i], y[0, i], color=c)
    plt.grid()
    plt.xlim(x.min() - 0.2, x.max() + 0.2)
    plt.ylim(y.min() - 0.2, y.max() + 0.2)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def clamp_magnitude(v: np.ndarray) -> np.ndarray:
    
    magnitude = np.linalg.norm(v)
    return v * (max_magnitude / magnitude) if magnitude > max_magnitude else v

def compute_acceleration(x: np.ndarray, y: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    for i in range(3):
        
        dx = (x[i] - x).T
        dy = (y[i] - y).T
        
        d = np.sqrt(dx ** 2 + dy ** 2) ** 3
        
        ax[:, i] = np.nan_to_num(-dx * M * G / d)
        ay[:, i] = np.nan_to_num(-dy * M * G / d)
        
    ax_tot = np.sum(ax, axis=0)
    ay_tot = np.sum(ay, axis=0)

    return ax_tot, ay_tot

def run_euler(init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    x, y = np.zeros((2, len(t) + 1, 3), dtype=float_type)
    vx, vy = np.zeros((2, 3), dtype=float_type)
    ax, ay = np.zeros((2, 3, 3), dtype=float_type)
    
    x[0], y[0] = setup() if init is None else init

    for i in range(len(t)):

        ax_tot, ay_tot = compute_acceleration(x[i], y[i], ax, ay)

        x[i+1] = x[i] + vx * step
        y[i+1] = y[i] + vy * step

        vx, vy = np.apply_along_axis(
            func1d=clamp_magnitude, axis=1,
            arr=np.stack((vx + ax_tot * step, vy + ay_tot * step), axis=1)
        ).T

    return x, y

def animation_show(x: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    iterator = list(zip(range(3), "gbr"))
    for i, c in iterator:
        ax.scatter(x[0, i], y[0, i], color=c)
    lines = [ax.plot(x[0, i], y[0, i], color=c)[0] for i, c in iterator]
    ax.grid()
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 0.2, y.max() + 0.2)

    def update(frame):
        for i, line in enumerate(lines):
            line.set_xdata(x[:frame, i])
            line.set_ydata(y[:frame, i])
        return lines

    ani = FuncAnimation(fig=fig, func=update, frames=len(x), interval=5)
    plt.show()

def generate_data(n_simulations: int = 10_000, data_dir = 'data/') -> None:
    
    # [Simulation_id, (x, y), timestep, body_id]
    data = np.zeros((n_simulations, 2, len(t) + 1, 3), dtype=float_type)
    
    for i in tqdm(range(n_simulations)):
        data[i] = np.stack(run_euler())
    assert data.dtype == float_type
    np.save(data_dir + 'all_data.npy', data)


if __name__ == "__main__":
    generate_data()
