import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

def generate_neural_nebula(num_stars=80000, num_arms=2):
    arm_stars = int(num_stars * 0.7)
    core_stars = num_stars - arm_stars
    
    core_x = np.random.normal(0, 1.5, core_stars)
    core_y = np.random.normal(0, 1.5, core_stars)
    core_z = np.random.normal(0, 0.6, core_stars)
    
    arm_angles = np.random.uniform(0, 3 * np.pi, arm_stars)
    arm_radius = 2 * arm_angles 
    arm_offset = np.random.randint(0, num_arms, arm_stars) * (2 * np.pi / num_arms)
    
    noise_x = np.random.normal(0, 1.2, arm_stars)
    noise_y = np.random.normal(0, 1.2, arm_stars)
    noise_z = np.random.normal(0, 0.4, arm_stars)
    
    arm_x = (arm_radius + noise_x) * np.cos(arm_angles + arm_offset)
    arm_y = (arm_radius + noise_y) * np.sin(arm_angles + arm_offset)
    arm_z = noise_z
    
    x = np.concatenate([core_x, arm_x])
    y = np.concatenate([core_y, arm_y])
    z = np.concatenate([core_z, arm_z])
    
    return x, y, z

class NeuralColorMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid() 
        )
    def forward(self, coords):
        return self.net(coords)

def render_galaxy(image_filename, output_filename):
    x, y, z = generate_neural_nebula()
    
    try:
        img = Image.open(image_filename).convert('RGB')
    except FileNotFoundError:
        return

    img_data = np.array(img) / 255.0
    height, width, _ = img_data.shape

    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    pixel_x = np.clip((x_norm * (width - 1)).astype(int), 0, width - 1)
    pixel_y = np.clip(((1 - y_norm) * (height - 1)).astype(int), 0, height - 1)

    target_colors = img_data[pixel_y, pixel_x]
    
    coords = np.vstack((x, y, z)).T
    inputs = torch.tensor(coords, dtype=torch.float32)
    targets = torch.tensor(target_colors, dtype=torch.float32)

    model = NeuralColorMapper()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    epochs = 250
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    final_colors = model(inputs).detach().numpy()

    fig = plt.figure(figsize=(7, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axis('off')

    scatter = ax.scatter(x, y, z, c=final_colors, s=0.1, alpha=0.6)

    def update_camera(frame):
        total_frames = 150
        progress = frame / total_frames
        
        current_azim = frame * 1.5 
        current_elev = 20 - (15 * progress)
        ax.view_init(elev=current_elev, azim=current_azim)
        
        zoom_factor = 1.0 - (progress ** 2.5) * 0.98 
        limit = 7.0 * zoom_factor
        
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        
        return scatter,

    ani = animation.FuncAnimation(fig, update_camera, frames=150, interval=33, blit=False)

    try:
        ani.save(output_filename, writer='ffmpeg', fps=30, dpi=150)
    except Exception:
        pass

if __name__ == "__main__":
    render_galaxy('galaxy.webp', 'nebula_dive.mp4')
  
