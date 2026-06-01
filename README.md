# Neural-Nebula: Generative 3D Galaxy & Deep Dive Animation

This project programmatically generates a 3D point-cloud representation of a spiral galaxy and trains a lightweight PyTorch neural network to map realistic colors onto the coordinates based on a source image. Finally, it uses Matplotlib to render a cinematic, vertical "deep dive" flythrough animation suitable for mobile platforms.

## Overview
1. **Mathematical Structure:** Generates an 80,000-point 3D spiral galaxy structure using normal distributions and trigonometric offsets.
2. **AI Color Mapping:** Uses an MLP (Multi-Layer Perceptron) neural network to sample colors from a reference image (`galaxy.webp`) and predict the optimal `(R, G, B)` values for each `(x, y, z)` coordinate in the 3D space.
3. **Cinematic Render:** Animates the camera view by smoothly adjusting elevation, azimuth, and axis limits to create a hyper-zoom "deep dive" effect, saving the output as `nebula_dive.mp4`.

## Project Demo
You can watch the final rendered AI galaxy deep dive directly on YouTube Shorts.
**Watch the video here:** https://youtube.com/shorts/MCwKfPmJ6XY?feature=share

## Prerequisites
* Python 3.x
* PyTorch (`torch`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* Pillow (`Pillow`)
* FFmpeg (Required for saving the `.mp4` animation)

## Usage
1. Place a reference galaxy image named `galaxy.webp` in the root directory.
2. Run the script:
```bash
   python neural_nebula_dive.py
