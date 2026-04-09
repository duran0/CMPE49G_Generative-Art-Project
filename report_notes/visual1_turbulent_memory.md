# Turbulent Memory

Story:
A field of drifting trajectories suggests recollections that keep reshaping themselves. The motion is directional, but the diffusion term keeps the image unstable and imperfect.

Method:
Particles move through a Perlin-noise flow field with Brownian diffusion added at every step. Thin trails build up the image over time and the final particle positions remain visible as brighter endpoints.

Parameters:
- Seed: 77
- Particles: 300
- Steps: 600
- Diffusion coefficient D: 0.0036
- Time step dt: 1.0
- Drift strength: 0.014
- Noise scale: 2.4
- Octaves: 2
