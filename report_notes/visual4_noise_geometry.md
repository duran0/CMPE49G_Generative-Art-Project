# Noise-Driven Geometry

Story:
Instead of tracing motion, this composition turns a noise field into a structured surface. Repetition provides order while local changes in radius, direction, and brightness keep the grid from feeling mechanical.

Method:
A regular grid is sampled with Perlin noise. Each point uses the sampled values to control circle radius, line direction, opacity, and color intensity, producing a geometric field with organic variation.

Parameters:
- Seed: 42
- Grid size: 29 x 29
- Noise scale: 1.55
- Noise octaves: 4
- Circle radius range: 0.009 to 0.043
