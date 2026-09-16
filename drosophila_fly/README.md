# 🪰 Fruit Fly Optomotor Vision (`drosophila_fly`)

A model of visual motion detection and flight stabilization inspired by the fruit fly (Drosophila).

## How to Run
```bash
python3 drosophila_fly/fly_optomotor_sim.py
```

## What It Shows
- Uses elementary motion detectors (EMDs) across a 16-channel eye to measure optical drift.
- Automatically stabilizes yaw against aerodynamic wind turbulence.
- Simulates rapid escape saccades when a looming shadow approaches.
