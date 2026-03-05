# Golfball-tracker

A project to detect and track a golf balls flight and then predict its trajectory and draw it.

![Demo](results/output.gif)

## Project Structure

```
golf-ball-tracker/
│
├── data/               # Place input videos here
├── src/
│   ├── detection.py     # Detection of the golf ball
│   ├── tracking.py      # Kalman Filter implementation
│   ├── trajectory.py    # Parabolic curve fitting
│   ├── visualization.py # Drawing utilities
│   └── main.py          # Main execution pipeline
│
├── results/            # Output videos will be saved here
├── README.md
└── requirements.txt
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the tracker on a video file:

```bash
python src/main.py --video data/my_golf_shot.mp4
```

The processed video with tracking overlays will be saved to `results/output.mp4`.