# OpenEgo: A Multimodal Egocentric Dataset for Dexterous Manipulation

<div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="https://www.openegocentric.com">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/abs/2509.05513">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="#citation">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

<p align="center">
  <strong>1107 hours | 119.6M frames | 290 manipulation tasks | 600+ environments | 344.5k recordings</strong>
</p>

## Overview

OpenEgo is a multimodal egocentric manipulation dataset with standardized hand-pose annotations and intention-aligned action primitives. The dataset consolidates six public egocentric datasets to enable learning dexterous manipulation from egocentric video and support reproducible research in vision-language-action learning.

### Key Features

- **1107 hours** of egocentric video data across 119.6M frames
- **290 manipulation tasks** spanning kitchen activities, assembly, and daily tasks
- **344.5k recordings** in 600+ unique environments (10 kitchens, 610 indoor rooms)
- **Standardized 21-joint MANO hand poses** in camera coordinate frame
- **Intention-aligned language annotations** with timestamped action primitives
- **Unified format** across all datasets for consistent API access

## Dataset Composition

OpenEgo consolidates six public egocentric datasets:

| Dataset | Hours | Frames | Tasks | Recordings | Fine-Grained | Dexterous | License |
|---------|-------|--------|-------|------------|--------------|-----------|---------|
| **CaptainCook4D** | 54 | 5.6M | 24 | 200 | ✗ | ✗ | Apache 2.0 |
| **HOI4D** | 44 | 2.4M | 16 | 4k | ✗ | ✓ | CC BY-NC 4.0 |
| **HoloAssist** | 166 | 17.9M | 20 | 2.2k | ✓ | ✓ | CDLA v2 |
| **EgoDex** | 829 | 90M | 194 | 338k | ✗ | ✓ | CC BY-NC-ND 4.0 |
| **HOT3D** | 13.3 | 3.7M | 33 | 19 | ✗ | ✓ | CC BY-SA/BY-NC-SA 4.0 |
| **HO-Cap** | 0.67 | 73k | 3 | 64 | ✗ | ✓ | CC BY 4.0 |

All datasets are processed to include:
- Unified 21-joint MANO hand pose format in camera coordinates
- Intention-aligned action primitives with timestamps
- Standardized metadata and annotations

## Installation

```bash
# Clone the repository
git clone https://github.com/ahadjawaid/openego.git
cd openego

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with visualization support
pip install -e ".[vis]"
```

## Quick Start

```python
from openego import OpenEgoDataProvider, Action
from pathlib import Path

# Initialize the data provider
data_provider = OpenEgoDataProvider(
    data_dir=Path('path/to/openego/data'),
    data_types=['joint', 'rgb', 'annotation', 'metadata']
)

# Get dataset statistics
print(f"Number of demonstrations: {data_provider.num_demos}")
print(f"Total duration: {data_provider.duration/3600:.1f} hours")
print(f"Available benchmarks: {data_provider.benchmarks}")

# Access a demonstration
demo_data = data_provider[0]

# Access different data modalities
rgb_frames = demo_data['rgb']  # Video frames as numpy array
joints = demo_data['joint']  # Hand joint positions
annotation = demo_data['annotation']  # Action annotations
metadata = demo_data['metadata']  # Video metadata

# Work with action annotations
for action_dict in annotation['actions']:
    action = Action(**action_dict)
    print(f"Action: {action.label}")
    print(f"Duration: {action.end_timestamp - action.start_timestamp:.2f}s")
    print(f"Objects: {action.objects}")
```

## Data API

The OpenEgo API provides easy access to the multimodal dataset:

### OpenEgoDataProvider

The main interface for loading OpenEgo data:

```python
from openego import OpenEgoDataProvider
from pathlib import Path

provider = OpenEgoDataProvider(
    data_dir=Path('path/to/data'),
    data_types=['joint', 'rgb', 'annotation', 'metadata']  # Select modalities
)

# Access demonstration by index
demo = provider[0]

# Get dataset statistics
num_demos = provider.num_demos
total_frames = provider.num_frames
duration_hours = provider.duration / 3600
```

### Action Annotations

Work with intention-aligned action primitives:

```python
from openego import Action

# Load action from annotation
for action_dict in demo_data['annotation']['actions']:
    # Add required fields
    action_dict['fps'] = 30
    action_dict['video_joints'] = demo_data['joint']
    action_dict['video_path'] = Path('path/to/video.mp4')
    
    action = Action(**action_dict)
    
    # Access properties
    print(f"Action: {action.label}")
    print(f"Objects: {action.objects}")
    print(f"Actors: {action.actors}")  # left_hand, right_hand, both_hands
    print(f"Duration: {action.end_timestamp - action.start_timestamp:.2f}s")
    
    # Get hand trajectories for this action
    left_hand = action.left_hand_joints  # [frames, 21, 3]
    right_hand = action.right_hand_joints  # [frames, 21, 3]
    
    # Get 2D pixel coordinates
    left_pixels = action.left_hand_pixel_joints  # [frames, 21, 2]
```

## Dataset Structure

The OpenEgo dataset follows a standardized directory structure:

```
openego/
├── <benchmark_name>/              # e.g., HO-Cap, HOI4D, etc.
│   └── demo_<number>/            # e.g., demo_0000, demo_0001
│       ├── video.mp4             # RGB video file
│       │   # If video not available:
│       │   └── <video_name>.txt  # Contains URL or ID of video
│       ├── annotation.json       # Action annotations
│       │   ├── task: str         # Task description
│       │   ├── actions: List     # List of actions
│       │   │   ├── start_timestamp: float
│       │   │   ├── end_timestamp: float
│       │   │   ├── objects: List[str]
│       │   │   ├── actors: List[str]
│       │   │   └── label: str
│       │   └── video_info: dict
│       │       ├── num_frames: int
│       │       ├── duration: float
│       │       ├── fps: int
│       │       ├── height: int
│       │       └── width: int
│       ├── joints.hdf5           # Hand joint data
│       │   ├── left_hand: [num_frames, num_joints, 3] float32
│       │   ├── right_hand: [num_frames, num_joints, 3] float32
│       │   ├── left_hand_visibility: [num_frames] int
│       │   ├── right_hand_visibility: [num_frames] int
│       │   ├── joint_names: List[str]
│       │   └── intrinsics: [3, 3] float64
│       ├── metadata.hdf5         # Video metadata
│       │   ├── intrinsics: [3, 3]
│       │   ├── num_frames: int
│       │   ├── fps: int
│       │   ├── width: int
│       │   └── height: int
│       ├── original_metadata.hdf5 # Original dataset metadata
│       └── license.txt           # Optional license file
│
├── EgoDex/                       # Special case - must preserve original structure
│   ├── part<number>/
│   │   └── <task_name>/
│   │       ├── <demo_num>.mp4   # Video file
│   │       └── <demo_num>.hdf5  # Joint data
│   └── annotations/
│       └── part<number>/
│           └── <task_name>/
│               └── <demo_num>.json

```

### Data Format Details

#### annotation.json
Contains high-level task description and fine-grained action primitives:
```json
{
    "task": "Picking up and moving various objects on a table",
    "actions": [
        {
            "start_timestamp": 1.0,
            "end_timestamp": 3.6,
            "objects": ["Tazo green tea carton"],
            "actors": ["left_hand"],
            "label": "left hand grasps, lifts, and then places the Tazo green tea carton back on the table"
        }
    ],
    "video_info": {
        "num_frames": 676,
        "duration": 22.53,
        "fps": 30,
        "height": 720,
        "width": 1280
    }
}
```

#### joints.hdf5
- **left_hand/right_hand**: 3D joint positions in camera coordinates [num_frames, 21, 3]
- **left_hand_visibility/right_hand_visibility**: Binary visibility flags [num_frames]
- **joint_names**: MANO joint naming (wrist + 4 joints per finger)
- **intrinsics**: Camera intrinsic matrix [3, 3] for 3D→2D projection

### Language Annotations

OpenEgo provides intention-aligned language primitives that:
- Specify manipulated objects and actions with timestamps
- Include actor labels (left_hand, right_hand, both_hands, person)
- Describe complete action sequences from intention onset to completion
- Examples: "navigate to desk", "remove GoPro camera from case", "right hand unzips black camera case while left hand holds it"

## Download

We will be releasing all the data throughout the coming weeks. A download script will be added soon to facilitate dataset access. In the meantime, please download the datasets on our box: [dataset download link](https://utdallas.box.com/s/1fzj9gr331atfumk5on3bifzpp2idnjr).

## Project Structure

```
openego/
├── openego/                 # Main package
│   ├── __init__.py
│   ├── data/               # Data loading modules
│   │   ├── __init__.py
│   │   ├── openego.py      # OpenEgoDataProvider
│   │   └── annotations.py  # Action annotation classes
│   └── core/               # Core utilities
│       ├── __init__.py
│       ├── constants.py    # Joint names and mappings
│       ├── utils.py        # Video and file utilities
│       └── projection.py   # 3D to 2D projection
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_openego_provider.py
│   └── test-data/          # Sample data for testing
├── licenses/               # Dataset licenses
├── ATTRIBUTION.md          # Dataset attributions
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── pytest.ini             # Test configuration
├── run_tests.py           # Test runner script
└── README.md              # This file
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
python run_tests.py

# Or use pytest directly
pytest

# Run tests with coverage
pytest --cov=openego

# Run specific test file
pytest tests/test_openego_provider.py
```

## License

The OpenEgo codebase is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The datasets included in OpenEgo retain their original licenses. Please refer to:
- Individual dataset licenses in the `licenses/` directory
- Detailed attribution in [ATTRIBUTION.md](ATTRIBUTION.md)


## Applications

OpenEgo supports research in:
- **Language-conditioned imitation learning** for dexterous manipulation
- **3D hand trajectory prediction** from egocentric observations
- **Vision-language-action (VLA) models** with hierarchical action primitives
- **World models** for manipulation planning
- **Human-to-robot transfer** of dexterous skills

## Citation

If you use OpenEgo in your research, please cite:

```bibtex
@article{jawaid2025openego,
  title={OpenEgo: A Multimodal Egocentric Dataset for Dexterous Manipulation},
  author={Jawaid, Ahad and Xiang, Yu},
  year={2025},
  archivePrefix={arXiv},
  eprint={2509.05513}
}
```

## License and Attribution

OpenEgo inherits licenses from its constituent datasets. Please refer to:
- Individual dataset licenses in the `licenses/` directory
- Detailed attribution in `ATTRIBUTION.md`

For EgoDex (CC BY-NC-ND), annotation files are available with permission; users must retrieve the underlying data from the official source.

## Contact

For questions and issues, please open an issue on GitHub or visit [www.openegocentric.com](https://www.openegocentric.com).

## Acknowledgments

We thank the authors of the original datasets for making their data publicly available. This work consolidates:
CaptainCook4D, HOI4D, HoloAssist, EgoDex, HOT3D, and HO-Cap.
