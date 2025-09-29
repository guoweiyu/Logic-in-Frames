# VSLS: Visual Semantic-Logical Keyframe Search Framework


**TStar** is a comprehensive framework designed to integrate **KeyFrame Searching** into Vision-Language Models (VLMs) to enhance Video Question Answering (VQA). By leveraging efficient keyframe searching, TStar dynamically identifies relevant frames in videos, enabling state-of-the-art VLMs like **LLaVA** to achieve improved performance in understanding and reasoning over video data.

## Features
- **Keyframe Searching**: Dynamically detects and extracts relevant keyframes.
- **Modular Design**: Easily integrates various grounding and searching backends.
- **Efficient Video QA**: Combines T* searching with advanced VQA.

---

## Getting Started
### Installation

```bash
## Follow docs/installation to implemet Grounding (e.g., LLaVA) and Searching (e.g., YOLO) Function
###  Install Query Grounder Interface(LLaVA or GPT-API)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
### Install Image Grid Scorer Interface e.g., YOLO-WORLD
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

### Environment
```bash
conda env create -f environment.yml
conda activate haystack
```

Currently, the CUDA version in our experimental setup is 12.1. If you encounter any problems when installing mmcv, mmyolo or other packages with different CUDA versions, try to follow this [official guide](https://mmyolo.readthedocs.io) for help..

### Structure:
```bash
VL-Haystack/
├── LLaVA-NeXT/                # Query grounding and QA interface (LLaVA or skip by using GPT-4o-API)
├── YOLO-World/                # Heuristic-based image scoring and searching (YOLO)
├── TStar/                     # Core T* searching and framework integration
│   ├── interface_llm.py       # LLM-based interface for question grounding and answering
│   ├── interface_yolo.py      # YOLO-based object detection interface
│   ├── interface_searcher.py  # Searching logic for T* heuristic processing
│   ├── TStarFramewor.py  # Demonstration class for integrating T* searching with QA
├── scripts              # Script for running VSLS pipeline
│   ├── get_Tstar_grounding_objects.py  # Grounding objects and relations for given Video QA dataset 
│   ├── ger_Tstar_key_frames.py # Performing keyframe search based on object grounding results
│   ├── ger_qa_results.py # Feeding keyframes into VLM to get qa results



├── README.md                  # Project readme

```

## Run VSLS Pipeline:
### get grounding objects

```bash
export OPENAI_API_KEY=your_openai_api_key

python scripts/get_Tstar_grounding_objects.py \
    --dataset VideoMME \
    --video_root ./Datasets/ego4d/ego4d_data/v1/256p \
    --obj_path ./runs/obj/obj_result.json 
```
Running the command above will ground objects and relations using LLMs(gpt-4o) for your Video QA dataset. Currently we support LongVideoBench and VideoMME. You can add customized json parsing function in `utils/data_loader.py` for other Video QA datasets. [video_root] stands for the root directory where the videos are saved. The object grounding results will be saved in [obj_path], and the json file should look like:

```json
[
    {
        "video_id": "fFjv93ACGo8",
        "video_path": "/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/Video-MME/videos/data/fFjv93ACGo8.mp4",
        "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?",
        "options": "A) Apples.\nB) Candles.\nC) Berries.\nD) The three kinds are of the same number.",
        "answer": "C",
        "gt_frame_index": [],
        "duration_group": "short",
        "position": [],
        "grounding_objects": {
            "target_objects": [
                "apples",
                "candles",
                "berries"
            ],
            "cue_objects": [
                "Christmas tree",
                "decorations",
                "green branches"
            ],
            "relations": [
                [
                    "apples",
                    "Christmas tree",
                    "spatial"
                ],
                [
                    "candles",
                    "Christmas tree",
                    "spatial"
                ],
                [
                    "berries",
                    "Christmas tree",
                    "spatial"
                ]
            ]
        },
        "task_type": "Counting Problem"
    },
    ...
]    
```

### search key frames

``` bash
python scripts/get_Tstar_key_frames.py \
    --obj_path ./runs/obj/obj_result.json \
    --kfs_path ./runs/kfs/kfs_result.json
```

Then we can move to the next and search key frames based on previous object grounding results. We only need to specify an object grouding result file and the output key frame search result path. For a quick start, we already provide some raw experimental results in `runs/`, which can be used for a simple test.

