# Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding 🚀

> 🏆 **Accepted to NeurIPS 2025**  

[![arXiv](https://img.shields.io/badge/arXiv-2503.13139-b31b1b.svg)](https://arxiv.org/abs/2503.13139)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-ff6f00.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/115148)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This is the official implementation for our NeurIPS 2025 paper: 
**“Logic-in-frames: Dynamic keyframe search via visual semantic-logical verification for long video understanding.”**  

Our method VSLS makes long video QA lighter by:

1. grounding objects and relations from the question,
2. searching keyframes with a `T*`-style heuristic guided by these cues,
3. sending only the useful frames to the VLM.

Each stage is script-based, so you can run or replace them separately.

---

## 1. Installation ⚙️

First install the external toolkits:

```bash
# 1) Query grounding interface (LLaVA-NeXT, or skip and use GPT API)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT

# 2) Image / grid scoring interface, e.g. YOLO-World
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```

Then create the environment:

```bash
conda env create -f environment.yml
conda activate haystack
# Make 'sys.path' include the directory which contains YOLO-Wolrd
export PYTHONPATH=$PYTHONPATH:your_YOLO-World_path
```
Potential issues encountered during installation：
```bash
# 1）PackagesNotFoundError: - pip=2.24.2*
# Set pip in environment.yml to pip >= 20.0
# 2）ModuleNotFoundError: No module named ‘mmcv._ext’, please try installing
pip install mmcv==2.0.0rc4
```

We used CUDA 12.1. If your CUDA version is different and encountered
 and `mmcv` or `mmyolo` fails, please follow the official guide: https://mmyolo.readthedocs.io.

## 2. Repository Structure 📁

```
VL-Haystack/
├── LLaVA-NeXT/                 # LLM-based query grounding and QA interface
├── YOLO-World/                 # Detector / image scoring backend
├── VSLS/                       # Core semantic-logical T* search
│   ├── interface_llm.py        # LLM interface for grounding and answering
│   ├── interface_yolo.py       # Detector interface for scoring frames
│   ├── interface_searcher.py   # T*-style search logic
│   ├── VSLSFramework.py        # Example class to connect search with QA
├── scripts/                    # End-to-end runnable scripts
│   ├── get_VSLS_grounding_objects.py   # Ground objects/relations for a video QA set
│   ├── get_VSLS_key_frames.py          # Search keyframes based on grounding
│   ├── get_qa_results.py               # Feed keyframes into VLM to get answers
│   ├── compute_qa_acc.py               # Compute QA accuracy
├── runs/                       # Example outputs for a quick start
├── README.md
```

Notes:

- You can skip cloning `LLaVA-NeXT` if you only plan to call an LLM API.
- For a new dataset, add its JSON parser in `utils/data_loader.py`.

## 3. Run the VSLS Pipeline 🏃

Below is a standard workflow for `VideoMME` or `LongVideoBench`. Change paths to your own.

### 3.1 Ground objects and relations

Set your OpenAI API key if you use GPT-based grounding:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

Run:

```bash
python scripts/get_VSLS_grounding_objects.py \
    --dataset VideoMME \
    --video_root ./Datasets/VideoMME \
    --obj_path ./runs/obj/obj_result.json
```

This will:
- read the dataset,
- ask the LLM to extract target objects, cue objects and relations,
- save them to ./runs/obj/obj_result.json.

Current datasets: `LongVideoBench`, `VideoMME`. To support others, extend `utils/data_loader.py`.

Output example:

```json
[
  {
    "video_id": "fFjv93ACGo8",
    "video_path": "/data/new-VL-Haystack/VL-Haystack/Datasets/Video-MME/videos/data/fFjv93ACGo8.mp4",
    "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?",
    "options": "A) Apples.\nB) Candles.\nC) Berries.\nD) The three kinds are of the same number.",
    "answer": "C",
    "duration_group": "short",
    "grounding_objects": {
      "target_objects": ["apples", "candles", "berries"],
      "cue_objects": ["Christmas tree", "decorations", "green branches"],
      "relations": [
        ["apples", "Christmas tree", "spatial"],
        ["candles", "Christmas tree", "spatial"],
        ["berries", "Christmas tree", "spatial"]
      ]
    },
    "task_type": "Counting Problem"
  }
]
```

### 3.2 Search keyframes

```bash
python scripts/get_VSLS_key_frames.py \
    --obj_path ./runs/obj/obj_result.json \
    --kfs_path ./runs/kfs/kfs_result.json
```

This calls the detector to score frames and then runs the VSLS T*-based search to select frames that best match the grounded cues.
For a quick check, we provide some sample results in runs/.

### 3.3 QA on selected frames

```bash
python scripts/get_qa_results.py \
    --kfs_path ./runs/kfs/kfs_result.json \
    --qa_path ./runs/qa/qa_results.json
```

This extracts the needed frames and feeds them into the target VLM to get the final answers. Results are saved to ./runs/qa/qa_results.json.

### 3.4 Compute accuracy

```bash
python scripts/compute_qa_acc.py \
    --qa_path ./runs/qa/qa_results.json \
python scripts/get_qa_results.py \
    --kfs_path ./runs/kfs/kfs_result.json \
    --qa_path ./runs/qa/qa_results.json
```

This extracts the needed frames and feeds them into the target VLM to get the final answers. Results are saved to `./runs/qa/qa_results.json`.


## 4. Dataset and Path Notes 📦

- `--video_root` must point to your actual video directory.
- Make sure the dataset JSON has the correct video_path or video_id so that the script can find the video.
- If you only need API-based LLM grounding (no local LLaVA), the grounding script already supports that.



## 5. Support 🛠️

If you meet issues, please open a GitHub issue with:
- OS and CUDA version
- full error log
- script and arguments

If there is no reply in 2 business days, you can email Weiyu Guo: `wguo395@connect.hkust-gz.edu.cn`.



## 6. Citation 📚

Please cite this work if you find this repository helpful:

```
@inproceedings{guo2025logic,
  title={Logic-in-frames: Dynamic keyframe search via visual semantic-logical verification for long video understanding},
  author={Guo, Weiyu and Chen, Ziyang and Wang, Shaoguang and He, Jianxiang and Xu, Yijie and Ye, Jinhui and Sun, Ying and Xiong, Hui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
}
```

