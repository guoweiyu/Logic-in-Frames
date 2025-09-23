'''
get_qa_results.py

3rd step of the pipeline: perform qa inference based on keyframe searching results

'''


import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.append('./')
from TStar.interface_llm import TStarUniversalGrounder

import argparse
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


FILTER_TASK_TYPES = ['OCR Problems', 'Counting Problem', 'Temporal Perception', 'Information Synopsis', 'Temporal Reasoning']

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_video_fps(video_path: str) -> float:
    """
    get fps of the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        raise ValueError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        logger.error(f"Unable to get fps: {video_path}")
        raise ValueError(f"Unable to get fps: {video_path}")
    logger.debug(f"{video_path} fps: {fps}")
    return fps

def extract_frames(video_path: str, frame_indices: List[int] = None, numframe: int = 16) -> List[Optional[Image.Image]]:
    """
    extract certain frames from video
    convert the frames to PIL format
    uniform sample numframe frames if frame_indices is None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # uniform sample numframe frames if frame_indices is None
    if frame_indices is None:
        frame_indices = np.linspace(0, total_frames - 1, numframe, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if numframe > 8:
                w, h = pil_image.size
                pil_image = pil_image.resize((int(w/2), int(h/2)), Image.Resampling.LANCZOS)
            frames.append(pil_image)
        else:
            frames.append(None)  # None for error in reading frame

    cap.release()
    return frames

def compute_qa_accuracy(
    result_data: List[Dict[str, Any]],
    tstar_grounder: TStarUniversalGrounder,
    frame_key: str ="uniform",
    ground_truth_key: str = "answer",
    output_file: str = "Rebuttal/qa_results.jsonl",
    frame_num: int = 4,
    sample_method: str="possibility"
) -> Tuple[Dict, List[Dict[str, Any]]]:
    """
    perform qa based on key frame searching results result_data
    compute the accuracy based on ground truth answer
    save the qa results to output_file
    """
    qa_results = []
    correct_count = 0
    total_count = 0
    correct_table = {15:{'correct_count': 0, 'total_count' : 0}, 
                         60:{'correct_count': 0, 'total_count' : 0},
                         600:{'correct_count': 0, 'total_count' : 0},
                         3600:{'correct_count': 0, 'total_count' : 0}}

    # use cache to avoid repeating loading video fps
    fps_cache = {}
          
    # compute accuracy by category
    for idx, entry in tqdm(enumerate(result_data), desc="Extract frames and performing QA"):
        try:
            task_type = entry.get("task_type", "")
            print(task_type)
            if task_type in FILTER_TASK_TYPES:
                continue

            video_path = entry['video_path']
                

            # get frames directly from score distribution 
            if sample_method == "possibility" or sample_method == "tstar":
                frame_distribution = entry["frame_distribution"]
                frame_timestamps = np.argsort(frame_distribution)[-frame_num:]
            elif sample_method == "score":
                frame_distribution = entry["score_list"]
                frame_timestamps = np.argsort(frame_distribution)[-frame_num:]

            question = entry['question']
            options = entry['options']
            gt_answer = entry.get(ground_truth_key, "None")

            if video_path in fps_cache:
                fps = fps_cache[video_path]
                
            else:
                try:
                    fps = load_video_fps(video_path)
                    fps_cache[video_path] = fps
                except ValueError as e:
                    logger.error(f"Unable to get fps of {video_path}: {e}")
                    continue

            # get key frames based on frame_key value
            if frame_key == "uniform":
                frames = extract_frames(video_path, None, numframe=frame_num)

            else: 
                frame_timestamps.sort()                    
                pred_frame_nums = [int(ts * fps) for ts in frame_timestamps]                    
                frames = extract_frames(video_path, pred_frame_nums)
                                    

            # intialize qa_results keys
            entry[f"{frame_key}_pred_answer"] = None
            entry["correct"] = None
            frame_distribution = entry.pop("frame_distribution")
            if not frames or len(frames) < 1:
                logger.warning(f"Unable to extract frames for entry {idx}。")
                entry["correct"] = False
                
            # perform QA inference
            else:
                try:
                    pred_answer = tstar_grounder.inference_qa(
                        frames=frames,
                        question=question,
                        options=options,
                        temperature=0.2,
                    )
                    print(f"QA answer for entry {idx}: {pred_answer}")

                    # compare with ground truth answer
                    gt_answer_clean = gt_answer.strip().lower()
                    pred_answer_clean = pred_answer.strip().lower()

                    correct = (pred_answer_clean == gt_answer_clean)
                    entry[f"{frame_key}_pred_answer"] = pred_answer
                    entry["correct"] = correct

                    if correct:
                        correct_count += 1
                    total_count += 1
                except Exception as e:
                    logger.error(f"Error when performing QA for entry {idx}: {e}")
                    entry[f"{frame_key}_pred_answer"] = "QA Error."
                    entry["correct"] = False

        except Exception as e:
            logger.error(f"Error when extracting frames or performing QA for entry {idx}: {e}")
            entry[f"{frame_key}_pred_answer"] = "Processing Error."
            entry["correct"] = False

        qa_results.append(entry)
        if (idx + 1) % 50 == 0 or idx == (len(result_data) - 1):
            with open(output_file, "w", encoding="utf-8") as jsonl_file:
                json.dump(qa_results, jsonl_file, ensure_ascii=False)
                jsonl_file.close()
        
    if total_count == 0:
        logger.warning("No QA evaluations were performed.")
        accuracy = 0.0
    else:
        accuracy = correct_count / total_count

    logger.info(f"QA Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_count})")

    return correct_table, qa_results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--backend', type=str, default="gpt4", help='The backend used for question qa.')
    parser.add_argument('--frame_key', type=str, default="adaptive", help='Frame sampling method.')
    parser.add_argument('--frame_num', type=int, default=4, help='The number of frames fed into qa model.')
    parser.add_argument('--dataset', type=str, default="LongVideoBench", help='The Video QA dataset, currently support LongVideoBench or VideoMME')
    parser.add_argument('--kfs_path', type=str, default="./runs/kfs/kfs_rel0.3_VideoMME.json", help='input kfs json path')
    parser.add_argument('--qa_path', type=str, default="./runs/qa/qa_VideoMME.json", help='output qa json path')
    parser.add_argument('--sample_method', type=str, default="possibility", help="possibility or score")
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(2025)
    args = parse_arguments()

    tstar_grounder = TStarUniversalGrounder(
        backend=args.backend,
        model_path="zdvz",
        num_frames=8
    )

    str_id = str(args.frame_num)
    if args.frame_num == 1:
        str_id = str(4)

    with open(args.kfs_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    correct_table, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        ground_truth_key="answer",
        frame_key=args.frame_key,
        frame_num=args.frame_num,
        sample_method=args.sample_method,
        output_file=args.qa_path
    )

    print(correct_table)      