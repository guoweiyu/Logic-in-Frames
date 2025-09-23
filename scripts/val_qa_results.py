import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
from TStar.interface_llm import TStarUniversalGrounder
import argparse
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


FILTER_TASK_TYPES = ['OCR Problems', 'Counting Problem', 'Temporal Perception', 'Information Synopsis', 'Temporal Reasoning']

# 配置日志
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
    获取视频的帧率（FPS）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        raise ValueError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        logger.error(f"无法获取视频帧率（FPS）: {video_path}")
        raise ValueError(f"无法获取视频帧率（FPS）: {video_path}")
    logger.debug(f"视频 {video_path} 的 FPS: {fps}")
    return fps

def extract_frames(video_path: str, frame_indices: List[int] = None, numframe: int = 16) -> List[Optional[Image.Image]]:
    """
    从视频中提取指定的帧，并转换为 PIL 图像。如果没有提供帧索引，则均匀地采样指定数量的帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数

    # 如果没有提供 frame_indices，进行均匀采样
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
            frames.append(None)  # 使用 None 表示无法读取帧

    cap.release()
    return frames

def compute_qa_accuracy(
    result_data: List[Dict[str, Any]],
    tstar_grounder: TStarUniversalGrounder,
    frame_key: str ="uniform",
    ground_truth_key: str = "answer",
    max_workers: int = 4,
    output_file: str = "Rebuttal/qa_results.jsonl",
    frame_num: int = 4,
    sample_method: str="possibility"
) -> Tuple[Dict, List[Dict[str, Any]]]:
    """
    处理 result_data，执行 QA 推理，并计算 QA 准确率，动态保存每个条目的 QA 结果为 JSONL 格式。
    """
    qa_results = []
    correct_count = 0
    total_count = 0
    correct_table = {15:{'correct_count': 0, 'total_count' : 0}, 
                         60:{'correct_count': 0, 'total_count' : 0},
                         600:{'correct_count': 0, 'total_count' : 0},
                         3600:{'correct_count': 0, 'total_count' : 0}}

    # 缓存 FPS 以避免重复加载
    fps_cache = {}
          
    # 分类别统计准确率
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
                    logger.error(f"获取视频 {video_path} 的 FPS 失败: {e}")
                    continue

            # get key frames based on frame_key value
            if frame_key == "uniform":
                frames = extract_frames(video_path, None, numframe=frame_num)

            else: 
                frame_timestamps.sort()                    
                pred_frame_nums = [int(ts * fps) for ts in frame_timestamps]                    
                frames = extract_frames(video_path, pred_frame_nums)
                                    

            # 初始化 qa_results 条目
            entry[f"{frame_key}_pred_answer"] = None
            entry["correct"] = None
            frame_distribution = entry.pop("frame_distribution")
            if not frames or len(frames) < 1:
                logger.warning(f"无法提取帧用于条目 {idx}。")
                entry["correct"] = False
                
                # 执行 QA 推理
            else:
                try:
                    # 使用预测帧执行 QA 推理 
                    pred_answer = tstar_grounder.inference_qa(
                        frames=frames,
                        question=question,
                        options=options,
                        temperature=0.2,
                    )
                    print(f"条目 {idx} 的 QA 答案: {pred_answer}")

                    # 比较预测答案与真实答案（忽略大小写和前后空格）
                    gt_answer_clean = gt_answer.strip().lower()
                    pred_answer_clean = pred_answer.strip().lower()

                    correct = (pred_answer_clean == gt_answer_clean)
                    entry[f"{frame_key}_pred_answer"] = pred_answer
                    entry["correct"] = correct

                    if correct:
                        correct_count += 1
                    total_count += 1
                except Exception as e:
                    logger.error(f"条目 {idx} 的 QA 推理失败: {e}")
                    entry[f"{frame_key}_pred_answer"] = "QA 推理失败。"
                    entry["correct"] = False

        except Exception as e:
            logger.error(f"提取帧或执行 QA 推理时发生错误 for 条目 {idx}: {e}")
            entry[f"{frame_key}_pred_answer"] = "处理失败。"
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
    parser.add_argument('--json_file', type=str, default="/data/guoweiyu/new-VL-Haystack/VL-Haystack/results/rebuttal/2025-07-26-07-34-34kfs_4frames_0_rel0.3_LongVideoBench_gpt4_cot_0_vid1.json", help='The video dataset used for processing.')
    parser.add_argument('--frame_key', type=str, default="adaptive", help='Frame sampling method.')
    parser.add_argument('--frame_num', type=int, default=4, help='The number of frames fed into qa model.')
    parser.add_argument('--dataset', type=str, default="LongVideoBench", help='The Video QA dataset, currently support LongVideoBench or VideoMME')
    parser.add_argument('--data_path', type=str, default="/data/guoweiyu/new-VL-Haystack/VL-Haystack/results/rebuttal/final_kfs/kfs_32_frames_rel0.3_VideoMME.json", help='input kfs json path')
    parser.add_argument('--sample_method', type=str, default="possibility", help="possibility or score")
    return parser.parse_args()

if __name__ == "__main__":
    
    np.random.seed(2025)
    args = parse_arguments()
    # 初始化 TStarUniversalGrounder
    tstar_grounder = TStarUniversalGrounder(
        # backend="gpt4",
        backend=args.backend,
        model_path="zdvz",
        num_frames=8
    )

    str_id = str(args.frame_num)
    if args.frame_num == 1:
        str_id = str(4)

    data_json_path = args.data_path
    with open(data_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    output_root = "./results/rebuttal/2025-07-29-final_qa_results"
    # 计算 QA 准确率
    correct_table, qa_results = compute_qa_accuracy(
        result_data=result_data,
        tstar_grounder=tstar_grounder,
        ground_truth_key="answer",
        frame_key=args.frame_key,
        frame_num=args.frame_num,
        sample_method=args.sample_method,
        max_workers=1,
        output_file=os.path.join(output_root, str(nowTime) + "qa_" + str(args.frame_num) + "frames_" + args.backend + "_" + args.sample_method + "_" + args.dataset + "_" + args.frame_key +".json")
    )

    print(correct_table)
      