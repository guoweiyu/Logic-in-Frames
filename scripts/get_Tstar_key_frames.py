'''
get_Tstar_key_frames.py

2nd step of the pipeline: perform key frame search using T* on videos based on grounding objects
'''

import pandas as pd
import os
import string
import json
import os
import sys
import cv2
import torch
import copy
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline

# Import custom TStar interfaces
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloInterface
from TStar.interface_searcher import TStarSearcher
from TStar.TStarFramework import TStarFramework, initialize_yolo  # better to keep interfaces separate for readability



import os
import ast
from datasets import load_dataset
from typing import List
import datetime

FILTER_TASK_TYPES = ['OCR Problems', 'Counting Problem', 'Temporal Perception', 'Information Synopsis', 'Temporal Reasoning']

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

v_p = "/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/videos/BtaVRhoLpC0.mp4"
np.random.seed(2025)

def LVHaystack2TStar_json(dataset_meta: str = "LVHaystack/LongVideoHaystack", 
                          video_root: str = "Datasets/Ego4D_videos") -> List[dict]:
    """Load and transform the dataset into the required format for T*.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """
    # Load the dataset from the given source
    dataset = load_dataset(dataset_meta)
    
    # Extract the 'test' split from the dataset
    LVHaystact_testset = dataset["test"]

    # List to hold the transformed data
    TStar_format_data = []

    # Iterate over each row in the dataset
    for idx, entry in enumerate(LVHaystact_testset):
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("video_id")
            question = entry.get("question")
            answer = entry.get("answer", "")

            options_str = entry.get("options", "")
            gt_frame_index = entry.get("frame_indexes", []) #gt frame index for quetion
            position = entry.get("position", [])
            
            # Validate required fields
            if not video_id or not question or not options_str:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            # print("type: ", type(options_str))
            if options_str:
                # options_dict = ast.literal_eval(options_str)
                options_dict = options_str

                # Format the options with letter prefixes (A, B, C, D...)
                options = ""
                for i, (key, value) in enumerate(options_dict.items()):
                    options += f"{key}) {value}\n"

                options = options.rstrip('\n')  # Remove the trailing newline

            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),  # Build the full video path
                "question": question,
                "options": options,
                "answer": answer,
                "gt_frame_index": gt_frame_index,
                "position": position,
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return TStar_format_data

def VideoMME2TStar_json(dataset_meta: str = "LVHaystack/LongVideoHaystack", 
                          video_root: str = "Datasets/Ego4D_videos") -> List[dict]:
    """Load and transform the dataset into the required format for T*.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """

    dataset = load_dataset("lmms-lab/Video-MME")
    VideoMME_testset = dataset["test"]
    TStar_format_data = []

    for idx, entry in enumerate(VideoMME_testset):
        print(entry)
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("videoID")
            question = entry.get("question")
            answer = entry.get("answer", "")
            options_str = entry.get("options", "")
            gt_frame_index = entry.get("frame_indexes", []) #gt frame index for quetion            
            duration = entry.get("duration")
            position = entry.get("position", [])

            # Validate required fields
            if not video_id or not question or not options_str:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_str:

                # Format the options with letter prefixes (A, B, C, D...)
                options = ""
                for option in options_str:
                    options += option[0] + ') ' + option[3: ] + '\n'

                options = options.rstrip('\n')  # Remove the trailing newline

            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),  # Build the full video path
                "question": question,
                "options": options,
                "answer": answer,
                "gt_frame_index": gt_frame_index,
                "duration_group": duration,
                "position": position,
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    # with open('/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/Video-MME/test.json', 'w', encoding='utf-8') as file:
    #     json.dump(TStar_format_data, file, indent=4, ensure_ascii=False)
        
    return TStar_format_data

def LongVideoBench2TStar_json(video_root: str = "/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/videos") -> List[dict]:
    """Load and transform the dataset into the required format for T*.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """

    with open("/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/lvb_val.json", 'r', encoding='utf-8') as file:
        lvb_dataset = json.load(file)

    # List to hold the transformed data
    TStar_format_data = []
    num2letter = ['A', 'B', 'C', 'D', 'E']
    # Iterate over each row in the dataset
    for idx, entry in enumerate(lvb_dataset):
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("video_id")
            video_path = entry.get("video_path")
            question = entry.get("question")
            answer = entry.get("correct_choice", "")
            answer = num2letter[answer]
            question_category = entry.get("question_category")
            duration_group = entry.get("duration_group")
            position = entry.get("position", [])
            
            # filter out subtitle questions based on question category
            if 'T' in question_category:
                continue

            options_list = entry.get("candidates", "")
            
            # gt_frame_index = entry.get("frame_indexes", []) #gt frame index for quetion

            # Validate required fields
            if not video_id or not question or not options_list:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_list:
                options = ""

                # Format the options with letter prefixes (A, B, C, D...)
                for idx in range(len(options_list)):
                    options += num2letter[idx] + ') ' + options_list[idx] + '\n'
                
                options = options.rstrip('\n')  # Remove the trailing newline

            
            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, video_path),  # Build the full video path
                "question": question,
                "options": options,
                "answer": answer,
                "duration_group": duration_group,
                "gt_frame_index": position,
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return TStar_format_data

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--dataset', type=str, default="VideoMME", help='The video dataset used for processing.')
    parser.add_argument('--data_path', type=str, default="obj_VideoMME_gpt4.json", help='The data path.')
    parser.add_argument('--dataset_meta', type=str, default="LVHaystack/LongVideoHaystack", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--video_root', type=str, default='/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/ego4d/ego4d_data/v1/256p', help='Root directory where the input video files are stored.')
    parser.add_argument('--results_dir', type=str, default='./results/', help='Path to save the batch processing results.')
    
    # Common arguments
    parser.add_argument('--config_path', type=str, default="./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=1.0, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    parser.add_argument('--backend', type=str, default='gpt4', help='Backend used for grounding(gpt4 or llava).')
    parser.add_argument('--prompt_type', type=str, default='cot', help='Prompt type used.')
    parser.add_argument('--save_batch', type=int, default=10, help='Save batch results to output_json every N entries.')
    parser.add_argument('--num', type=int, default=100, help='Number of videos to process.')
    parser.add_argument('--upload_video', type=int, default=1, help='Upload video to OpenAI API.')
    parser.add_argument('--update_method', type=str, default='spline', help='Update distribution method.')
    parser.add_argument('--relation_alpha', type=float, default=0.3, help='Weight of relation score.')
    return parser.parse_args()


def process_TStar_onVideo(args, data_item,
        yolo_scorer: YoloInterface,
        grounder: TStarUniversalGrounder,) -> dict:
    """
    Process a single video search and QA.

    Args:
        args (argparse.Namespace): Parsed arguments.
        entry (dict): Dictionary containing 'video_path', 'question', and 'options'.
        yolo_scorer (YoloV5Interface): YOLO interface instance.
        grounder (TStarUniversalGrounder): Universal Grounder instance.

    Returns:
        dict: Results containing 'video_path', 'grounding_objects', 'frame_timestamps', 'answer'.
    """
    # Initialize VideoSearcher
    TStar_framework = TStarFramework(
        grounder=grounder,
        yolo_scorer=yolo_scorer,
        video_path=data_item['video_path'],
        question=data_item['question'],
        options=data_item['options'],
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        device=args.device,
        update_method=args.update_method
    )

    # load objects from json
    target_objects = data_item['grounding_objects']['target_objects']
    cue_objects = data_item['grounding_objects']['cue_objects']
    relations = data_item['grounding_objects'].get('relations', [])

    # Initialize Searching Targets to TStar Seacher
    video_searcher = TStar_framework.set_searching_targets(target_objects, cue_objects, relations)
    video_searcher.relation_alpha = args.relation_alpha

    # Perform search
    all_frames, time_stamps = TStar_framework.perform_search(video_searcher)

    # Output the results
    print("Final Results:")
    print(f"Grounding Objects: {data_item['grounding_objects']}")
    print(f"Frame Timestamps: {TStar_framework.results['timestamps']}")

    # Collect the results
    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": data_item['grounding_objects'],
        "keyframe_timestamps": TStar_framework.results.get('timestamps', []),
        "frame_distribution": video_searcher.P_history[-1],
        "score_list": video_searcher.Score_history[-1],
        "num_iterations": TStar_framework.results.get('num_iterations', 0)
    }

    return result


def main():
    """
    Main function to execute TStar key frame search.
    """
    args = parse_arguments()    

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Grounder
    grounder = TStarUniversalGrounder(
        backend=args.backend,
        gpt4_model_name="gpt-4o",
        model_path="/data/guoweiyu/new-VL-Haystack/VL-Haystack/LLaVA-NeXT/llava-onevision-qwen2-7b-ov"
    )

    # Initialize YOLO interface
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    
    input_json = '/data/guoweiyu/new-VL-Haystack/VL-Haystack/results/rebuttal/2025-07-29-final_qa_results/2025-07-31-01-17-21qa_8frames_llava_score_LongVideoBench_adaptive.json'

    results = []
    
    output_json = '/data/guoweiyu/new-VL-Haystack/VL-Haystack/results/kfs_32frames_LongVideoBench_for_test.json'

    dataset = []
    with open(input_json, 'r', encoding='utf-8') as f_read:
        dataset = json.load(f_read)
            
    for idx, data_item in enumerate(dataset):
        task_type = data_item.get('task_type', ' ')
        
        if data_item['video_id'] != "Ip9DbdOtqF4":
            continue

        if task_type in FILTER_TASK_TYPES:
            continue

        try:
            result = process_TStar_onVideo(args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)            
            print(f"Completed: {data_item['video_id']}\n")

        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            result = {
                "video_id": data_item.get('video_id', ''),
                "grounding_objects": [],
                "keyframe_timestamps": [],
                "answer": "",
                "error": str(e)
            }
            
        data_item.update(result)
        results.append(data_item)
        if (idx + 1) % args.save_batch == 0 or (idx + 1) == len(dataset):
            # Save batch results to output_json
            with open(output_json, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {output_json}")

if __name__ == "__main__":
    main()
