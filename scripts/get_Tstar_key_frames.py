'''
get_Tstar_key_frames.py

2nd step of the pipeline: perform key frame search using T* on videos based on grounding objects
'''

import sys
import argparse
import json
import numpy as np

sys.path.append('./')
sys.path.append('./YOLO-World/')

# Import custom TStar interfaces
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloInterface
from TStar.TStarFramework import TStarFramework, initialize_yolo  # better to keep interfaces separate for readability
import datetime

FILTER_TASK_TYPES = ['OCR Problems', 'Counting Problem', 'Temporal Perception', 'Information Synopsis', 'Temporal Reasoning']

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

v_p = "/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/videos/BtaVRhoLpC0.mp4"
np.random.seed(2025)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--obj_path', type=str, default="./runs/obj/obj_VideoMME.json", help='The input data path of grounding objects.')
    parser.add_argument('--kfs_path', type=str, default='./runs/kfs/kfs_VideoMME.json', help='Path to save the key frame searching results.')
    
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

    # Initialize Grounder
    grounder = TStarUniversalGrounder(
        backend=args.backend,
        gpt4_model_name="gpt-4o"
    )

    # Initialize YOLO interface
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    results = []

    with open(args.obj_path, 'r', encoding='utf-8') as f_read:
        dataset = json.load(f_read)
            
    for idx, data_item in enumerate(dataset):
        task_type = data_item.get('task_type', ' ')
        
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
            with open(args.kfs_path, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {args.kfs_path}")

if __name__ == "__main__":
    main()
