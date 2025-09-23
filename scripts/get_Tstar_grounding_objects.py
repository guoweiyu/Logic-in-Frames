'''
get_Tstar_grounding_objects.py

1st step of the pipeline: get grounding objects on each video & question pair
'''

import os
import json
import sys
import argparse
import json
import numpy as np

sys.path.append('./')
sys.path.append('./YOLO-World/')
# Import custom TStar interfaces
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloInterface
from TStar.TStarFramework import TStarFramework, initialize_yolo
from utils.data_loader import LVHaystack2TStar_json, VideoMME2TStar_json, LongVideoBench2TStar_json

np.random.seed(2025)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--dataset', type=str, default="LongVideoBench", help='The video dataset used for processing.')
    parser.add_argument('--dataset_meta', type=str, default="LVHaystack/LongVideoHaystack", help='Path to the input JSON file for batch processing.')
    parser.add_argument('--video_root', type=str, default='/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/ego4d/ego4d_data/v1/256p', help='Root directory where the input video files are stored.')
    parser.add_argument('--results_dir', type=str, default='./results/object_grounding/', help='Path to save the batch processing results.')
    
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
    parser.add_argument('--upload_video', type=int, default=1, help='Upload video or not to OpenAI API.')
    return parser.parse_args()

def grounding_objects_onVideo(args, data_item,
        yolo_scorer: YoloInterface,
        grounder: TStarUniversalGrounder,) -> dict:
    """
    grounding objects on a single video.

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
        update_method="spline"
    )

    # Use Grounder to get target and cue objects
    target_objects, cue_objects, relations = TStar_framework.get_grounded_objects(args.prompt_type, args.upload_video)

    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": TStar_framework.results.get('Searching_Objects', []),
    }

    return result

def main():
    """
    Main function to execute TStarSearcher.
    """
    args = parse_arguments()
    
    if args.dataset == "LongVideoBench":
        dataset = LongVideoBench2TStar_json(video_root="/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/videos")
        
    elif args.dataset == "LVHaystack":
        dataset = LVHaystack2TStar_json(video_root=args.video_root)

    
    elif args.dataset == "VideoMME":
        dataset = VideoMME2TStar_json(video_root="/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/Video-MME/videos/data")


    print(len(dataset), "%"*30)

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

    results = []
    output_json = args.results_dir + "obj_IMAGE_ONLY" + args.dataset + "_" + args.backend + "_" + \
        args.prompt_type + "_vid" + str(args.upload_video) + ".json"
    
    for idx, data_item in enumerate(dataset):
        
        print(f"Processing {idx+1}/{len(dataset)}: {data_item['video_id']}")
        # 过滤与subtitle相关的问题
        if "subtitle" in data_item['question']:
            continue

        
        try:
            result = grounding_objects_onVideo(args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)
            print(f"Completed: {data_item['video_id']}\n")

        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            result = {
                "video_id": data_item.get('video_id', ''),
                "grounding_objects": [],
                "answer": data_item.get('answer', ''),
                "error": str(e)
            }
                
        data_item.update(result)        
        results.append(data_item)

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {output_json}")        
        
if __name__ == "__main__":    
    main()
