import json
import os
from typing import List
from datasets import load_dataset
import argparse # 1. 导入 argparse 库

# --- 数据集处理函数 (保持不变，但会从main函数调用) ---
# 用于将数据集处理成json，方便后面调用

# def LVHaystack2TStar_json(input_path: str, video_root: str) -> List[dict]:
def LVHaystack2TStar_json(video_root: str, 
                          input_path: str = "/data/guoweiyu/new-VL-Haystack/VL-Haystack/Datasets/LVBench/lvb_val.json") -> List[dict]:
    """Load and transform the LVHaystack dataset."""
    print(f"Processing LVHaystack from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as js_file:
        LVHaystact_testset = json.load(js_file)
        
    TStar_format_data = []
    
    video_ids = LVHaystact_testset['video_id']
    questions = LVHaystact_testset['question']
    answers = LVHaystact_testset['answer']
    options_strs = LVHaystact_testset['options']
    gt_frame_indexes = LVHaystact_testset['frame_indexes'] 
            
    for idx in range(len(video_ids)):
        try:
            video_id = video_ids[idx]
            question = questions[idx]
            answer = answers[idx]
            options_str = options_strs[idx]
            gt_frame_index = gt_frame_indexes[idx]
            position = []
            if not video_id or not question or not options_str:
                continue

            options = ""
            for i, (key, value) in enumerate(options_str.items()):
                options += f"{key}) {value}\n"
            options = options.rstrip('\n')

            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),
                "question": question,
                "options": options,
                "answer": answer,
                "gt_frame_index": gt_frame_index,
                "position": position,
            }
            TStar_format_data.append(transformed_entry)
        except Exception as e:
            print(f"Error processing LVHaystack entry {idx+1}: {str(e)}")

    return TStar_format_data

def VideoMME2TStar_json(video_root: str) -> List[dict]:
    """Load and transform the Video-MME dataset from Hugging Face."""
    print("Processing Video-MME from Hugging Face hub...")
    dataset = load_dataset("lmms-lab/Video-MME")
    VideoMME_testset = dataset["test"]
    TStar_format_data = []

    for idx, entry in enumerate(VideoMME_testset):
        try:
            video_id = entry.get("videoID")
            question = entry.get("question")
            answer = entry.get("answer", "")
            options_str = entry.get("options", "")
            duration = entry.get("duration")
            position = entry.get("position", [])

            if not video_id or not question or not options_str:
                continue
            
            options = ""
            for option in options_str:
                options += option[0] + ') ' + option[3:] + '\n'
            options = options.rstrip('\n')

            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),
                "question": question,
                "options": options,
                "answer": answer,
                "duration_group": duration,
                "position": position,
            }
            TStar_format_data.append(transformed_entry)
        except Exception as e:
            print(f"Error processing Video-MME entry {idx+1}: {str(e)}")
        
    return TStar_format_data

def LongVideoBench2TStar_json(input_path: str, video_root: str) -> List[dict]:
    """Load and transform the LongVideoBench dataset."""
    print(f"Processing LongVideoBench from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as file:
        lvb_dataset = json.load(file)

    TStar_format_data = []
    num2letter = ['A', 'B', 'C', 'D', 'E']

    for idx, entry in enumerate(lvb_dataset):
        try:
            video_id = entry.get("video_id")
            video_path_suffix = entry.get("video_path")
            question = entry.get("question")
            answer_idx = entry.get("correct_choice", "")
            answer = num2letter[answer_idx]
            duration_group = entry.get("duration_group")
            position = entry.get("position", [])
            
            if not video_id or not question:
                continue

            options_list = entry.get("candidates", [])
            options = ""
            for i in range(len(options_list)):
                options += num2letter[i] + ') ' + options_list[i] + '\n'
            options = options.rstrip('\n')

            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, video_path_suffix),
                "question": question,
                "options": options,
                "answer": answer,
                "duration_group": duration_group,
                "position": position,
            }
            TStar_format_data.append(transformed_entry)
        except Exception as e:
            print(f"Error processing LongVideoBench entry {idx+1}: {str(e)}")

    return TStar_format_data

# --- 2. 新增的主执行逻辑 ---

def main():
    """
    主函数，用于解析命令行参数并调用相应的数据处理函数。
    """
    parser = argparse.ArgumentParser(description="Process video QA datasets into a unified format.")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=['longvideobench', 'videomme', 'lvhaystack'],
        help="The type of dataset to process."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="The directory where the output JSON file will be saved."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="The root directory where video files are stored."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="The path to the input JSON file. (Not required for 'videomme')"
    )
    
    args = parser.parse_args()

    # 根据数据集选择调用哪个函数
    processed_data = []
    if args.dataset == 'longvideobench':
        if not args.input_file:
            raise ValueError("--input_file is required for 'longvideobench'")
        processed_data = LongVideoBench2TStar_json(input_path=args.input_file, video_root=args.video_root)
    
    elif args.dataset == 'videomme':
        # VideoMME 从HuggingFace加载，不需要输入文件
        processed_data = VideoMME2TStar_json(video_root=args.video_root)
        
    elif args.dataset == 'lvhaystack':
        if not args.input_file:
            raise ValueError("--input_file is required for 'lvhaystack'")
        processed_data = LVHaystack2TStar_json(input_path=args.input_file, video_root=args.video_root)

    # 统一的文件保存逻辑
    if processed_data:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 根据数据集命名输出文件
        output_filename = f"{args.dataset}_processed.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"\nSaving {len(processed_data)} processed entries to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, indent=4, ensure_ascii=False)
        print("Done!")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()
    
    
    
