import json
import os
from typing import List
from datasets import load_dataset


def LVHaystack2TStar_json(video_root: str) -> List[dict]:
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

    with open("/data/yourname/new-VL-Haystack/VL-Haystack/Datasets/LVHaystack_tiny.json", 'r', encoding='utf-8') as js_file:
        LVHaystact_testset = json.load(js_file)
        
        
    # List to hold the transformed data
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
            # Validate required fields
            if not video_id or not question or not options_str:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            # print("type: ", type(options_str))
            if options_str:
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

def VideoMME2TStar_json(video_root: str) -> List[dict]:
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

    with open('/data/yourname/new-VL-Haystack/VL-Haystack/Datasets/Video-MME/test.json', 'w', encoding='utf-8') as file:
        json.dump(TStar_format_data, file, indent=4, ensure_ascii=False)
        
    return TStar_format_data

def LongVideoBench2TStar_json(video_root: str) -> List[dict]:
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

    with open("/data/yourname/new-VL-Haystack/VL-Haystack/Datasets/LVBench/lvb_val.json", 'r', encoding='utf-8') as file:
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
            
            # Filter out entries where question_category contains 'T'
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
                "position": position,
                "question_category": question_category,
                # "gt_frame_index": gt_frame_index,
            }

            # Add the transformed entry to the result list
            TStar_format_data.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return TStar_format_data
