import os
from typing import Dict, Optional, List
import re
import sys
sys.path.append("./LLaVA-NeXT")
sys.path.append("./VSLS")
import openai
from typing import List, Dict
from PIL import Image
import numpy as np
import cv2
from utilites import *
import requests
import json
import warnings
warnings.filterwarnings("ignore")


def load_video(self, video_path, max_frames_num,fps=1,force_sample=False):
    # 空帧处理
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = cv2.VideoReader(video_path, ctx=cpu(0),num_threads=1) # 加载视频
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()  # 总帧数/平均帧率=视频时长
    fps = round(vr.get_avg_fps()/fps) 
    frame_idx = [i for i in range(0, len(vr), fps)] # 根据fps参数设置采样间隔，获得采样帧索引
    frame_time = [i/fps for i in frame_idx] # 获得对应时间戳
    if len(frame_idx) > max_frames_num or force_sample: # 强制均匀采样
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int) # 生成均匀分布的帧索引
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time]) # 将时间戳格式化为字符串形式，比如“0.00s，1.00s”
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    # spare_frames形状为(采样帧数,336,336,3)，后面为分辨率，frame_time为帧时间字符串
    return spare_frames,frame_time,video_time # 返回采样帧，帧时间戳以及视频总时长

class LocalVLMInterface:
    def __init__(self, model_name: str="default", base_url: str="yourl api url"):
        """
        deploy VLM locally
        access via RESTful API        
        """
        self.model_name = model_name
        self.client = openai.Client(api_key="None", base_url=base_url)

    def inference_text_only(
        self, 
        query: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        inference only with text input
        """
        try:
            stream_request = self.client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query,
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            stream_response = ""

            for chunk in stream_request:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    stream_response += content

            print(stream_response)
            print("-" * 30)
                
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames_all_in_one(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> str:
        try:
            # resize frame to (224, 224)
            inputs = [{"type": "text", "text": query}]
            print("len(frames)", len(frames))
            for i, frame in enumerate(frames):
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            },
                        "modalities": "multi-images",
                    }
                # Adding visual context (images) to messages if supported by the model
                inputs.append(visual_context)
                
            response = self.client.chat.completions.create(
                # model="InternVL2_5-78B",
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": inputs,
                    },
                ],
                temperature=0.2,
            )
            print("\nResponse:")
            print(response.choices[0].message.content)
            return response.choices[0].message.content
             
        except Exception as e:
            print(f"Error in inference_with_frames_all_in_one: {str(e)}")
            return f"Error: {str(e)}"

    def inference_qa(
        self,
        question: str,
        options: str,
        frames: List[Image.Image] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        try:
        # 构建查询
            query = f"Question: {question}\nOptions: {options}\nAnswer with the letter corresponding to the best choice."

            inputs = [{"type": "text", "text": query}]
            print("len(frames)", len(frames))
            for i, frame in enumerate(frames):
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            }
                    }
                # Adding visual context (images) to messages if supported by the model
                inputs.append(visual_context)
                
            response = self.client.chat.completions.create(
                model="default",
                messages=[
                        {
                            "role": "user",
                            "content": inputs,
                        },
                ],
                temperature=temperature,
            )
            print("\nResponse:")
            print(response)
            print(response.choices[0].message.content)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in inference_qa: {str(e)}")
            return f"Error: {str(e)}"

class GPT4Interface:
    def __init__(self,model="gpt-4", api_key=None):
        """
        Initialize the GPT-4 API client.

        Reads the OpenAI API key from the environment variable `OPENAI_API_KEY`.
        """
        self.url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
        self.api_key = api_key
        self.model_name = model
        if api_key==None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
    
        self.headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {self.api_key}" #Please change your KEY. If your key is XXX, the Authorization is "Authorization": "Bearer XXX"
        }

        openai.api_key = self.api_key

    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Perform inference using the GPT-4 API.

        Args:
            query (str): User's query or input.
            system_message (str): System message to guide the model's behavior.
            temperature (float): Sampling temperature for the response.
            max_tokens (int): Maximum number of tokens for the response.

        Returns:
            str: The response generated by the GPT-4 model.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
        data = { 
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.7 
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response = response.json()
            print(response)

            
            return response['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def inference_with_frames_all_in_one(self, query: str, frames: List[Image.Image], system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Perform inference using the GPT-4 API with video frames as context.

        Args:
            query (str): User's query or input.
            frames (List[Image.Image]): List of PIL.Image objects to provide visual context.
            system_message (str): System message to guide the model's behavior.
            temperature (float): Sampling temperature for the response.
            max_tokens (int): Maximum number of tokens for the response.

        Returns:
            str: The response generated by the GPT-4 model.
        """

        # Messages format
        inputs = [{"type": "text", "text": query}]

        # Encode frames as Base64 strings
        for i, frame in enumerate(frames):
            try:
                # Convert PIL Image to Base64 string
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                            }
                    }
                # Adding visual context (images) to messages if supported by the model
                inputs.append(visual_context)

            except Exception as e:
                return f"Error encoding frame {i}: {str(e)}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": inputs},
        ]
        data = { 
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.1 
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            print(response.json())
            response = response.json()
            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            return f"Error: {str(e)}"

    def inference_qa(self, question: str, options: str, frames: List[Image.Image] = None, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Perform inference for a multiple-choice question with optional visual frames as context.

        Args:
            question (str): The question to answer.
            options (str): Multiple-choice options formatted as a string.
            frames (List[Image.Image], optional): List of PIL.Image objects to provide additional visual context.
            system_message (str): System message to guide the model's behavior.
            temperature (float): Sampling temperature for the response.
            max_tokens (int): Maximum number of tokens for the response.

        Returns:
            str: The selected option or answer.
        """
        # Construct query
        query = f"Question: {question}\nOptions: {options}\nAnswer with the letter corresponding to the best choice."

        # Messages format
        inputs = [{"type": "text", "text": query}]

        if frames:
            print("frames: ", frames)
            # Encode frames as Base64 strings
            for i, frame in enumerate(frames):
                frame.save(f"frame_{i}.jpg")
                try:
                    frame_base64 = encode_image_to_base64(frame)
                    visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                        }
                    }
                    # Adding visual context (images) to messages if supported by the model
                    inputs.append(visual_context)

                except Exception as e:
                    return f"Error encoding frame {i}: {str(e)}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": inputs},
        ]
        data = { 
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.1 
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            print(response.json())
            response = response.json()

            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"

class VSLSUniversalGrounder:
    """
        A universal interface for different VLM backends to perform video question grounding.
    """
    def __init__(
        self,
        backend: str = "llava",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        gpt4_model_name: str = "gpt-4o",
        gpt4_api_key: Optional[str] = None,
        num_frames: Optional[int] = 8,
    ):
        """
        backend: local vlm or gpt4
        model_name & base_url: local vlm
        gpt4_model_name, gpt4_api_key: GPT4 model name & API Key
        """
        self.backend = backend.lower()
        self.num_frames = num_frames
        if self.backend == "llava" or self.backend == "internvl" or self.backend == "qwenvl":
            if not model_name:
                raise ValueError("Please provide certain model_name")
            self.VLM_model_interfance = LocalVLMInterface(model_name=model_name, base_url=base_url)

        elif self.backend == "gpt4":
            self.VLM_model_interfance = GPT4Interface(model=gpt4_model_name, api_key=None)

        else:
            raise ValueError("backend must be either 'llava', 'internvl', 'qwenvl' or 'gpt4'.")

    def inference_query_grounding(
        self,
        video_path: str,
        question: str,
        upload_video: bool = True,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Dict[str, List[str]]:
        """
        identify target_objects and cue_objects that help answer the question.
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames)
        # 构建 prompt
        if self.backend == "llava":            
            system_prompt = (
            "Analyze the following video frames and question:\n"
            f"Question: {question}\n"
            )
            
        elif self.backend == "gpt4":
            system_prompt = (
                "Here is a video:\n"
                + "\n".join(["<image>"] * len(frames))  
                + "\nHere is a question about the video:\n"
                f"Question: {question}\n"
            )

        else:
            raise ValueError("backend must be either 'llava' or 'gpt4'.")
        
        
        print("???", len(frames))
            
        if options:
            system_prompt += f"Options: {options}\n"
            
        system_prompt += (
            "\nWhen answering this question about the video:\n"
            "1. What key objects to locate the answer?\n"
            "   - List potential key objects (short sentences, separated by commas).\n"
            "2. What cue objects might be near the key objects and might appear in the scenes?\n"
            "   - List potential cue objects (short sentences, separated by commas).\n\n"
            "Please provide your answer in two lines, directly listing the key and cue objects based on the question and options provided, separated by commas.\n"
            "Your response format should be strictly like this in two lines:\n"
            "Key Objects:object1,object2,object3\n"
            "Cue Objects:object1,object2,object3"
        )

        if upload_video:
            response = self.VLM_model_interfance.inference_with_frames_all_in_one(
                query=system_prompt,
                frames=frames,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.VLM_model_interfance.inference_text_only(
                query=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # parse response based on expected format
        lines = response.split("\n")
        if len(lines) < 2:
            raise ValueError("Unexpected response format. Could not extract objects.")

        target_objects = self.parse_objects(lines[0])
        cue_objects = self.parse_objects(lines[1])
        return target_objects, cue_objects
        
    def inference_query_grounding2(
        self,
        video_path: str,
        question: str,
        upload_video: bool = True,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Dict[str, List[str]]:
        """
        identify target_objects and cue_objects that help answer the question.
        """

        if upload_video:
            frames = load_video_frames(video_path=video_path, num_frames=self.num_frames)
        
        else:
            system_prompt = (
                "Here is a question:\n"
                +f"Question: {question}\n"
            )        
        
        if self.backend == "llava" or self.backend == "internvl":
            system_prompt = (
            "Analyze the following video frames, question and options:\n"
            f"Question: {question}\n"
            )
            
        elif self.backend == "gpt4":
            system_prompt = (
                "Here is a video:\n"
                + "\n".join(["<image>"] * len(frames))  
                + "\nHere is a question about the video:\n"
                f"Question: {question}\n"
            )

        else:
            raise ValueError("backend must be either 'llava' or 'gpt4'.")

        if options:
            system_prompt += f"Options: {options}\n"

        system_prompt += ( # Rebuttal Change
                        #   Extract 3-5 core objects detectable by computer vision
                        #   List 2-4 scene elements that help locate key objects based on options provided
                        #   • Condition: Both objects in each relationship must be present in the extracted Key Objects and Cue Objects.
            """Step 1: Key Object Identification

                • Extract 5-8 core objects detectable by computer vision

                • Use YOLO-compatible noun phrases (e.g., “person”, “mic”)

                • Format: Key Objects: obj1, obj2, obj3

                Step 2: Contextual Cues

                • List 3-5 scene elements that help locate key objects based on options provided

                • Use YOLO-compatible detectable noun phrases (avoid abstract concepts)

                • Format: Cue Objects: cue1, cue2, cue3
                
                Step 3: Relationship Triplets​

                • Relationship types:
                    •	Spatial: Objects must appear in the same frame
                    •	Attribute: Color/size/material descriptions (e.g., “red clothes”, “large”)
                    •	Time: Appear in different frames within a few seconds
                    •	Causal: There is a temporal order between the objects
                
                • Condition: Both objects in each relationship must be present in the extracted Key Objects and Cue Objects.

                • Format: Rel: (object, relation_type, object), relation_type should be exactly one of spatial/attribute/time/causal

                Output Rules
                    1.	One line each for Key Objects/Cue Objects/Rel starting with exact prefixes
                    2.	Separate items with comma except for triplets where items are separated by semicolon
                    3.	Never use markdown or natural language explanations
                    4.  If you cannot identify any key objects or cue objects from the video provided, please just identify the possible key or cue objects from the question and options provided 
                
                Below is an example of the procedure:
                    Question: For “When does the person in red clothes appear with the dog?”
                    Response:
                        Key Objects: person, dog, red clothes
                        Cue Objects: grassy_area, leash, fence
                        Rel: (person; attribute; red clothes), (person; spatial; dog)

                Format your response EXACTLY like this in three lines:
                        Key Objects: object1, object2, object3
                        Cue Objects: object1, object2, object3
                        Rel: (object1; relation_type1; object2), (object3; relation_type2; object4)
            """
            )
        
        if upload_video:
            response = self.VLM_model_interfance.inference_with_frames_all_in_one(
                query=system_prompt,
                frames=frames,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.VLM_model_interfance.inference_text_only(
                query=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        print("response: ", response)

        lines = response        
        lines = re.sub(r'\n+', '\\n', lines)
        lines = re.sub(r'\n$', '', lines)
        lines = re.sub(r'^\n', '', lines)
        
        start_pos = lines.find("Key Objects: ")
        Rel_pos = lines.find("Rel: ")
        end_pos = lines.find('\n', Rel_pos)
        if end_pos == -1:
            end_pos = len(lines)
        lines = lines[start_pos:end_pos]

        lines = lines.split("\n")
        if len(lines) < 2:
            raise ValueError(f"Unexpected response format from inference_query_grounding() --> {response}.")

        target_objects = self.parse_objects(lines[0])
        cue_objects = self.parse_objects(lines[1])
        relations = self.parse_relations(lines[2])
        print(relations)
                
        return target_objects, cue_objects, relations

    def parse_objects(self, objects: str):
        object_list = objects.split(",")
        object_list[0] = object_list[0].replace("Key Objects:", "")
        object_list[0] = object_list[0].replace("Cue Objects:", "")
        for idx in range(len(object_list)):
            object_list[idx] = object_list[idx].strip()   # strip head and tail spaces

        return object_list
    
    def parse_relations(self, rels: str):
        rel_list = rels.split(",")
        rel_list[0] = rel_list[0].replace("Rel:", "")
        for idx in range(len(rel_list)):
            rel_list[idx] = rel_list[idx].strip()

        return_list = []
        for relation in rel_list:
            relation_break_down = relation.split(';')
            obj1 = relation_break_down[0].replace('(', "").strip()
            obj2 = relation_break_down[2].replace(')', "").strip()
            rel_type = relation_break_down[1].strip()
            return_list.append((obj1, obj2, rel_type))

        return return_list        

    def inference_qa(
        self,
        frames: List[Image.Image],
        question: str,
        options: str,
        temperature: float = 0.2,
        max_tokens: int = 128,
        video_time: float = 50,
        frame_timestamps: List = [1, 2, 3, 4, 5, 6, 7, 8]
    ) -> str:
        """
        answer multiple choice questions
        return the most possible option 'A', 'B', 'C' or 'D'
        """
        if self.backend == "gpt4":
            system_prompt = (
                "Select the best answer to the following multiple-choice question based on the video.\n"
                + "\n".join(["<image>"] * len(frames))  
                + f"\nQuestion: {question}\n"
                + f"Options: {options}\n"
                + "Answer with the option’s letter from the given choices directly.\n"
                + "Your response format should be strictly an upper case letter A,B,C,D or E.\n"
            )
            print("system_prompt:\n",system_prompt)

        elif self.backend == "llava" or self.backend == "internvl" or self.backend == "qwenvl":
            system_prompt = (
                "Select the best answer to the following multiple-choice question based on the video.\n"
                + "\n".join(["<image>"] * len(frames))  
                + f"\nQuestion: {question}\n"
                + f"Options: {options}\n"
                + "Answer with the option’s letter from the given choices directly.\n"
                + "Your response format should be strictly an upper case letter A,B,C,D or E.\n"
            )
            print("system_prompt:\n",system_prompt)

        response = self.VLM_model_interfance.inference_with_frames_all_in_one(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=30
        )
        print(response)
        return response.strip()
    
if __name__ == "__main__":
    """
    test example
    """
    
    llava_interface = LocalVLMInterface(
        model_name="default",
        base_url="your url service"
    )
    llava_interface.inference_text_only("What is the color of the sky?")

    print("=== Using internvl backend ===")
    llava_grounder = VSLSUniversalGrounder(
        backend="internvl",
        model_path="/data/yourname/new-VL-Haystack/VL-Haystack/LLaVA-NeXT/llava-onevision-qwen2-7b-ov"
    )
    
    question_mc = "In a room with a wall tiger and a map on the wall, there is a man wearing a white shirt. What is he doing?",
    options_mc = "A) drinking water\nB) playing with a cell phone\nC) speaking\nD) dancing",

    answer_llava = llava_grounder.inference_query_grounding2("/data/yourname/new-VL-Haystack/VL-Haystack/Datasets/LVBench/videos/_1kZe-2kiuQ.mp4", question_mc, options_mc)
    print("Internvl Grounding Answer:", answer_llava)

