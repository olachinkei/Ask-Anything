import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from .config import (
    DATA_LIST, JSON_DIR, VIDEO_DIR, NUM_SEGMENTS, RESOLUTION,
    INPUT_MEAN, INPUT_STD
)

class MVBenchDataset(Dataset):
    def __init__(self, data_dir=JSON_DIR, data_list=DATA_LIST, num_segments=NUM_SEGMENTS, resolution=RESOLUTION):
        self.data_list = []
        for k, v in data_list.items():
            json_path = os.path.join(data_dir, v[0])
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': os.path.join(VIDEO_DIR, v[1]),
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform pipeline
        crop_size = resolution
        scale_size = resolution
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(INPUT_MEAN, INPUT_STD) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            res += f"{v} for {k} ({option_list[k]} options => {v/option_list[k]*100:.2f}%)\n"
        return res
    
    def __len__(self):
        return len(self.data_list)
    
    def read_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        total_frame_num = total_frame_num - 1
        uniform_sampled_idx = np.linspace(0, total_frame_num, num=self.num_segments, dtype=int)
        frame_idx = uniform_sampled_idx + 1
        video_data = vr.get_batch(frame_idx).asnumpy()
        return video_data
    
    def read_gif(self, video_path):
        video_data = []
        gif = Image.open(video_path)
        for i in range(gif.n_frames):
            gif.seek(i)
            frame = gif.convert('RGB')
            frame = np.array(frame)
            video_data.append(frame)
        video_data = np.stack(video_data)
        return video_data
    
    def read_frame(self, video_path):
        video_data = []
        for i in range(self.num_segments):
            frame_path = os.path.join(video_path, f"{i}.jpg")
            frame = Image.open(frame_path).convert('RGB')
            frame = np.array(frame)
            video_data.append(frame)
        video_data = np.stack(video_data)
        return video_data
    
    def __getitem__(self, index):
        data = self.data_list[index]
        video_data = self.decord_method[data['data_type']](data['data']['video_path'])
        video_data = self.transform(video_data)
        
        return {
            'video': video_data,
            'question': data['data']['question'],
            'candidates': data['data']['candidates'],
            'answer': data['data']['answer'],
            'task_type': data['task_type']
        } 