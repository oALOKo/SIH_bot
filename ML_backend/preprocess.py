import os
import cv2
import torch
import numpy as np
from sklearn.utils import resample
from torch.utils.data import Dataset
from torchvision import transforms

# Parsing the dataset directory
def parse_dataset(base_dir):
    """
    Parse the dataset directory to create a mapping of video paths and their labels.
    
    Args:
        base_dir (str): Base directory containing the train, devel, and test splits.
    
    Returns:
        dict: A dictionary containing train, devel, and test splits with video paths and labels.
    """
    splits = ['train', 'devel', 'test']
    label_map = {'real': 0, 'attack': 1}
    dataset = {split: [] for split in splits}

    for split in splits:
        for label_name, label in label_map.items():
            folder_path = os.path.join(base_dir, split, label_name)
            for file in os.listdir(folder_path):
                if file.endswith(".mov"):
                    video_path = os.path.join(folder_path, file)
                    dataset[split].append((video_path, label))
    
    return dataset

# Transformations for video frames
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Frame preprocessing function
def preprocess_frames(frames):
    """
    Apply transformations to a sequence of frames.

    Args:
        frames (list): A list of frames (images) to preprocess.
    
    Returns:
        torch.Tensor: A tensor of processed frames.
    """
    processed_frames = [frame_transform(frame) for frame in frames]
    return torch.stack(processed_frames).permute(1, 0, 2, 3)

# Balancing the dataset
def balance_dataset(train_data):
    """
    Balance the dataset by oversampling the minority class.

    Args:
        train_data (list): A list of tuples (video_path, label).
    
    Returns:
        list: A balanced dataset with equal instances of each class.
    """
    class0 = [item for item in train_data if item[1] == 0]
    class1 = [item for item in train_data if item[1] == 1]

    class0_oversampled = resample(class0, replace=True, n_samples=len(class1), random_state=42)
    balanced_train_data = class0_oversampled + class1
    np.random.shuffle(balanced_train_data)
    return balanced_train_data

# Custom Video Dataset
class VideoDataset(Dataset):
    """
    Custom dataset for handling video data.
    """
    def __init__(self, data, num_frames=32, transform=None):
        """
        Initialize the dataset.

        Args:
            data (list): A list of tuples (video_path, label).
            num_frames (int): Number of frames to sample from each video.
            transform (callable): Transformation to apply to each frame.
        """
        self.data = data
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def extract_frames(self, video_path):
        """
        Extract frames from a video.

        Args:
            video_path (str): Path to the video file.
        
        Returns:
            torch.Tensor: A tensor of extracted frames.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                break

        cap.release()

        if len(frames) < self.num_frames:
            frames.extend([torch.zeros(3, 112, 112)] * (self.num_frames - len(frames)))  # Padding

        return torch.stack(frames).permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)
