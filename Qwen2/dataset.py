import os
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchaudio
from transformers import AutoProcessor
import torch.nn.functional as F

class Qwen2AudioDataset(Dataset):
    def __init__(self, prefix, ann_path, processor_path, task=None):
        """
        Dataset for Qwen2-Audio tasks like ASR and AAC.

        Args:
            prefix (str): Base path for locating audio files.
            ann_path (str): Path to the annotation JSON file.
            processor_path (str): Path to the pretrained Qwen2-Audio processor.
            task (str, optional): Task type (e.g., "asr" or "aac").
        """
        self.prefix = prefix
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.task = task

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        """
        Collate function to combine a batch of samples into model inputs.
        """
        # Find the maximum length of raw_wav in the batch
        max_length = max(len(s["raw_wav"]) for s in samples)

        entity = {
            "testset_id": [s["testset_id"] for s in samples],
            "task": [s["task"] for s in samples],
            "Q": [s["Q"] for s in samples],
            "id": [s["id"] for s in samples],
            "raw_wav": [s["raw_wav"] for s in samples],
        }

        # Add "text" only if it exists (e.g., in training mode)
        if self.task is not None and any("text" in s for s in samples):
            entity["text"] = [s.get("text", "") for s in samples]  # Default to empty string if "text" is missing

        return entity

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = os.path.join(self.prefix, ann["path"])

        # Load audio
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Failed to load {audio_path}. Error: {e}. Using silent audio as fallback.")
            audio = torch.zeros(16000)  # 1 second of silent audio at 16 kHz
            sr = 16000

        # Convert stereo to mono if necessary
        if audio.ndim == 2:
            audio = audio.mean(dim=0)

        # Validate audio length
        if audio.size(0) == 0:
            print(f"Warning: Audio at {audio_path} is empty. Using silent audio as fallback.")
            audio = torch.zeros(16000)  # 1 second of silent audio at 16 kHz

        # Pad or truncate audio to at least 1 second and at most 30 seconds
        if audio.size(0) < sr:
            padding = torch.zeros(sr - audio.size(0))
            audio = torch.cat((audio, padding), dim=0)
        audio = audio[: sr * 30]  # Max duration: 30 seconds

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        else:
            audio = audio.clone().detach()

        entity = {
            "testset_id": ann["testset_id"],
            "raw_wav": audio,  # Convert to numpy for compatibility with processor
            "task": ann.get("task", "asr"),
            "Q": ann.get("Q", ""),
            "id": ann["path"],
        }

        # Add "text" field only if it exists and task is provided
        if self.task is not None and "text" in ann:
            entity["text"] = ann["text"]

        return entity