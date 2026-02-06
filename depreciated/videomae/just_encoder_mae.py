import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor

model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
model.eval()

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Example: video as list of PIL Images or numpy arrays
# For demo, create dummy video (16 frames, 224x224, RGB)
from PIL import Image
import numpy as np

dummy_frames = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(16)]
video = dummy_frames

inputs = processor(video, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

cls_embedding = embeddings[:, 0]
patch_embeddings = embeddings[:, 1:]
clip_embedding = patch_embeddings.mean(dim=1)

print(f"Input video: {len(video)} frames")
print(f"Embeddings shape: {embeddings.shape}")
print(f"CLS embedding shape: {cls_embedding.shape}")
print(f"Patch embeddings shape: {patch_embeddings.shape}")
print(f"Clip embedding shape: {clip_embedding.shape}")
print(f"Hidden dimension: {embeddings.shape[-1]}")

