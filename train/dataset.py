from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import os
import json
class SimpleDataset(Dataset):
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.dataset = self._get_dataset()
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )

    def _get_dataset(self):
        args = self.args
        if not args.train_data_json or not os.path.isdir(args.train_data_json):
            raise ValueError("`train_data_json` must be provided and be a valid directory path containing 'self_bench.txt'.")
        
        data_dir = args.train_data_json
        bench_file = os.path.join(data_dir, 'self_bench.txt')
        if not os.path.exists(bench_file):
            raise FileNotFoundError(f"'{bench_file}' not found in the provided data directory.")

        samples = []
        with open(bench_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                num, text = line.split('-', 1)
                samples.append({'num': num, 'text': text, 'data_dir': data_dir})
        
        with self.accelerator.main_process_first():
            if args.seed is not None:
                random.Random(args.seed).shuffle(samples)
            else:
                random.shuffle(samples)

            if args.max_train_samples is not None:
                samples = samples[:args.max_train_samples]
        
        return samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        
        try:
            num = example['num']
            data_dir = example['data_dir']
            
            source_path = os.path.join(data_dir, f"test{num}_source.png")
            mask_path = os.path.join(data_dir, f"test{num}_mask.png")
            ref_path = os.path.join(data_dir, f"test{num}_ref.png")
            
            source_pil = Image.open(source_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
            ref_pil = Image.open(ref_path).convert("RGB")

            prompt = f"The text is '{example['text'].strip()}'"

            ground_truth_tensor = self.image_transforms(ref_pil)
            source_tensor = self.image_transforms(source_pil)
            
            mask_array = np.array(mask_pil)
            if np.sum(mask_array) == 0:
                bbox_coords = (0, 0, ref_pil.width, ref_pil.height)
            else:
                rows = np.any(mask_array > 0, axis=1)
                cols = np.any(mask_array > 0, axis=0)
                if not (np.any(rows) and np.any(cols)):
                    raise ValueError("Mask is not empty but bounding box could not be determined.")
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox_coords = (xmin, ymin, xmax, ymax)
            
            style_image = ref_pil.crop(bbox_coords)

            mask_tensor = self.mask_transforms(mask_pil)
            mask_tensor = (mask_tensor > 0.5).float()
            
        except Exception as e:
            print(f"Error loading sample {index} (num: {example.get('num', 'N/A')}): {e}. Skipping and trying another.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        drop_image_embed = 1 if random.random() < 0.05 else 0

        return {
            "pixel_values": ground_truth_tensor,
            "source_image": source_tensor,
            "prompts": prompt,
            "style_images": style_image,
            "drop_image_embeds": drop_image_embed,
            "mask": mask_tensor,
        }

class LTB_Dataset(Dataset):
    """LongText-Bench dataset loader."""

    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.dataset = self._load_data()
        self.transform = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _load_data(self):
        path = Path(self.args.train_data_json)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append({
                    "prompt": data["prompt"],
                    "image_path": data["image_path"],
                })

        with self.accelerator.main_process_first():
            if self.args.seed is not None:
                random.Random(self.args.seed).shuffle(samples)
            if self.args.max_train_samples:
                samples = samples[:self.args.max_train_samples]

        return samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        return {
            "pixel_values": pixel_values,
            "prompts": item["prompt"],
        }

        
def collate_ltb(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    prompts = [e["prompts"] for e in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def collate_fn(examples, clip_image_processor):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    source_images = torch.stack([example["source_image"] for example in examples])
    source_images = source_images.to(memory_format=torch.contiguous_format).float()

    prompts = [example["prompts"] for example in examples]
    
    style_images = [example["style_images"] for example in examples]
    clip_images = clip_image_processor(images=style_images, return_tensors="pt").pixel_values
    
    drop_image_embeds = torch.tensor([example["drop_image_embeds"] for example in examples])
    
    masks = torch.stack([example["mask"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "source_image": source_images,
        "prompts": prompts,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "mask": masks,
    }
