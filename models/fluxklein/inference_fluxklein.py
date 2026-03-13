import os
import sys
import json
import inspect
import torch
import argparse
from PIL import Image
from safetensors.torch import load_file
from diffusers.utils import load_image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from .models import AutoencoderKLFlux2, Flux2Transformer2DModel
from .pipeline_flux2_klein import Flux2KleinPipeline


def load_model_weights(model, model_path, subfolder, torch_dtype):
    """Load model weights from safetensors or pytorch files."""
    weights_path = os.path.join(model_path, subfolder)
    
    # Check for sharded weights (index file) - try different naming conventions
    index_files = [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ]
    
    index_file = None
    for idx_file in index_files:
        idx_path = os.path.join(weights_path, idx_file)
        if os.path.exists(idx_path):
            index_file = idx_path
            break
    
    if index_file:
        # Load sharded weights
        print(f"  Loading sharded weights from {os.path.basename(index_file)}...")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        weight_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for weight_file in weight_files:
            file_path = os.path.join(weights_path, weight_file)
            print(f"    Loading {weight_file}...")
            state_dict.update(load_file(file_path))
    else:
        # Try single safetensors file
        safetensors_file = os.path.join(weights_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(safetensors_file):
            print(f"  Loading single weight file: diffusion_pytorch_model.safetensors")
            state_dict = load_file(safetensors_file)
        else:
            # Try pytorch file
            pytorch_file = os.path.join(weights_path, "diffusion_pytorch_model.bin")
            if os.path.exists(pytorch_file):
                print(f"  Loading pytorch weight file: diffusion_pytorch_model.bin")
                state_dict = torch.load(pytorch_file, map_location="cpu")
            else:
                raise FileNotFoundError(f"No weights found in {weights_path}")
    
    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in {subfolder}: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Warning: Missing keys in {subfolder}: {missing_keys}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys in {subfolder}: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Warning: Unexpected keys in {subfolder}: {unexpected_keys}")
    
    return model.to(torch_dtype)


def load_flux2_klein_pipeline(model_path, torch_dtype=torch.bfloat16, device="cuda"):
    """
    Manually load FLUX.2 Klein pipeline components using local model definitions.
    This bypasses diffusers' dynamic class loading to ensure we use local models.
    """
    print(f"Loading FLUX.2 Klein pipeline from {model_path}...")
    
    # Load scheduler
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, 
        subfolder="scheduler"
    )
    
    # Load VAE config and create model with local class
    print("Loading VAE...")
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    with open(vae_config_path, 'r') as f:
        vae_config = json.load(f)
    
    # Remove metadata
    vae_config.pop("_class_name", None)
    vae_config.pop("_diffusers_version", None)
    
    # Get valid parameters for AutoencoderKLFlux2
    valid_params = set(inspect.signature(AutoencoderKLFlux2.__init__).parameters.keys())
    valid_params.discard('self')
    
    # Filter config to only include valid parameters
    filtered_config = {k: v for k, v in vae_config.items() if k in valid_params}
    removed_params = set(vae_config.keys()) - set(filtered_config.keys())
    if removed_params:
        print(f"  Removed unsupported VAE config parameters: {removed_params}")
    
    vae = AutoencoderKLFlux2(**filtered_config)
    vae = load_model_weights(vae, model_path, "vae", torch_dtype)
    
    # Load transformer config and create model with local class
    print("Loading transformer...")
    transformer_config_path = os.path.join(model_path, "transformer", "config.json")
    with open(transformer_config_path, 'r') as f:
        transformer_config = json.load(f)
    
    # Remove metadata and unsupported parameters
    transformer_config.pop("_class_name", None)
    transformer_config.pop("_diffusers_version", None)
    transformer_config.pop("guidance_embeds", None)  # Not supported in Flux2Transformer2DModel
    
    # Get valid parameters for Flux2Transformer2DModel
    valid_params = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys())
    valid_params.discard('self')
    
    # Filter config to only include valid parameters
    filtered_config = {k: v for k, v in transformer_config.items() if k in valid_params}
    removed_params = set(transformer_config.keys()) - set(filtered_config.keys())
    if removed_params:
        print(f"  Removed unsupported config parameters: {removed_params}")
    
    transformer = Flux2Transformer2DModel(**filtered_config)
    transformer = load_model_weights(transformer, model_path, "transformer", torch_dtype)
    
    # Load text encoder and tokenizer using transformers (these don't have the same issue)
    print("Loading text encoder...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype
    )
    
    print("Loading tokenizer...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_path,
        subfolder="tokenizer"
    )
    
    # Create pipeline with manually loaded components
    print("Assembling pipeline...")
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    
    print("Pipeline loaded successfully!")
    return pipe


class FluxKleinGenerator:
    def __init__(self, model_path="black-forest-labs/FLUX.2-klein-base-9B", device="cuda", enable_cpu_offload=False):
        print("Initializing Flux-Klein pipeline...")
        self.device = device
        self.dtype = torch.bfloat16
        
        # Use custom loader to ensure local model definitions are used
        self.pipe = load_flux2_klein_pipeline(
            model_path,
            torch_dtype=self.dtype,
            device=device
        )
        
        if enable_cpu_offload:
            # IMPORTANT: In multi-process/multi-GPU, diffusers defaults to gpu_id=0.
            # If we don't pass the correct gpu_id, every process will offload/execute on GPU 0 -> OOM.
            gpu_id = 0
            if isinstance(device, str) and device.startswith("cuda:"):
                try:
                    gpu_id = int(device.split("cuda:")[-1])
                except ValueError:
                    gpu_id = 0
            try:
                self.pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            except TypeError:
                # Backward compatibility for older diffusers that don't accept gpu_id.
                # In this case, ensure the current CUDA device is set by the caller.
                self.pipe.enable_model_cpu_offload()
            print("Flux-Klein pipeline initialized with CPU offload.")
        else:
            self.pipe.to(device)
            print(f"Flux-Klein pipeline initialized on {device}.")

    def generate(
        self,
        prompt: str,
        image: Image.Image = None,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        output_path: str = "output/fluxklein/output.png"
    ):
        """
        Generate or edit an image using Flux-Klein.
        
        Args:
            prompt: Text description of the image to generate
            image: Optional input image for image-to-image editing. If None, generates from scratch.
            seed: Random seed for reproducibility
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Output image height (only used when image is None)
            width: Output image width (only used when image is None)
            output_path: Path to save the generated image
        
        Returns:
            PIL.Image: Generated image
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare kwargs
        kwargs = {
            "prompt": prompt,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add image for editing mode, or height/width for generation mode
        if image is not None:
            # Image editing mode (image-to-image)
            kwargs["image"] = [image]  # multi-image input format
            print(f"Running in image editing mode with input image size: {image.size}")
        else:
            # Text-to-image generation mode
            new_width = (width // 32) * 32
            new_height = (height // 32) * 32
            kwargs["height"] = new_height
            kwargs["width"] = new_width
            print(f"Running in text-to-image mode with size: {new_width}x{new_height}")
        
        result = self.pipe(**kwargs).images[0]
        
        result.save(output_path)
        print(f"Image saved to {output_path}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Flux-Klein Text-to-Image and Image Editing Script")
    parser.add_argument("--model_path", type=str, 
                       default="black-forest-labs/FLUX.2-klein-base-9B",
                       help="Path to the Flux-Klein model.")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="The prompt describing the image to generate or edit.")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to input image for editing. If not provided, generates from scratch.")
    parser.add_argument("--output_path", type=str, default="output/fluxklein.png", 
                       help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--height", type=int, default=1024, help="Image height (for text-to-image mode).")
    parser.add_argument("--width", type=int, default=1024, help="Image width (for text-to-image mode).")
    parser.add_argument("--enable_cpu_offload", action="store_true", 
                       help="Enable CPU offload (requires more VRAM).")
    args = parser.parse_args()

    generator = FluxKleinGenerator(
        model_path=args.model_path,
        enable_cpu_offload=args.enable_cpu_offload
    )
    
    # Load input image if provided
    input_image = None
    if args.image_path:
        input_image = load_image(args.image_path).convert("RGB")
        print(f"Loaded input image from {args.image_path}")
    
    generator.generate(
        prompt=args.prompt,
        image=input_image,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
