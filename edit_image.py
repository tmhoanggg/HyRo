import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from cat_seg import add_cat_seg_config
from train_net import Trainer 

class CATSegDiffusionEditor:
    def __init__(self, config_path, weights_path, device="cuda"):
        self.device = device
        
        # 1. Load CAT-Seg model
        cfg = get_cfg()
        add_cat_seg_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).resume_or_load(weights_path)
        
        # Get metadata which contains class names
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        
        # Get class names from metadata
        if hasattr(self.metadata, 'stuff_classes'):
            self.class_names = self.metadata.stuff_classes
            print(f"Found {len(self.class_names)} stuff classes")
        elif hasattr(self.metadata, 'thing_classes'):
            self.class_names = self.metadata.thing_classes
            print(f"Found {len(self.class_names)} thing classes")
        else:
            self.class_names = None
            print("No class names found in metadata")
        
        # 2. Load Stable Diffusion Inpainting Pipeline
        print("Loading Stable Diffusion Inpainting model...")
        self.diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16
        ).to(device)
        print("Stable Diffusion model loaded!")

    @torch.no_grad()
    def get_mask_for_class(self, image_path, target_class_name):
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Preprocessing
        image_tensor = torch.as_tensor(np.array(img).transpose(2, 0, 1)).to(self.device)
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        # Forward pass
        outputs = self.model(inputs)
        
        if "sem_seg" not in outputs[0]:
            raise ValueError("Model output does not contain semantic segmentation!")
        
        sem_seg = outputs[0]["sem_seg"]
        
        # Find the class index for the target class name
        target_class_idx = None
        if self.class_names:
            # Try exact match first
            if target_class_name in self.class_names:
                target_class_idx = self.class_names.index(target_class_name)
            else:
                # Try partial match (case-insensitive)
                target_lower = target_class_name.lower()
                for idx, class_name in enumerate(self.class_names):
                    if target_lower in class_name.lower() or class_name.lower() in target_lower:
                        target_class_idx = idx
                        print(f"Found similar class: '{class_name}' (index {idx})")
                        break
        
        if target_class_idx is None:
            raise ValueError(f"Class '{target_class_name}' not found in available classes!")
        
        # Generate mask for the target class
        mask = (sem_seg.argmax(dim=0) == target_class_idx).cpu().numpy().astype(np.uint8) * 255
        coverage = (mask > 0).sum() / mask.size * 100
        
        print(f"Target class '{target_class_name}' (index {target_class_idx}): {coverage:.2f}% coverage")
        
        if coverage < 0.1:
            print(f"Warning: Very low coverage ({coverage:.2f}%) - the class might not be present in the image")
        
        return Image.fromarray(mask), img, target_class_idx

    @torch.no_grad()
    def get_all_masks(self, image_path):
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        image_tensor = torch.as_tensor(np.array(img).transpose(2, 0, 1)).to(self.device)
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        outputs = self.model(inputs)
        
        if "sem_seg" not in outputs[0]:
            return {}, img
        
        sem_seg = outputs[0]["sem_seg"]
        num_classes = sem_seg.shape[0]
        
        masks = {}
        print("\nAvailable classes in image:")
        print("=" * 60)
        
        for class_idx in range(num_classes):
            mask = (sem_seg.argmax(dim=0) == class_idx).cpu().numpy().astype(np.uint8) * 255
            coverage = (mask > 0).sum() / mask.size * 100
            
            if coverage > 0.1:  # Only show classes with >0.1% coverage
                if self.class_names and class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"class_{class_idx}"
                
                masks[class_name] = Image.fromarray(mask)
                print(f"  [{class_idx:3d}] {class_name:30s}: {coverage:6.2f}% coverage")
        
        print("=" * 60)
        return masks, img

    def edit_class(self, image_path, target_class, edit_prompt, negative_prompt="", 
                   strength=0.75, guidance_scale=7.5, num_inference_steps=50,
                   dilate_pixels=20, blur_kernel=51):
        # Get mask for the target class
        mask_img, original_img, class_idx = self.get_mask_for_class(image_path, target_class)
        
        # Improve mask quality for better blending
        mask_np = np.array(mask_img)
        
        # Dilate to expand mask slightly
        if dilate_pixels > 0:
            kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            print(f"Dilated mask by {dilate_pixels}px")

        # Save binary mask BEFORE blurring
        binary_mask = Image.fromarray(mask_np)  # Pure black and white
        
        # Blur edges for smooth transition
        if blur_kernel > 0:
            # Ensure kernel size is odd
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            mask_np = cv2.GaussianBlur(mask_np, (blur_kernel, blur_kernel), 0)
            print(f"Blurred mask edges with kernel size {blur_kernel}")
        
        mask_img = Image.fromarray(mask_np)
        
        # Add better default negative prompt if not provided
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, artifacts, watermark"
        
        print(f"\nGenerating edited image...")
        print(f"Prompt: {edit_prompt}")
        print(f"Negative prompt: {negative_prompt}")
        print(f"Strength: {strength}, Guidance: {guidance_scale}, Steps: {num_inference_steps}")
        
        # Run Stable Diffusion inpainting
        edited_image = self.diffusion_pipe(
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            image=original_img,
            mask_image=mask_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=(original_img.height // 8) * 8,
            width=(original_img.width // 8) * 8,
        ).images[0]
        
        return edited_image, binary_mask, original_img

    @torch.no_grad()
    def edit_everything_except(self, image_path, keep_classes, edit_prompt, negative_prompt="",
                              strength=0.75, guidance_scale=7.5, num_inference_steps=50,
                              dilate_pixels=0, blur_kernel=51):
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Get all segmentation
        image_tensor = torch.as_tensor(np.array(img).transpose(2, 0, 1)).to(self.device)
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        outputs = self.model(inputs)
        
        if "sem_seg" not in outputs[0]:
            raise ValueError("Model output does not contain semantic segmentation!")
        
        sem_seg = outputs[0]["sem_seg"]
        
        # Create a mask for everything EXCEPT the keep_classes
        keep_mask = torch.zeros_like(sem_seg[0], dtype=torch.bool)
        
        print("\nClasses being protected (kept unchanged):")
        for keep_class in keep_classes:
            if self.class_names and keep_class in self.class_names:
                class_idx = self.class_names.index(keep_class)
                class_mask = (sem_seg.argmax(dim=0) == class_idx)
                coverage = class_mask.sum().item() / class_mask.numel() * 100
                keep_mask = keep_mask | class_mask
                print(f"  ✓ {keep_class} (index {class_idx}): {coverage:.2f}% coverage")
            else:
                print(f"  ✗ Warning: '{keep_class}' not found in image")
        
        # Invert: mask everything EXCEPT what we want to keep
        edit_mask = (~keep_mask).cpu().numpy().astype(np.uint8) * 255
        
        # Optional: Erode the protected area slightly to avoid bleeding
        if dilate_pixels < 0:  # Negative value = erode protected area
            kernel = np.ones((abs(dilate_pixels), abs(dilate_pixels)), np.uint8)
            edit_mask = cv2.erode(edit_mask, kernel, iterations=1)
            print(f"Eroded protected area by {abs(dilate_pixels)}px to avoid edge artifacts")

        # Save binary mask BEFORE blurring
        binary_mask = Image.fromarray(edit_mask)  # Pure black and white

        # Blur for smooth edges (for inpainting only)
        if blur_kernel > 0:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            edit_mask = cv2.GaussianBlur(edit_mask, (blur_kernel, blur_kernel), 0)
            print(f"Blurred mask edges with kernel size {blur_kernel}")

        mask_img_for_inpainting = Image.fromarray(edit_mask)  # Blurred version
        
        coverage = (edit_mask > 128).sum() / edit_mask.size * 100
        print(f"\nEditing {coverage:.2f}% of image (protecting {100-coverage:.2f}%)")
        
        # Add better default negative prompt
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, ugly, same"
        
        print(f"\nGenerating edited image...")
        print(f"Prompt: {edit_prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Run Stable Diffusion inpainting
        edited_image = self.diffusion_pipe(
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask_img_for_inpainting,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=(img.height // 8) * 8,
            width=(img.width // 8) * 8,
        ).images[0]
        
        return edited_image, binary_mask, img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAT-Seg + Stable Diffusion Image Editor")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--config", required=True, help="Path to CAT-Seg config file")
    parser.add_argument("--weights", required=True, help="Path to CAT-Seg model weights")
    parser.add_argument("--output", default="/kaggle/working/", help="Output directory")
    
    # Mode selection
    parser.add_argument("--mode", choices=["list", "edit", "edit-background"], default="list",
                        help="'list': see classes | 'edit': edit specific class | 'edit-background': keep subject, change background")
    
    # For 'edit' mode - edit specific class
    parser.add_argument("--target-class", type=str, 
                        help="Class to edit (e.g., 'floor-tile', 'wall')")
    
    # For 'edit-background' mode - keep certain classes unchanged
    parser.add_argument("--keep-classes", nargs='+', 
                        help="Classes to keep unchanged (e.g., cat person dog)")
    
    # Common edit parameters
    parser.add_argument("--prompt", type=str, 
                        help="Edit prompt (e.g., 'lush green grass lawn in a garden')")
    parser.add_argument("--negative-prompt", type=str, default="", 
                        help="Negative prompt (what to avoid)")
    parser.add_argument("--strength", type=float, default=0.75, 
                        help="Edit strength (0.0-1.0, lower=subtle, higher=dramatic)")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="How closely to follow prompt (7-15 typical)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Quality/speed tradeoff (50-100 recommended)")
    
    # Mask refinement parameters
    parser.add_argument("--dilate-pixels", type=int, default=20,
                        help="Expand mask (positive) or shrink protected area (negative for edit-background)")
    parser.add_argument("--blur-kernel", type=int, default=51,
                        help="Blur amount for smooth edges (must be odd, 0 to disable)")
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize editor
    editor = CATSegDiffusionEditor(
        config_path=args.config,
        weights_path=args.weights
    )
    
    if args.mode == "list":
        # List all available classes in the image
        print("\n" + "=" * 60)
        print("LISTING ALL CLASSES IN IMAGE")
        print("=" * 60)
        masks, original_img = editor.get_all_masks(args.input)
        
        # Save all masks for reference
        for class_name, mask in masks.items():
            clean_name = class_name.replace(" ", "_").replace("/", "_")
            mask_path = os.path.join(args.output, f"mask_{clean_name}.png")
            mask.save(mask_path)
        
        print(f"\nSaved {len(masks)} masks to {args.output}")
        print("\n" + "=" * 60)
        print("USAGE EXAMPLES:")
        print("=" * 60)
        print("\n1. To edit a specific class (e.g., change floor to grass):")
        print(f"   python {os.path.basename(__file__)} --mode edit --target-class floor-tile \\")
        print(f"     --prompt 'lush green grass lawn' --input {args.input} ...")
        print("\n2. To keep subject and change background (e.g., keep cat, change floor):")
        print(f"   python {os.path.basename(__file__)} --mode edit-background --keep-classes cat \\")
        print(f"     --prompt 'cat on green grass in sunny garden' --input {args.input} ...")
        print("=" * 60)
        
    elif args.mode == "edit":
        # Edit specific class
        if not args.target_class or not args.prompt:
            raise ValueError("--target-class and --prompt are required for edit mode!")
        
        print("\n" + "=" * 60)
        print("EDITING SPECIFIC CLASS")
        print("=" * 60)
        
        edited_img, mask_img, original_img = editor.edit_class(
            image_path=args.input,
            target_class=args.target_class,
            edit_prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            dilate_pixels=args.dilate_pixels,
            blur_kernel=args.blur_kernel
        )
        
        # Save results
        original_img.save(os.path.join(args.output, "original.png"))
        mask_img.save(os.path.join(args.output, "mask.png"))
        edited_img.save(os.path.join(args.output, "edited.png"))
        
        print("\n" + "=" * 60)
        print("SAVED FILES:")
        print("=" * 60)
        print(f"Original: {os.path.join(args.output, 'original.png')}")
        print(f"Mask: {os.path.join(args.output, 'mask.png')}")
        print(f"Edited: {os.path.join(args.output, 'edited.png')}")
        print("=" * 60)
        
    elif args.mode == "edit-background":
        # Keep certain classes, edit everything else
        if not args.keep_classes or not args.prompt:
            raise ValueError("--keep-classes and --prompt are required for edit-background mode!")
        
        print("\n" + "=" * 60)
        print("EDITING BACKGROUND (KEEPING SUBJECTS)")
        print("=" * 60)
        
        edited_img, mask_img, original_img = editor.edit_everything_except(
            image_path=args.input,
            keep_classes=args.keep_classes,
            edit_prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            dilate_pixels=args.dilate_pixels,
            blur_kernel=args.blur_kernel
        )
        
        # Save results
        original_img.save(os.path.join(args.output, "original.png"))
        mask_img.save(os.path.join(args.output, "mask.png"))
        edited_img.save(os.path.join(args.output, "edited.png"))
        
        print("\n" + "=" * 60)
        print("SAVED FILES:")
        print("=" * 60)
        print(f"Original: {os.path.join(args.output, 'original.png')}")
        print(f"Mask: {os.path.join(args.output, 'mask.png')}")
        print(f"Edited: {os.path.join(args.output, 'edited.png')}")
        print("=" * 60)