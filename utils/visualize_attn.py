import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import clip  # Standard CLIP library for tokenizer

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import Checkpointer
from cat_seg import add_cat_seg_config

def setup_cfg(args):
    cfg = get_cfg()
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Visualize CAT-Seg Class Attention Maps")
    parser.add_argument("--config", required=True, help="Path to config file (e.g., configs/vitb_384.yaml)")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--cls", required=True, help="Comma separated classes (e.g., 'cat,dog,tree')")
    parser.add_argument("--output", default="output_vis", help="Output directory")
    args = parser.parse_args()

    # 1. Load Model
    cfg = setup_cfg(args)
    model = build_model(cfg)
    Checkpointer(model).load(args.weights)
    model.eval()

    # 2. Load Image
    raw_image = cv2.imread(args.input)
    if raw_image is None:
        print(f"Error: Could not load image {args.input}")
        return

    image_tensor = torch.as_tensor(raw_image.astype("float32").transpose(2, 0, 1))
    classes = [c.strip() for c in args.cls.split(",")]
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        # Preprocessing using CLIP-specific mean/std
        clip_image = (image_tensor.to(model.device) - model.clip_pixel_mean) / model.clip_pixel_std
        clip_image = F.interpolate(clip_image.unsqueeze(0), size=model.clip_resolution, mode='bilinear', align_corners=False)

        # 3. Extract Image Features
        # Using dense=True to get patch-level tokens for spatial attention
        clip_features = model.sem_seg_head.predictor.clip_model.encode_image(clip_image, dense=True)
        image_embeds = clip_features[:, 1:, :] # Skip CLS token
        image_embeds = F.normalize(image_embeds, dim=-1)

        for class_name in classes:
            print(f"Processing: {class_name}")
            
            # 4. Corrected Tokenization using global clip module
            text_prompt = f"a photo of a {class_name}"
            text_tokens = clip.tokenize([text_prompt]).to(model.device)
            
            # 5. Extract Text Embedding
            text_embeds = model.sem_seg_head.predictor.clip_model.encode_text(text_tokens)
            text_embeds = F.normalize(text_embeds, dim=-1)

            # 6. Generate Similarity (Attention) Map
            similarity = torch.matmul(image_embeds, text_embeds.t()) 
            grid_size = int(np.sqrt(similarity.shape[1]))
            attn_map = similarity.view(grid_size, grid_size).cpu().numpy()

            # 7. Post-Processing & Save
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map_resized = cv2.resize(attn_map, (raw_image.shape[1], raw_image.shape[0]))
            
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
            vis_output = cv2.addWeighted(raw_image, 0.6, heatmap, 0.4, 0)

            out_path = os.path.join(args.output, f"attn_{class_name.replace(' ', '_')}.jpg")
            cv2.imwrite(out_path, vis_output)

    print(f"Done! All visualizations saved in: {args.output}")

if __name__ == "__main__":
    main()