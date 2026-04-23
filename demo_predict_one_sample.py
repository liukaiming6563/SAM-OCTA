import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import torch

from dataset import octa500_2d_dataset
from display import show_result_sample_figure
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal SAM-OCTA inference demo for one sample")
    parser.add_argument("--task", type=str, default="lv_global", choices=["lv_global", "faz_local"],
                        help="Demo preset: lv_global=3M+LargeVessel+Global, faz_local=3M+FAZ+Local")
    parser.add_argument("--sample-id", type=str, default="10301", help="Sample id inside OCTA-500")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_weights/sam_vit_b_01ec64.pth",
                        help="Base SAM ViT-B checkpoint")
    parser.add_argument("--lora-checkpoint", type=str, default=None,
                        help="Task LoRA checkpoint (.pth). If omitted, uses preset path by task")
    parser.add_argument("--output-dir", type=str, default="demo_outputs", help="Directory to save visualization")
    parser.add_argument("--prompt-positive-num", type=int, default=1, help="Positive prompt points")
    parser.add_argument("--prompt-negative-num", type=int, default=1, help="Negative prompt points")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible prompt sampling")
    return parser.parse_args()


def task_preset(task: str) -> Tuple[str, bool, str]:
    if task == "lv_global":
        return "LargeVessel", False, "sam_weights/vit_b/3M_LargeVessel_Global.pth"
    return "FAZ", True, "sam_weights/vit_b/3M_FAZ_Local.pth"


def resolve_lora_checkpoint(task: str, lora_checkpoint: str):
    if lora_checkpoint:
        return lora_checkpoint
    return task_preset(task)[2]


def load_model(sam_checkpoint: str, lora_checkpoint: str, device: torch.device):
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    model = LoRA_Sam(sam, r=4).to(device)

    state_dict = torch.load(lora_checkpoint, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_type, is_local, _ = task_preset(args.task)
    lora_checkpoint = resolve_lora_checkpoint(args.task, args.lora_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_transform = ResizeLongestSide(224)

    # use existing repo dataset + prompt generator flow
    dataset = octa500_2d_dataset(
        fov="3M",
        label_type=label_type,
        prompt_positive_num=args.prompt_positive_num,
        prompt_negative_num=args.prompt_negative_num,
        is_local=is_local,
        is_training=False,
    )

    if args.sample_id not in dataset.sample_ids:
        raise ValueError(f"sample id {args.sample_id} not found under datasets/OCTA-500")

    sample_index = dataset.sample_ids.index(args.sample_id)
    image, prompt_points, prompt_type, selected_component, sample_id = dataset[sample_index]

    # visualization uses FULL layer (channel index 2)
    image_for_show = image[2].astype(np.uint8)
    gt_for_show = (selected_component[0] * 255).astype(np.uint8)

    image_tensor = torch.tensor(np.array([image]), dtype=torch.float32, device=device)
    prompt_points_tensor = torch.tensor(np.array([prompt_points]), dtype=torch.float32, device=device)
    prompt_type_tensor = torch.tensor(np.array([prompt_type]), dtype=torch.float32, device=device)

    original_size = tuple(image_tensor.shape[-2:])
    image_tensor = sam_transform.apply_image_torch(image_tensor)
    prompt_points_tensor = sam_transform.apply_coords_torch(prompt_points_tensor, original_size)

    model = load_model(args.sam_checkpoint, lora_checkpoint, device)

    with torch.no_grad():
        pred = model(image_tensor, original_size, prompt_points_tensor, prompt_type_tensor)
        pred = torch.gt(pred, 0.8).int()[0, 0].cpu().numpy().astype(np.uint8) * 255

    prompt_info = np.concatenate(
        [prompt_points.astype(np.int32), prompt_type[:, np.newaxis].astype(np.int32)], axis=1
    )

    vis = show_result_sample_figure(image_for_show, gt_for_show, pred, prompt_info)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.task}_{sample_id}.png"
    output_path = os.path.join(args.output_dir, output_name)
    cv2.imwrite(output_path, vis)

    print("[SAM-OCTA Demo]")
    print(f"task           : {args.task}")
    print(f"label_type     : {label_type}")
    print(f"is_local       : {is_local}")
    print(f"sample_id      : {sample_id}")
    print(f"sam_checkpoint : {args.sam_checkpoint}")
    print(f"lora_checkpoint: {lora_checkpoint}")
    print(f"output_path    : {output_path}")


if __name__ == "__main__":
    main()
