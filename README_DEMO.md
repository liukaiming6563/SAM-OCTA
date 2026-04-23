# SAM-OCTA Minimal Inference Demo (No Training)

This demo is for a 10-minute group presentation and only runs inference on **one OCTA-500 sample**.

## Required Weights

Place these files in your repo:

- Base SAM weight (ViT-B):
  - `sam_weights/sam_vit_b_01ec64.pth`
- LoRA task weights:
  - `sam_weights/vit_b/3M_LargeVessel_Global.pth`
  - `sam_weights/vit_b/3M_FAZ_Local.pth`

## Default Sample

- Default sample id: `10301`
- Data path expected by repository code:
  - `datasets/OCTA-500/OCTA_3M/...`

## How to Run

### Option A: Run both demos on Windows

```bat
run_demo_predict.bat
```

### Option B: Run one task manually

LargeVessel + Global:

```bat
python demo_predict_one_sample.py --task lv_global --sample-id 10301 --sam-checkpoint sam_weights/sam_vit_b_01ec64.pth --lora-checkpoint sam_weights/vit_b/3M_LargeVessel_Global.pth --output-dir demo_outputs
```

FAZ + Local:

```bat
python demo_predict_one_sample.py --task faz_local --sample-id 10301 --sam-checkpoint sam_weights/sam_vit_b_01ec64.pth --lora-checkpoint sam_weights/vit_b/3M_FAZ_Local.pth --output-dir demo_outputs
```

## Output Location

Output images are saved to:

- `demo_outputs/lv_global_10301.png`
- `demo_outputs/faz_local_10301.png`

Each output image contains:

- original image (FULL layer)
- GT target mask used for prompting
- prompt points overlay
- prediction result

## Notes

- This script does **not** train.
- It reuses existing repository modules:
  - `dataset.py`
  - `prompt_points.py`
  - `sam_lora_image_encoder.py`
  - `display.py`
