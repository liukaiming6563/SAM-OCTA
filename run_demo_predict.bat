@echo off
REM Run from project root: D:\study\project\SAM-OCTA

REM Demo 1: 3M + LargeVessel + Global
python demo_predict_one_sample.py --task lv_global --sample-id 10301 --sam-checkpoint sam_weights/sam_vit_b_01ec64.pth --lora-checkpoint sam_weights/vit_b/3M_LargeVessel_Global.pth --output-dir demo_outputs

REM Demo 2: 3M + FAZ + Local
python demo_predict_one_sample.py --task faz_local --sample-id 10301 --sam-checkpoint sam_weights/sam_vit_b_01ec64.pth --lora-checkpoint sam_weights/vit_b/3M_FAZ_Local.pth --output-dir demo_outputs
