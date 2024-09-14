import os
import sys
import subprocess

env_vars = os.environ.copy()
HOME = os.getcwd()
sys.path.insert(0, "weights")
sys.path.insert(0, "weights/GroundingDINO")
sys.path.insert(0, "weights/segment-anything")
os.chdir("/home/ni/prop/grounded_sam_replicate2/weights/GroundingDINO")
#subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir("/home/ni/prop/grounded_sam_replicate2/weights/segment-anything")
#subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
os.chdir(HOME)

import torch
from typing import Iterator
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam
# import uuid
from hf_path_exports import cache_config_file, cache_file


def setup():
    """Load the model into memory to make running multiple predictions efficient"""
    print("Loading pipelines...")

    # 选择设备：GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义加载模型的函数
    def load_model_hf(device='cpu'):
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print(f"Model loaded from {cache_file} \n => {log}")
        _ = model.eval()
        return model

    # 加载 groundingdino 模型
    groundingdino_model = load_model_hf(device)

    # 加载 SAM 模型
    sam_checkpoint = '/home/ni/prop/grounded_sam_replicate2/weights/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    return groundingdino_model, sam_predictor


def predict(groundingdino_model, sam_predictor, image, mask_prompt, negative_mask_prompt, adjustment_factor):
    """运行预测"""
    #predict_id = str(uuid.uuid4())
    #print(f"Running prediction: {predict_id}...")

    # 调用 groundingdino 和 sam 模型进行预测
    annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = run_grounding_sam(
        image, mask_prompt, negative_mask_prompt,
        groundingdino_model, sam_predictor,
        adjustment_factor
    )
    annotated_picture_mask.save("1.png")
    neg_annotated_picture_mask.save("2.png")
    mask.save("3.png")
    inverted_mask.save("4.png")
    print("Done!")
    return annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask


# 使用函数
groundingdino_model, sam_predictor = setup()

# # 定义预测所需的输入参数
image = "/home/ni/prop/grounded_sam_replicate2/image_11057_original.jpeg"  # 你的图片数据（例如numpy array或其他格式）
mask_prompt = "watermark, text"  # mask的提示
negative_mask_prompt = ""  # 负mask提示
adjustment_factor = 10  # 你希望的调整因子
#
# # 调用 predict 
annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = predict(groundingdino_model, sam_predictor,image, mask_prompt, negative_mask_prompt, adjustment_factor)
annotated_picture_mask.save("1.png")
neg_annotated_picture_mask.save("2.png")
mask.save("3.png")
inverted_mask.save("4.png")
image = "/home/ni/prop/grounded_sam_replicate2/image_01790_original.jpeg"
annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = predict(groundingdino_model, sam_predictor,image, mask_prompt, negative_mask_prompt, adjustment_factor)
annotated_picture_mask.save("4.png")
neg_annotated_picture_mask.save("5.png")
mask.save("6.png")
inverted_mask.save("7.png")
