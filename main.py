import os
import sys
import subprocess
import torch
from typing import Iterator
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from grounded_sam import run_grounding_sam
# import uuid
from hf_path_exports import cache_config_file, cache_file
class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def load_model_hf(device='cpu'):
            args = SLConfig.fromfile(cache_config_file)
            args.device = device
            model = build_model(args)
            checkpoint = torch.load(cache_file, map_location=device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model

        self.groundingdino_model = load_model_hf(device)
        sam_checkpoint = '/src/weights/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    def predict(image, mask_prompt, negative_mask_prompt, adjustment_factor):
        # predict_id = str(uuid.uuid4())

        # print(f"Running prediction: {predict_id}...")

        annotated_picture_mask, neg_annotated_picture_mask, mask, inverted_mask = run_grounding_sam(image,
                                                                                                    mask_prompt,
                                                                                                    negative_mask_prompt,
                                                                                                    self.groundingdino_model,
                                                                                                    self.sam_predictor,
                                                                                                    adjustment_factor)

        print("Done!")



predictor = Predictor()

# 调用 setup 方法加载模型
predictor.setup()

# # 定义预测所需的输入参数
# image = ...  # 你的图片数据（例如numpy array或其他格式）
# mask_prompt = ...  # mask的提示
# negative_mask_prompt = ...  # 负mask提示
# adjustment_factor = 1.0  # 你希望的调整因子
#
# # 调用 predict 函数
# predictor.predict(image, mask_prompt, negative_mask_prompt, adjustment_factor)