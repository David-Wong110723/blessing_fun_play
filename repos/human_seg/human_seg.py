import os
import numpy as np
import torch
import PIL
from typing import List

import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

from human_detection import DetectionModel

# segment anything
from segment_anything import build_sam, build_sam_vit_l, build_sam_vit_b, SamPredictor, SamAutomaticMaskGenerator
# BLIP, need pip install git+https://xxx to install the beta version
from transformers import BlipProcessor, BlipForConditionalGeneration

root_dir = os.path.dirname(os.path.abspath(__file__))

#detector_model_path = f"{root_dir}/../../models/cv_resnet18_human-detection"

sam_buiders = {
    "sam_vit_b": build_sam_vit_b,
    "sam_vit_l": build_sam_vit_l,
    "sam_vit_h": build_sam,
}
'''
sam_checkpoints = {
    "sam_vit_b": f"{root_dir}/../../models/grounded_sam/sam_vit_b_01ec64.pth",
    "sam_vit_l": f"{root_dir}/../../models/grounded_sam/sam_vit_l_0b3195.pth",
    "sam_vit_h": f"{root_dir}/../../models/grounded_sam/sam_vit_h_4b8939.pth"
}'''

class HumanSeg():
    def __init__(self, sam_type="sam_vit_l", half=False, device="cuda", weight_dir="", blip_model_path="/cfs/zhlin/sd_models/blip/blip-image-captioning-large"):
        self.half = half
        self.device = device
        # detector
        detector_model_path = f"{weight_dir}/cv_resnet18_human-detection"
        sam_checkpoints = {
            "sam_vit_b": f"{weight_dir}/grounded_sam/sam_vit_b_01ec64.pth",
            "sam_vit_l": f"{weight_dir}/grounded_sam/sam_vit_l_0b3195.pth",
            "sam_vit_h": f"{weight_dir}/grounded_sam/sam_vit_h_4b8939.pth"
        }

        self.detector = DetectionModel(detector_model_path, device)
        print("Detection Model is Builded.")
        # segment anything model
        assert sam_checkpoints[sam_type], 'sam_checkpoint is not found!'
        sam = sam_buiders[sam_type](checkpoint=sam_checkpoints[sam_type])
        sam.to(device=device)
        if half:
            sam.half()
        self.sam_predictor = SamPredictor(sam)
        print("SAM Model is Builded.")
        # BLIP model
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_path)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path, torch_dtype=torch.float16).to(device)
        print("BLIP Model is Builded.")
    
    def get_detection_output(self, image, with_phrase=False):
        with torch.no_grad():
            bboxes, scores, labels = self.detector(image)
            
        # get phrase
        if with_phrase:
            return bboxes, scores, labels
        else:
            return bboxes, None, None

    def detect(self, image, with_phrase=False):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        boxes_filt, scores, pred_phrases = self.get_detection_output(image, with_phrase)
        # return
        if with_phrase:
            return boxes_filt, scores, pred_phrases
        else:
            return boxes_filt
        
    def segment(self, image, boxes_filt):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        
        if not isinstance(boxes_filt, torch.Tensor):
            boxes_filt = torch.Tensor(boxes_filt)
        
        self.sam_predictor.set_image(image)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        return masks

    def generate_caption(self, image):
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device, torch.float16)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
