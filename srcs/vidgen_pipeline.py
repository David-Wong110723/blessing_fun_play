import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
from easydict import EasyDict as edict

from repos.face_tools.Retinaface import FaceDetector
from repos.face_tools.AgeGender import AgeGenderEstimator
from repos.human_seg.human_seg import HumanSeg
from srcs.utils import pad_bbox, crop_image
from repos.musepose.pose_align import run_align_video_with_filterPose_translate_smooth
from repos.musepose.main import main

import warnings
warnings.filterwarnings("ignore")


def draw_mask(mask, image=None, random_color=False):
    if isinstance(mask, Image.Image):
        mask = np.array(mask, np.float32) / 255.0
        mask = mask[:, :, 0]
    
    height, width = mask.shape
    mask_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_image)

    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
    
    if image is not None:
        image = image.convert('RGBA')
        image.alpha_composite(mask_image)
        return image 
    else:
        return mask_image
    

class VideoGENPipeline:
    def __init__(
        self, 
        config_path="./configs/models.json", 
        pose_align_config_path="./configs/pose_align.json",
        dancing_config_path="./configs/dancing.json",
    ):
        # 0.1 load model config
        with open(config_path, "r") as f:
            model_config = edict(json.load(f))
            f.close()
        # 0.2 load align pose config
        with open(pose_align_config_path, "r") as f:
            self.pose_align_config = edict(json.load(f))
            f.close()
        # 0.3 load dancing config
        with open(dancing_config_path, "r") as f:
            self.dancing_config = edict(json.load(f))
            f.close()
        # 1. load face detector
        self.face_detector = FaceDetector(weight_path=model_config.FaceDetection.path)
        # 2. load body segmentor
        self.body_segmentor = HumanSeg(weight_dir=model_config.HumanSegmentation.path)
        
    def face_detect(self, image):
        image = image.convert('RGB')
        width, height = image.size
        image_np_rgb = np.array(image, np.uint8)
        image_np_bgr = image_np_rgb[:,:,::-1].copy()  # RGB -> BGR
        faces, boxes, scores, landmarks = self.face_detector.detect_align(image_np_bgr)
        bbox = boxes[0].cpu().numpy().tolist()
        padded_face_bbox = pad_bbox(
            bbox, padding_ratios=[0.2, 0.25, 0.2, 0.2], img_width=width, img_height=height, to_square=True
        )
        face_image = crop_image(image, padded_face_bbox)

        return face_image

    def body_seg(self, image, return_square=True):
        bboxes, scores, labels = self.body_segmentor.detect(image, with_phrase=True)
        masks = self.body_segmentor.segment(image, bboxes)
        for mask in masks:
            image = draw_mask(mask[0].cpu().numpy(), image, random_color=True)
        return image

    def align_pose(self):
        run_align_video_with_filterPose_translate_smooth(self.pose_align_config)

    def singing(self, image, src_music):
        pass

    def dancing(self, image, src_video):
        self.align_pose()
        main(self.dancing_config)

    