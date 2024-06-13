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

import warnings
warnings.filterwarnings("ignore")


config_path = "/cfs/zhlin/projects/aigc_engine/apps/vidgen/configs/models.json"
with open(config_path, "r") as f:
    model_config = edict(json.load(f))
    f.close()    

age_gender_estimator = AgeGenderEstimator(weight_path=model_config.FaceDetection.path)
face_detector = FaceDetector(weight_path=model_config.FaceDetection.path)
human_segger = HumanSeg(weight_dir=model_config.HumanSegmentation.path)


def draw_faces(img_raw, bboxes, scores, landmarks, genders, ages):
    image = img_raw.copy()
    for bbox, score, landmark, gender, age in zip(bboxes, scores, landmarks, genders, ages):
        text_score = f"{score[0]: .4f}"
        text_gender_age = f"{gender}, {age}"
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cx = int(bbox[0])
        cy = int(bbox[1]) + 12
        cv2.putText(image, text_score, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
        cv2.putText(image, text_gender_age, (cx, cy+15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        # landms
        cv2.circle(image, (int(landmark[0][0]), int(landmark[0][1])), 1, (0, 0, 255), 4)
        cv2.circle(image, (int(landmark[1][0]), int(landmark[1][1])), 1, (0, 255, 255), 4)
        cv2.circle(image, (int(landmark[2][0]), int(landmark[2][1])), 1, (255, 0, 255), 4)
        cv2.circle(image, (int(landmark[3][0]), int(landmark[3][1])), 1, (0, 255, 0), 4)
        cv2.circle(image, (int(landmark[4][0]), int(landmark[4][1])), 1, (255, 0, 0), 4)
    return image


def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)


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


def human_seg(image: Image.Image):
    bboxes, scores, labels = human_segger.detect(image, with_phrase=True)
    masks = human_segger.segment(image, bboxes)
    for mask in masks:
        image = draw_mask(mask[0].cpu().numpy(), image, random_color=True)
    return image


def detect(image: Image.Image):
    image = image.convert('RGB')
    width, height = image.size
    image_np_rgb = np.array(image, np.uint8)
    image_np_bgr = image_np_rgb[:,:,::-1].copy()  # RGB -> BGR
    faces, boxes, scores, landmarks = face_detector.detect_align(image_np_bgr)
    if len(faces) > 0:
        genders, ages = age_gender_estimator.detect(faces)
        vis_image = draw_faces(image_np_rgb, boxes, scores, landmarks, genders, ages)
    else:
        vis_image = image
    bbox = boxes[0].cpu().numpy().tolist()
    print(bbox)
    padded_face_bbox = pad_bbox(bbox, padding_ratios=[0.2, 0.25, 0.2, 0.2], img_width=width, img_height=height, to_square=True)
    print(padded_face_bbox)
    face_image = crop_image(image, padded_face_bbox)

    return vis_image, face_image


def run(image: Image.Image):
    seg_result = human_seg(image)
    det_result, face_image = detect(image)
    return [det_result, seg_result, face_image]


if __name__ == "__main__":
    import gradio as gr
    input = gr.Image(label="输入图片", show_label=True, elem_id="input_image", source="upload", type="pil", image_mode="RGB")
    #output = gr.Image(label="结果图片", show_label=True, elem_id="output_image", type="numpy", image_mode="RGB")
    interface = gr.Interface(
        fn=run, 
        inputs=[input], 
        outputs=[gr.Gallery(label='结果图', preview=True, elem_id="results")],
        examples=["/cfs/zhlin/projects/aigc_engine/apps/vidgen/assets/kunkun.jpg"], 
        title="图片分析Demo"
    )
    interface.launch(server_name="0.0.0.0", server_port=8899)
