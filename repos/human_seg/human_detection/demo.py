from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

import torch

from mmdet_model import DetectionModel

detector = DetectionModel(model_dir='/cfs/zhlin/dockers/sd_ctl-v1-1/baoensi_img2img_api/models/cv_resnet18_human-detection', half=True)


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


def detect_human(image_pil):
    image_np = np.array(image_pil, np.uint8)
    torch.cuda.synchronize()
    tic = time.time()
    bboxes, scores, labels = detector(image_np)
    print(f'bboxes: {bboxes}')
    print(f'scores: {scores}')
    print(f'labels: {labels}')
    torch.cuda.synchronize()
    print(f"Detecting faces, cost {time.time() - tic:.4f} secs.")
    image_draw = ImageDraw.Draw(image_pil)
    for box, score in zip(bboxes, scores):
        draw_box(box, image_draw, label=f'{score}')
    
    return image_pil


if __name__ == "__main__":
    import gradio as gr
    
    #image_pil = Image.open('/cfs/zhlin/stable-diffusion-webui/webui/test_images/001.jpeg').convert('RGB')
    #image_np = np.array(image_pil, np.uint8)
    
    input_image = gr.Image(label="输入图像", show_label=True, type="pil", image_mode="RGB", elem_id="input", source="upload")
    output_image = gr.Image(label="检测结果", show_label=True, type="pil", image_mode="RGB", elem_id="output")

    interface = gr.Interface(
        fn=detect_human, 
        inputs=[input_image], 
        outputs=[output_image],
        examples=[['/cfs/zhlin/stable-diffusion-webui/webui/test_images/001.jpeg']],
        title="Faster-RCNN (ResNet18) 人体检测(8803)"
        )
    
    interface.launch(server_name="0.0.0.0", server_port=8803)
