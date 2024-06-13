import warnings 
warnings.filterwarnings("ignore")

import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gradio as gr
from human_seg import HumanSeg


human_segger = HumanSeg()


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


def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def human_seg(image):
    size = image.size
    
    bboxes, scores, labels = human_segger.detect(image, with_phrase=True)
    masks = human_segger.segment(image, bboxes)

    mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    image_draw = ImageDraw.Draw(image)
    for box, label in zip(bboxes, labels):
        draw_box(box, image_draw, label)

    image = image.convert('RGBA')
    image.alpha_composite(mask_image)

    return image

if __name__ == "__main__":
    input_image = gr.Image(label="输入图像", show_label=True, type="pil", image_mode="RGB", elem_id="input", source="upload")
    output_image = gr.Image(label="检测结果", show_label=True, type="pil", image_mode="RGBA", elem_id="output")

    interface = gr.Interface(
        fn=human_seg, 
        inputs=[input_image], 
        outputs=[output_image],
        title="人体检测-分割-描述(8804)"
        )
    
    interface.launch(server_name="0.0.0.0", server_port=8804)
