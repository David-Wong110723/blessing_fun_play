import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
from easydict import EasyDict as edict

from srcs.vidgen_pipeline import VideoGENPipeline

import warnings
warnings.filterwarnings("ignore")


pipeline = VideoGENPipeline()


def run():
    pipeline.dancing()


if __name__ == "__main__":
    # import gradio as gr
    # input = gr.Image(label="输入图片", show_label=True, elem_id="input_image", source="upload", type="pil", image_mode="RGB")
    # #output = gr.Image(label="结果图片", show_label=True, elem_id="output_image", type="numpy", image_mode="RGB")
    # interface = gr.Interface(
    #     fn=run, 
    #     inputs=[input], 
    #     outputs=[gr.Gallery(label='结果图', preview=True, elem_id="results")],
    #     examples=["/cfs/zhlin/projects/aigc_engine/apps/vidgen/assets/kunkun.jpg"], 
    #     title="图片分析Demo"
    # )
    # interface.launch(server_name="0.0.0.0", server_port=8899)
    run()
