import os
import copy
from datetime import datetime
from pathlib import Path
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import cv2
import numpy as np
from PIL import Image
import moviepy.video.io.ImageSequenceClip
import random
from einops import repeat

import torch
import torch.nn.functional as F
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

from srcs.utils import load_config
#from srcs.dwpose import DWposeDetector
from srcs.pose.script.dwpose import DWposeDetector, draw_pose
from srcs.pose.script.util import size_calculate, warpAffine_kps
from srcs.musepose.models.pose_guider import PoseGuider
from srcs.musepose.models.pose_guider import PoseGuider
from srcs.musepose.models.unet_2d_condition import UNet2DConditionModel
from srcs.musepose.models.unet_3d import UNet3DConditionModel
from srcs.musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from srcs.musepose.utils.util import get_fps, read_frames, save_videos_grid
#from srcs.pose_align import align_img


'''
    Detect dwpose from img, then align it by scale parameters
    img: frame from the pose video
    detector: DWpose
    scales: scale parameters
'''
def align_img(img, pose_ori, scales, detect_resolution, image_resolution):

    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands = copy.deepcopy(pose_ori['hands'])
    faces = copy.deepcopy(pose_ori['faces'])

    '''
    计算逻辑:
    0. 该函数内进行绝对变换，始终保持人体中心点 body_pose[1] 不变
    1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
    2. 用点在图中的实际坐标来计算。
    3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
    4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
    注意：dwpose 输出是 (w, h)
    '''

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape 
    video_ratio = W_in / H_in
    body_pose[:, 0]  = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts 
    scale_neck      = scales["scale_neck"] 
    scale_face      = scales["scale_face"]
    scale_shoulder  = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand      = scales["scale_hand"]
    scale_body_len  = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):   
            scale_list[i] = scale_sum/count



    # offsets of each part 
    offset = dict()
    offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :] 
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)

    neck = body_pose[[0], :] 
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # body_pose_up_shoulder
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)

    body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14,15,16,17], :] = body_pose_up_shoulder

    # shoulder 
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2,5], :] 
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
    body_pose[[2,5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)

    body_len = body_pose[[8,11], :] 
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8,11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori['bodies']['candidate'] == -1.
    hands_none = pose_ori['hands'] == -1.
    faces_none = pose_ori['faces'] == -1.

    body_pose[body_pose_none] = -1.
    hands[hands_none] = -1. 
    nan = float('nan')
    if len(hands[np.isnan(hands)]) > 0:
        print('nan')
    faces[faces_none] = -1.

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.)
    hands = np.nan_to_num(hands, nan=-1.)
    faces = np.nan_to_num(faces, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose
    pose_align['hands'] = hands
    pose_align['faces'] = faces

    return pose_align


def scale_video(video,width,height):
    video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
    scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
    scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height, width)  # [batch, frames, channels, height, width]
    
    return scaled_video


class DancePipeline:
    def __init__(self, model_cfg_path, params_cfg_path):
        config = edict(load_config(model_cfg_path)['MusePose'])
        params_cfg = edict(load_config(params_cfg_path))
        # init models
        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        # VAE
        print(f"DANCEPIPELINE.__INIT__ Constructing vae from {config.pretrained_vae_path} ==>")    
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)
        # Reference Unet
        print(f"DANCEPIPELINE.__INIT__ Constructing reference_unet from {config.pretrained_base_model_path} ==>")  
        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")
        # Denosing Unet
        print(f"DANCEPIPELINE.__INIT__ Constructing denoising_unet from {config.pretrained_base_model_path} and {config.motion_module_path} ==>")  
        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")
        # PoseGuider
        print(f"DANCEPIPELINE.__INIT__ Constructing pose_guider ==>")   
        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )
        # Image Encoder
        print(f"DANCEPIPELINE.__INIT__ Constructing image_enc from {config.image_encoder_path} ==>")  
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")
        # Scheduler
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)
        # load pretrained weights
        print(f"DANCEPIPELINE.__INIT__ Loading denoising_unet from {config.denoising_unet_path} ==>")  
        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        print(f"DANCEPIPELINE.__INIT__ Loading reference_unet from {config.reference_unet_path} ==>")   
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        print(f"DANCEPIPELINE.__INIT__ Loading pose_guider from {config.pose_guider_path} ==>")    
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
        )
        # Pipeline
        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to("cuda", dtype=weight_dtype)
        pipe = pipe.to("cuda", dtype=weight_dtype)
        self.pipe = pipe
        
        self.m1 = config.pose_guider_path.split('.')[0].split('/')[-1]
        self.m2 = config.motion_module_path.split('.')[0].split('/')[-1]
        # infer params        
        self.params = params_cfg.Gen
        self.pose_args = params_cfg.Pose
        
        # init for pose detector
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = DWposeDetector(
            det_config = config.yolox_config, 
            det_ckpt = config.yolox_ckpt,
            pose_config = config.dwpose_config, 
            pose_ckpt = config.dwpose_ckpt, 
            keypoints_only=False
        )    
        self.detector = detector.to(device)
        
        print(f"DANCEPIPELINE.__INIT__  Done. ==>")  
    
    def get_aligned_pose(self, ref_image_pil: Image.Image, video_path: str, save_path: str="./output/pose"):
        vidfn=video_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        outfn=os.path.join(save_path, "pose_vis.mp4")
        
        video = cv2.VideoCapture(vidfn)
        width= video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height= video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
        total_frame= video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps= video.get(cv2.CAP_PROP_FPS)

        print("height:", height)
        print("width:", width)
        print("fps:", fps)

        H_in, W_in  = height, width
        H_out, W_out = size_calculate(H_in, W_in, self.pose_args.detect_resolution) 
        H_out, W_out = size_calculate(H_out, W_out, self.pose_args.image_resolution) 

        refer_img = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        
        output_refer, pose_refer = self.detector(
            refer_img, 
            detect_resolution=self.pose_args.detect_resolution, 
            image_resolution=self.pose_args.image_resolution, 
            output_type='cv2',return_pose_dict=True
        )
        body_ref_img  = pose_refer['bodies']['candidate']
        hands_ref_img = pose_refer['hands']
        faces_ref_img = pose_refer['faces']
        output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)
        
        skip_frames = self.pose_args.align_frame
        max_frame = self.pose_args.max_frame
        pose_list, video_frame_buffer, video_pose_buffer = [], [], []

        cap = cv2.VideoCapture('2.mp4')     # 读取视频
        while cap.isOpened():               # 当视频被打开时：
            ret, frame = cap.read()         # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
            if ret:                         # 若是读取成功
                cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
                key = cv2.waitKey(25)       # 等待一段时间，并且检测键盘输入
                if key == ord('q'):         # 若是键盘输入'q',则退出，释放视频
                    cap.release()           # 释放视频
                    break
            else:
                cap.release()
        cv2.destroyAllWindows()             # 关闭所有窗口

        for i in range(max_frame):
            ret, img = video.read()
            if img is None: 
                break 
            else: 
                if i < skip_frames:
                    continue           
                video_frame_buffer.append(img)

            # estimate scale parameters by the 1st frame in the video
            if i==skip_frames:
                output_1st_img, pose_1st_img = self.detector(
                    img, self.pose_args.detect_resolution, 
                    self.pose_args.image_resolution, 
                    output_type='cv2', 
                    eturn_pose_dict=True
                )
                body_1st_img  = pose_1st_img['bodies']['candidate']
                hands_1st_img = pose_1st_img['hands']
                faces_1st_img = pose_1st_img['faces']

                '''
                计算逻辑:
                1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
                2. 用点在图中的实际坐标来计算。
                3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
                4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
                注意：dwpose 输出是 (w, h)
                '''
                
                # h不变，w缩放到原比例
                ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
                ref_ratio = ref_W / ref_H
                body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
                hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
                faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

                video_ratio = width / height
                body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
                hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
                faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

                # scale
                align_args = dict()
                
                dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
                dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
                align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

                dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
                dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
                align_args["scale_face"] = dist_ref_img / dist_1st_img

                dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
                dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
                align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

                dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
                dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
                s1 = dist_ref_img / dist_1st_img
                dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
                dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
                s2 = dist_ref_img / dist_1st_img
                align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

                dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
                dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
                s1 = dist_ref_img / dist_1st_img
                dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
                dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
                s2 = dist_ref_img / dist_1st_img
                align_args["scale_arm_lower"] = (s1+s2)/2

                # hand
                dist_1st_img = np.zeros(10)
                dist_ref_img = np.zeros(10)      
                
                dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
                dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
                dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
                dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
                dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
                dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
                dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
                dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
                dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
                dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

                dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
                dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
                dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
                dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
                dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
                dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
                dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
                dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
                dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
                dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

                ratio = 0   
                count = 0
                for i in range (10): 
                    if dist_1st_img[i] != 0:
                        ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
                        count = count + 1
                if count!=0:
                    align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
                else:
                    align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

                # body 
                dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
                dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
                align_args["scale_body_len"]=dist_ref_img / dist_1st_img

                dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
                dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
                s1 = dist_ref_img / dist_1st_img
                dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
                dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
                s2 = dist_ref_img / dist_1st_img
                align_args["scale_leg_upper"] = (s1+s2)/2

                dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
                dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
                s1 = dist_ref_img / dist_1st_img
                dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
                dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
                s2 = dist_ref_img / dist_1st_img
                align_args["scale_leg_lower"] = (s1+s2)/2

                ####################
                ####################
                # need adjust nan
                for k,v in align_args.items():
                    if np.isnan(v):
                        align_args[k]=1

                # centre offset (the offset of key point 1)
                offset = body_ref_img[1] - body_1st_img[1]
            
        
            # pose align
            pose_img, pose_ori = self.detector(
                img, 
                self.pose_args.detect_resolution, 
                self.pose_args.image_resolution, 
                output_type='cv2', 
                return_pose_dict=True
            )
            video_pose_buffer.append(pose_img)
            pose_align = align_img(
                img, 
                pose_ori, 
                align_args, 
                self.pose_args.detect_resolution, 
                self.pose_args.image_resolution
            )
            
            # add centre offset
            pose = pose_align
            pose['bodies']['candidate'] = pose['bodies']['candidate'] + offset
            pose['hands'] = pose['hands'] + offset
            pose['faces'] = pose['faces'] + offset


            # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
            pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] / ref_ratio
            pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
            pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
            pose_list.append(pose)

        # stack
        body_list  = [pose['bodies']['candidate'][:18] for pose in pose_list]
        body_list_subset = [pose['bodies']['subset'][:1] for pose in pose_list]
        hands_list = [pose['hands'][:2] for pose in pose_list]
        faces_list = [pose['faces'][:1] for pose in pose_list]
    
        body_seq         = np.stack(body_list       , axis=0)
        body_seq_subset  = np.stack(body_list_subset, axis=0)
        hands_seq        = np.stack(hands_list      , axis=0)
        faces_seq        = np.stack(faces_list      , axis=0)


        # concatenate and paint results
        H = 768 # paint height
        W1 = int((H/ref_H * ref_W)//2 *2)
        W2 = int((H/height * width)//2 *2)
        result_demo = [] # = Writer(args, None, H, 3*W1+2*W2, outfn, fps)
        result_pose_only = [] # Writer(args, None, H, W1, args.outfn_align_pose_video, fps)
        for i in range(len(body_seq)):
            pose_t={}
            pose_t["bodies"]={}
            pose_t["bodies"]["candidate"]=body_seq[i]
            pose_t["bodies"]["subset"]=body_seq_subset[i]
            pose_t["hands"]=hands_seq[i]
            pose_t["faces"]=faces_seq[i]

            ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
            ref_img = cv2.resize(ref_img, (W1, H))
            ref_pose= cv2.resize(output_refer, (W1, H))
            
            output_transformed = draw_pose(
                pose_t, 
                int(H_in*1024/W_in), 
                1024, 
                draw_face=False,
                )
            output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
            output_transformed = cv2.resize(output_transformed, (W1, H))
            
            video_frame = cv2.resize(video_frame_buffer[i], (W2, H))
            video_pose  = cv2.resize(video_pose_buffer[i], (W2, H))

            res = np.concatenate([ref_img, ref_pose, output_transformed, video_frame, video_pose], axis=1)
            result_demo.append(res)
            result_pose_only.append(output_transformed)

        print(f"pose_list len: {len(pose_list)}")
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
        clip.write_videofile(outfn, fps=fps)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_only, fps=fps)
        outfn_align_pose_video = os.path.join(save_path, "aligned_pose.mp4")
        clip.write_videofile(outfn_align_pose_video, fps=fps)
        print('pose align done')
        
    def infer(self, ref_image_pil: Image.Image, pose_video_path, seed=None, ref_name="test"):
        if seed is None:
            seed = random.randint(0, 2**32-1)
        generator = torch.manual_seed(seed)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        args = self.params
        width, height = self.params.W, self.params.H 
        print ('handle===', pose_video_path)
        pose_name = Path(pose_video_path).stem.replace("_kps", "")

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        L = min(args.L, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        original_width,original_height = 0,0

        pose_images = pose_images[::args.skip+1]
        print("processing length:", len(pose_images))
        src_fps = src_fps // (args.skip + 1)
        print("fps", src_fps)
        L = L // ((args.skip + 1))
        
        for pose_image_pil in pose_images[: L]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
            original_width, original_height = pose_image_pil.size
            pose_image_pil = pose_image_pil.resize((width,height))

        # repeart the last segment
        last_segment_frame_num =  (L - args.S) % (args.S - args.O) 
        repeart_frame_num = (args.S - args.O - last_segment_frame_num) % (args.S - args.O) 
        for i in range(repeart_frame_num):
            pose_list.append(pose_list[-1])
            pose_tensor_list.append(pose_tensor_list[-1])

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = self.pipe(
            ref_image_pil,
            pose_list,
            width,
            height,
            len(pose_list),
            args.steps,
            args.cfg,
            generator=generator,
            context_frames=args.S,
            context_stride=1,
            context_overlap=args.O,
        ).videos

        save_dir_name = f"{time_str}-{args.cfg}-{self.m1}-{self.m2}"
        save_dir = Path(f"./output/video-{date_str}/{save_dir_name}")
        save_dir.mkdir(exist_ok=True, parents=True)

        result = scale_video(video[:,:,:L], original_width, original_height)
        save_videos_grid(
            result,
            f"{save_dir}/{ref_name}_{pose_name}_{args.cfg}_{args.steps}_{args.skip}.mp4",
            n_rows=1,
            fps=src_fps if args.fps is None else args.fps,
        )    

        video = torch.cat([ref_image_tensor, pose_tensor[:,:,:L], video[:,:,:L]], dim=0) 
        video = scale_video(video, original_width, original_height)     
        save_videos_grid(
            video,
            f"{save_dir}/{ref_name}_{pose_name}_{args.cfg}_{args.steps}_{args.skip}_{self.m1}_{self.m2}.mp4",
            n_rows=3,
            fps=src_fps if args.fps is None else args.fps,
        )
        