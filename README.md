# Video Generation with Human
## Source Code
- api
- assets
- configs
- demos
- docs
- model_weights
- repos
- srcs
README.MD


## Platform TODO List
- [x] Framework 
- [x] Human+Face Analysis
- [x] Face+Music to Video
- [x] Motion to Video, update
- [x] Gradio demo
- [ ] Video Matching
- [ ] Video Clip
- [ ] Music to Motion
- [ ] Subtitle Generation
- [ ] Video Merge
- [ ] Result Storge
- [ ] Platform Building

## Assets TODO List
- [ ] Music Collactions
- [ ] Video segments Collactions
- [ ] Text Contents (TTS), ChatTTS


### Build environment

We recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:

```shell
pip install -r requirements.txt
```

### mmlab packages
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Download weights
You can download weights manually as follows:

1. Download trained [AniPortrait](https://huggingface.co/ZJYang/AniPortrait/tree/main), which include the following parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth`, `motion_module.pth`, `audio2mesh.pt`, `audio2pose.pt` and `film_net_fp16.pt`. 

2. Download trained [MusePose](https://huggingface.co/TMElyralab/MusePose).

3. Download the weights of other components:
   - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
   - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/unet)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [yolox](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) - Make sure to rename to `yolox_l_8x8_300e_coco.pth`
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)
   - [blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
   - [RetinaFace](https://huggingface.co/akhaliq/RetinaFace-R50), rename it to resnet50.pth
   - [cv_resnet18_human-detection](https://modelscope.cn/models/iic/cv_resnet18_human-detection/summary)
   - [grounded_sam](https://github.com/facebookresearch/segment-anything)

Finally, these weights should be organized in `model_weights` as follows:
```
./model_weights/
|-- face_tools
|   |-- Retinaface
|   |   └── resnet50.pth
|-- cv_resnet18_human-detection
|   |-- configuration.json
|   |-- mmcv_config.py
|   └── pytorch_model.pt
|-- grounded_sam
|   |-- sam_vit_b_01ec64.pth
|   |-- sam_vit_h_4b8939.pth
|   └── sam_vit_l_0b3195.pth
|-- blip-image-captioning-large
|   |-- config.json
|   |-- preprocessor_config.json
|   |-- pytorch_model.bin
|   |-- special_tokens_map.json
|   |-- tf_model.h5
|   |-- tokenizer_config.json
|   |-- tokenizer.json
|   └── vocab.txt
|-- AniPortrait
|   |-- wave2vec2-base-960h
|   |   |-- config.json
|   |   |-- feature_extractor_config.json
|   |   |-- model.safetensors
|   |   |-- preprocessor_config.json
|   |   |-- pytorch_model.bin
|   |   |-- special_tokens_map.json
|   |   |-- tf_model.h5
|   |   |-- tokenizer_config.json
|   |   └── vocab.json
|   |-- audio2mesh.pt
|   |-- audio2pose.pt
|   |-- denosing_unet.pth
|   |-- film_net_fp16.pt
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- MusePose
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.pth
|   └── yolox_l_8x8_300e_coco.pth
|-- stable-diffusion-v1-5
|   |-- feature_extractor
|   |   └── preprocessor_config.json
|   |-- model_index.json
|   |-- unet
|   |   |-- config.json
|   |   └── diffusion_pytorch_model.bin
|   └── v1-inference.yaml
|-- sd-image-variations-diffusers
|   └── unet
|       |-- config.json
|       └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin

```