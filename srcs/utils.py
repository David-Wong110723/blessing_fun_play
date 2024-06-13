import io

import base64
import PIL
from PIL import PngImagePlugin, Image, ImageFont
import cv2
import numpy as np
import random

import datetime
import piexif
from fastapi import HTTPException


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = int(now.timestamp())
    return timestamp

# draw for debug
random.seed(142857)

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
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

############################################## Read Iamge ##############################################
def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img

def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im

def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

############################################## Image to Base64 && Base64 to Image ##############################################
def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    #try:
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))
    return image
    #except Exception as err:
    #    raise HTTPException(status_code=500, detail="Invalid encoded image")

def encode_pil_to_base64(image, image_format='png', jpeg_quality=80):
    with io.BytesIO() as output_bytes:

        if image_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=jpeg_quality)

        elif image_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if image_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()
 
    return base64.b64encode(bytes_data)

def base64_to_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image

############################################## Image to numpy.ndarray && np.ndarray to Image ##############################################
def np2pil(nparray, img_type="RGB"):
    assert(img_type in ["RGB","RGBA"])
    return Image.fromarray(nparray.astype(np.uint8)).convert(img_type)

def pil2np(image, img_type=np.uint8):
    assert(img_type in [np.uint8, np.float32])
    return np.array(image, img_type)

############################################## Resize Image aligned or cropped to 64*N ##############################################
def calc_size(ori_size, ratio):
    new_size = int(ori_size * ratio)
    if new_size % 2 == 1:
        new_size = new_size + 1
    tar_size = (new_size // 64) * 64
    pad_1 = (new_size - tar_size) // 2
    pad_2 = new_size - pad_1
    return new_size, tar_size, pad_1, pad_2

def resize_align(image, mask=None, short_size=512, long_size=0):
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]
    assert(short_size > 0 or long_size > 0)
    
    new_width, new_height = 0, 0
    if short_size > 0:
        if width > height:
            new_height = short_size
            new_width = int(width * new_height / height)
        else:
            new_width = short_size
            new_height = int(height * new_width / width)
    else:
        if width > height:
            new_width = long_size
            new_height = int(height * new_width / width)
        else:
            new_height = long_size
            new_width = int(width * new_height / height)
    
    if isinstance(image, PIL.Image.Image):
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        image_resized = np.clip(image_resized, 0, 255)

    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_resized = mask.resize((new_width, new_height), Image.NEAREST)
        else:
            mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized, new_width, new_height
    else:
        return image_resized, new_width, new_height
    
def resize_align64(image, mask=None, short_size=512, long_size=0):
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]
    assert(short_size > 0 or long_size > 0)
    new_width, new_height = 0, 0
    left, top, right, bottom = 0, 0, 0, 0
    if short_size > 0:
        if width > height:
            new_height = short_size
            new_width = int((width * new_height / height) / 64) * 64
        else:
            new_width = short_size
            new_height = int((height * new_width / width) / 64) * 64
    else:
        if width > height:
            new_width = long_size
            new_height = int((height * new_width / width) / 64) * 64
        else:
            new_height = long_size
            new_width = int((width * new_height / height) / 64) * 64
    
    if isinstance(image, PIL.Image.Image):
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        image_resized = np.clip(image_resized, 0, 255)

    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_resized = mask.resize((new_width, new_height), Image.NEAREST)
        else:
            mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized, new_width, new_height
    else:
        return image_resized, new_width, new_height

def resize_crop64(image, mask=None, short_size=512, long_size=0):
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]
    assert(short_size > 0 or long_size > 0)
    new_width, new_height = 0, 0
    left, top, right, bottom = 0, 0, 0, 0
    if short_size > 0:
        if width > height:
            new_height = short_size
            tar_height = new_height
            top, bottom = 0, new_height
            new_width, tar_width, left, right = calc_size(width, new_height / height)
        else:
            new_width = short_size
            tar_width = new_width
            left, right = 0, new_width
            new_height, tar_height, top, bottom = calc_size(height, new_width / width)
    else:
        if width > height:
            new_width = long_size
            tar_width = new_width
            left, right = 0, new_width
            new_height, tar_height, top, bottom = calc_size(height, new_width / width)
        else:
            new_height = long_size
            tar_height = new_height
            top, bottom = 0, new_height
            new_width, tar_width, left, right = calc_size(width, new_height / height)
    
    if isinstance(image, PIL.Image.Image):
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        image_resized = image_resized.crop((left, top, right, bottom))
    else:
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        image_resized = np.clip(image_resized, 0, 255)
        image_resized = image_resized[top:bottom, left:right]

    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_resized = mask.resize((new_width, new_height), Image.NEAREST)
            mask_resized = mask_resized.crop((left, top, right, bottom))
        else:
            mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized[top:bottom, left:right]
        return image_resized, mask_resized, tar_width, tar_height
    else:
        return image_resized, tar_width, tar_height

def resize_coord(x, y, src_size, tar_size):
    x = x * (tar_size[0] - 1) / (src_size[0] - 1)
    y = y * (tar_size[1] - 1) / (src_size[1] - 1)
    return x, y

############################################## Operations of Bounding Boxes ##############################################
def pad_bbox(bbox, padding_ratios, img_width, img_height, to_square=False):
    assert(len(padding_ratios) == 2 or len(padding_ratios) == 4, f"length of narrow ratios only support 2 or 4, {len(padding_ratios)} given.")
    if len(padding_ratios) == 2:
        padding_ratios = [padding_ratios[0], padding_ratios[1], padding_ratios[0], padding_ratios[1]]
    # bbox [left, top, right, bottom]
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]

    pad_left = width * padding_ratios[0]
    pad_top = height * padding_ratios[1]
    pad_right = width * padding_ratios[2]
    pad_bottom = height * padding_ratios[3]
    
    tar_width = width + pad_left + pad_right
    tar_height = height + pad_top + pad_bottom
    if to_square:
        if tar_height > tar_width:
            delta_width = (tar_height - tar_width) / 2
            pad_left = pad_left + delta_width
            pad_right = pad_right + delta_width
        else:
            delta_height = (tar_width - tar_height) / 2
            pad_top = pad_top + delta_height
            pad_bottom = pad_bottom + delta_height
    
    left = int(max(bbox[0] - pad_left, 0))
    top = int(max(bbox[1] - pad_top, 0))
    right = int(min(bbox[2] + pad_right, img_width - 1))
    bottom = int(min(bbox[3] + pad_bottom, img_height - 1))

    return [left, top, right, bottom]

def pad_boxes(bboxes, padding_ratios, img_width, img_height, to_square=False):
    # bboxes, list of bbox [left, top, right, bottom]
    if len(bboxes) == 0:
        return bboxes

    _bboxes = []
    for idx, bbox in enumerate(_bboxes):
        _bbox = pad_bbox(bbox, padding_ratios, img_width, img_height, to_square)
        _bboxes.append(_bbox)
    return _bboxes

def narrow_bbox(bbox, narrow_ratios, to_square=False):
    assert(len(narrow_ratios) == 2 or len(narrow_ratios) == 4, f"length of narrow ratios only support 2 or 4, {len(narrow_ratios)} given.")
    # bbox [left, top, right, bottom]
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    narrow_ratios = [0.49 if ratio > 0.49 else ratio for ratio in narrow_ratios]
    if len(narrow_ratios) == 2:
        narrow_ratios = [narrow_ratios[0], narrow_ratios[1], narrow_ratios[0], narrow_ratios[1]]

    nar_left = width * narrow_ratios[0]
    nar_top = height * narrow_ratios[1]
    nar_right = width * narrow_ratios[2]
    nar_bottom = height * narrow_ratios[3]
    
    tar_width = width - nar_left - nar_right
    tar_height = height - nar_top - nar_bottom
    if to_square:
        if tar_height > tar_width:
            delta_height = (tar_height - tar_width) / 2
            nar_top = nar_top + delta_height
            nar_bottom = nar_bottom + delta_height
        else:
            delta_width = (tar_width - tar_height) / 2
            nar_left = nar_left + delta_width
            nar_right = nar_right + delta_width
    
    left = int(bbox[0] + nar_left)
    top = int(bbox[1] + nar_top)
    right = int(bbox[2] - nar_right)
    bottom = int(bbox[3] - nar_bottom)

    return [left, top, right, bottom]

def bbox2mask(bboxes, width, height, inverse=False):
    mask = np.zeros((height, width)).astype(np.uint8)
    if len(bboxes) > 0:
        for bbox in bboxes:
            mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
    if inverse:
        mask = 255 - mask
    return np2pil(mask, "RGB")

def mask2bbox(mask):
    if isinstance(mask, PIL.Image.Image):
        msk = pil2np(mask, np.float32)[:,:,0]
    else:
        msk = mask.copy()
    max_value = np.max(msk)
    msk[msk < (max_value / 2)] = 0
    # inputs: numpy arrays
    non_zero_coords = np.nonzero(msk)
    x_min = min(non_zero_coords[1])
    y_min = min(non_zero_coords[0])
    x_max = max(non_zero_coords[1])
    y_max = max(non_zero_coords[0])
    return [x_min, y_min, x_max, y_max]

############################################## Padding && Crop ##############################################
def pad_image(image, mask=None, ratios=[0, 0, 0, 0], pad_color=255):
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
        image_np = pil2np(image, np.uint8)
    else:
        height, width, _ = image.shape
        image_np = image
    '''
    if width_ratio == 0 and height_ratio == 0:
        if width > height:
            height_ratio = ((width / height) - 1)*3/8
        else:
            width_ratio = ((height / width) - 1)*3/8
    '''

    left_pad = int(width * ratios[0])
    right_pad = int(width * ratios[2])
    upper_pad = int(height * ratios[1])
    lower_pad = int(height * ratios[3])
    x_min = left_pad
    y_min = upper_pad
    x_max = left_pad + width - 1
    y_max = upper_pad + height - 1

    image_pad = np.ones((height + upper_pad + lower_pad, width + left_pad + right_pad, image_np.shape[2]), np.uint8)
    image_pad[:,:] = pad_color
    image_pad[y_min:y_max+1, x_min:x_max+1] = image_np

    if isinstance(image, PIL.Image.Image):
        image_pad = np2pil(image_pad, "RGB")

    if mask is not None:
        if isinstance(mask, PIL.Image.Image):
            mask_np = pil2np(mask, np.uint8)
        else:
            mask_np = mask
        if mask_np.ndim == 3:
            mask_pad = np.zeros((height + upper_pad + lower_pad, width + left_pad + right_pad, mask_np.shape[2]), np.uint8)
        else:
            mask_pad = np.zeros((height + upper_pad + lower_pad, width + left_pad + right_pad), np.uint8)
        mask_pad[y_min:y_max+1, x_min:x_max+1] = mask_np
        if isinstance(mask, PIL.Image.Image):
            mask_pad = np2pil(mask_pad, "RGB")
        return image_pad, mask_pad, [x_min, y_min, x_max, y_max]
    else:
        return image_pad, [x_min, y_min, x_max, y_max]

def crop_image(image, bbox, mask=None):
    if isinstance(image, PIL.Image.Image):
        _image = image.crop((bbox[0], bbox[1], bbox[2]+1, bbox[3]+1))
    else:
        _image = image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
    if mask is None:
        return _image
    else:
        if isinstance(mask, PIL.Image.Image):
            _mask = mask.crop((bbox[0], bbox[1], bbox[2]+1, bbox[3]+1))
        else: 
            _mask = mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
        return _image, _mask

def centor_crop(image, tar_ratio):
    # ratio: target width/height
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    else:
        height, width, _ = image.shape
    
    left, top, right, bottom = 0, 0, width-1, height-1
    src_ratio = width / height
    if src_ratio > tar_ratio:
        tar_height = height
        tar_width = tar_height * tar_ratio
        left = (width - tar_width) // 2
        right = (width - 1) - (width - tar_width) // 2
    else:
        tar_width = width
        tar_height = tar_width / tar_ratio
        top = (height - tar_height) // 2
        bottom = (height - 1) - (height - tar_height) // 2

    image = crop_image(image, [left, top, right, bottom])
    return image        

def crop_coord(x, y, bbox):
    x = x - bbox[0]
    y = y - bbox[1]
    return x, y

############################################## Mask Operations ##############################################
def dilate(mask, size=(3,3), iterations=1):
    is_pil = False
    if isinstance(mask, PIL.Image.Image):
        mask = pil2np(mask, np.float32)
        is_pil = True
    _mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size), iterations=iterations)
    if is_pil:
        _mask = np2pil(mask, "RGB")
    return _mask

def erode(mask, size=(3,3), iterations=1):
    is_pil = False
    if isinstance(mask, PIL.Image.Image):
        mask = pil2np(mask, np.float32)
        is_pil = True
    _mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size), iterations=iterations)
    if is_pil:
        _mask = np2pil(mask, "RGB")
    return _mask

def compose_mask(mask1, mask2, reverse=False):
    if isinstance(mask1, PIL.Image.Image):
        mask1 = pil2np(mask1, np.float32)[:,:,0]
    else:
        mask1 = mask1.astype(np.float32) * 255
    
    if isinstance(mask2, PIL.Image.Image):
        mask2 = pil2np(mask2, np.float32)[:,:,0]
    else:
        mask2 = mask2.astype(np.float32) * 255
    
    if reverse:
        mask2 = 255 - mask2
    mask = np2pil((mask1*mask2/255), "RGB")
    return mask

def paste(img, msk, img2, msk2, local=True):
    img1 = img.copy()
    msk1 = msk.copy()
    # inputs: numpy arrays
    non_zero_coords = np.nonzero(msk1)
    lt_x = min(non_zero_coords[1])
    lt_y = min(non_zero_coords[0])
    rb_x = max(non_zero_coords[1])
    rb_y = max(non_zero_coords[0])
    
    #tar_w = ((rb_x - lt_x + 1) // 64) * 64
    #tar_h = ((rb_y - lt_y + 1) // 64) * 64
    tar_w = rb_x - lt_x + 1
    tar_h = rb_y - lt_y + 1

    src_h, src_w = msk2.shape[:2]
    tar_hw_ratio = tar_h / tar_w
    src_hw_ratio = src_h / src_w
    left, top, right, bottom = 0, 0, 0, 0
    if tar_hw_ratio > src_hw_ratio:
        tar_h = int(src_hw_ratio * tar_w)
        #if tar_h % 2 == 1:
        #    tar_h = tar_h + 1
        #padded_h = (tar_h // 64) * 64
        padded_h = tar_h
        padded_w = tar_w
        left = 0
        top = tar_h - padded_h
        right = tar_w
        bottom = tar_h - top
    else:
        tar_w = int(tar_h / src_hw_ratio)
        #if tar_w % 2 == 1:
        #    tar_w = tar_w + 1
        #padded_w = (tar_w // 64) * 64
        padded_w = tar_w
        padded_h = tar_h
        left = tar_w - padded_w
        top = 0
        right = tar_w - left
        bottom = tar_h
    img2_resize = cv2.resize(img2, (tar_w, tar_h), interpolation=cv2.INTER_CUBIC)   # resize
    img2_resize = img2_resize[top:bottom, left:right]                               # crop
    img2_resize = np.clip(img2_resize, 0, 255)                                      # clip
    
    msk2_resize = cv2.resize(msk2, (tar_w, tar_h), interpolation=cv2.INTER_CUBIC)   # resize
    msk2_resize = msk2_resize[top:bottom, left:right]                               # crop
    msk2_resize = np.clip(msk2_resize, 0, 1)                                        # clip
    msk2_resize = msk2_resize[:, :, np.newaxis]
    
    lt_x = rb_x - padded_w + 1
    lt_y = rb_y - padded_h + 1
    tar_img = img1[lt_y:rb_y+1, lt_x:rb_x+1, :]
    tar_img = tar_img * (1 - msk2_resize) + img2_resize * msk2_resize
    tar_img = np.clip(tar_img, 0, 255)

    img1[lt_y:rb_y+1, lt_x:rb_x+1, :] = tar_img
    msk1[msk1 == 1] = 0
    msk1[lt_y:rb_y+1, lt_x:rb_x+1] = msk2_resize[:,:,0]

    if local:
        return tar_img, msk2_resize[:,:,0], (tar_w, tar_h), (top, bottom, left, right)
    else:
        return img1, msk1, (tar_w, tar_h), (top, bottom, left, right)

def paste_to_bg(image, mask, bg_image, bg_mask, local=False):
    if isinstance(image, PIL.Image.Image):
        image_np = pil2np(image, np.float32)
    else:
        image_np = image
    if isinstance(mask, PIL.Image.Image):
        mask_np = pil2np(mask, np.float32)[:, :, 0] / 255. # [0, 255] -> [0, 1]
    else:
        mask_np = mask
    
    if local:
        # erode
        mask_np = erode(mask_np, size=(3,3), iterations=1)
    
    image_fusion_np, mask_fusion_np, tar_size, tar_crop = paste(bg_image, bg_mask, image_np, mask_np, local)

    if local:
        # dilate
        mask_fusion_np = dilate(mask_fusion_np, size=(7,7), iterations=3)
    
    image_fusion_np = (image_fusion_np).astype(np.uint8)
    image_fusion = Image.fromarray(image_fusion_np)

    mask_fusion_np = (mask_fusion_np * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    mask_fusion = Image.fromarray(mask_fusion_np).convert('RGB')

    return image_fusion, mask_fusion, tar_size, tar_crop

def clip_coord(x, y, size):
    if x < 0:
        x = 0
    if x > size[0] - 1:
        x = size[0] - 1
    if y < 0:
        y = 0
    if y > size[1] - 1:
        y = size[1] - 1
    return x, y