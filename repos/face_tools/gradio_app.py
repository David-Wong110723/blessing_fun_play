import warnings
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

from Retinaface import FaceDetector
from AgeGender import AgeGenderEstimator
from FairFace import FaceAttributeRecognition


weight_dir = "/cfs/zhlin/sd_models/extra_models/face_tools"
face_detector = FaceDetector(weight_path=weight_dir)
age_gender_estimator = AgeGenderEstimator(weight_path=weight_dir)
attribute_recognizer = FaceAttributeRecognition(weight_dir=weight_dir)


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

def detect(image_raw):
    image = image_raw[:,:,::-1].copy()  # RGB -> BGR
    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    if len(faces) > 0:
        genders, ages = age_gender_estimator.detect(faces)
        vis_image = draw_faces(image_raw, boxes, scores, landmarks, genders, ages)
        
        for face in faces:
            att_scores, att_names = attribute_recognizer(face)
            print(f"scores: {att_scores}")
            print(f"names: {att_names}")
    else:
        vis_image = image

    return vis_image

if __name__ == "__main__":
    import gradio as gr
    input = gr.Image(label="输入图片", show_label=True, elem_id="input_image", source="upload", type="numpy", image_mode="RGB")
    output = gr.Image(label="结果图片", show_label=True, elem_id="output_image", type="numpy", image_mode="RGB")
    interface = gr.Interface(fn=detect, inputs=[input], outputs=[output],title="人脸检测Demo")
    interface.launch(server_name="0.0.0.0", server_port=8899)
