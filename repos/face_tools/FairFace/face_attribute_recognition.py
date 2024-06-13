# The implementation is based on FairFace, available at
# https://github.com/dchen236/FairFace
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


class FaceAttributeRecognition(nn.Module):

    def __init__(self, weight_dir, device='cuda'):
        super().__init__()
        model_path = os.path.join(weight_dir, 'FairFace') if 'FairFace' not in weight_dir else weight_dir
        model_path = os.path.join(model_path, 'pytorch_model.pt')
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.cfg_path = model_path.replace('pytorch_model.pt',
                                           'configuration.json')
        fair_face = torchvision.models.resnet34(pretrained=False)
        fair_face.fc = nn.Linear(fair_face.fc.in_features, 18)
        self.net = fair_face
        self.load_model()
        self.net = self.net.to(device)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        male_list = ['Male', 'Female']
        age_list = [
            '0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69',
            '70+'
        ]
        self.map_list = [male_list, age_list]

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict, strict=True)
        self.net.eval()

    def forward(self, img):
        """ FariFace model forward process.

        Args:
            img: [h, w, c]

        Return:
            list of attribute result: [gender_score, age_score]
        """
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img = np.array(img)
        else:
            img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        inputs = self.trans(img)

        c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to(self.device)
        inputs = Variable(inputs, volatile=True)
        outputs = self.net(inputs)[0]

        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        gender_score = F.softmax(gender_outputs).detach().cpu().tolist()
        age_score = F.softmax(age_outputs).detach().cpu().tolist()

        return [gender_score, age_score], self.map_list
