import torch
import torchvision.transforms as transforms
import os

model_url = "https://storage.cloud.google.com/fmt-ds-bucket/document_aligment_correction/skew_correction/models/mobilenetv3_large_100-acc_93.pth"
model_path = "/tmp/mobilenetv3_large_100-acc_93.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/".join(os.getcwd().split("/")[:-1])

angle2label = {0:0, 90:1, 180:2, 270:3}
label2angle = {value: key for key, value in angle2label.items()}

train_params = {"batch_size": 8,
         "num_workers": 0,
         "shuffle": True}

test_params = {
        "batch_size": 8,
         "num_workers": 0,
         "shuffle": False}

#prediction transformation
prediction_transform = transforms.Compose([
                transforms.Resize((500, 500)),
                transforms.ToTensor()
            ])
