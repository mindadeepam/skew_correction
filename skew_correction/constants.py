import torch
import torchvision.transforms as transforms
import os

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
