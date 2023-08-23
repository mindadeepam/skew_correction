import matplotlib.pyplot as plt
from skew_correction.helper import read_raw_image, get_images_in_dir, pil2np, np2pil, tensor2pil, get_skew, hough_transform, remove_padding
from skew_correction.constants import root_dir, angle2label, label2angle, device, model_url
from skew_correction.data import tensor_transform
from skew_correction.model import TimmClassifier, ensure_model
from time import time
from ds_utils.gcp_utils import download_file_url_from_gcp_to_tempdir
import os

# model=None
# def ensure_model():
#     """
#     ensures we have a model loaded. add script to download from gcp if model is not available at path.
#     """
#     model_path = f"/var/tmp/{model_url.split('/')[-1]}"
#     if not os.path.exists(model_path):
#         model_path = download_file_url_from_gcp_to_tempdir(model_url)

#     global model
#     if model==None:
#         model = TimmClassifier('mobilenetv3_large_100', pretrained=False, num_classes=4, in_chans=1)
#         model.load(model_path)
#         model.to(device)
#     return model

def rectify_image(path=None, tensor=None, debug=False, model=None):
    """
    function that takes in one image and corrects its orientation.
    add ability to pass url, links too

    returns fixed_image tensor

    """
    if model==None:
        model = ensure_model()

    assert (path!=None) or (tensor!=None), "pass one of path or tensor"
    if tensor:
        org_img = tensor2pil(tensor)
    else:
        org_img = read_raw_image(path)

    # classical rectify
    t0 = time()

    img, angle = hough_transform(org_img)
    print(f"classical rectify done in {round(time() - t0, 2)}s")

    # model prediction
    t = time()
    tensor = tensor_transform(img)
    pred_class = model.predict(tensor)
    pred_class = pred_class.numpy()[0]
    print(f"model prediction done in {round(time() - t, 2)}s")

    # final rectify
    t = time()
    fixed_img = img.rotate(360-label2angle[pred_class], expand=True)
    fixed_img = remove_padding(fixed_img)
    print(f"final rectify done in {round(time() - t, 2)}s")

    print(f"Total skew_correction done in {round(time() - t0, 2)}s")
    print(f"========================================================")

    if debug==True:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 16))
        axes[0].imshow(pil2np(org_img))
        axes[0].set_title("original image")
        axes[1].imshow(pil2np(img))
        axes[1].set_title(f"after hough {round(angle, 2)}")
        axes[2].imshow(tensor.numpy().transpose(1, 2, 0))
        axes[2].set_title(f"model says this img is : {pred_class}")
        axes[3].imshow(pil2np(fixed_img))
        axes[3].set_title(f"final image")
        plt.tight_layout()

        # Display the figure with subplots
        plt.show()

    return fixed_img