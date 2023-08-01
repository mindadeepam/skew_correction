from skew_correction.data import DatasetClass
from torch.utils.data import DataLoader
from skew_correction.helper import read_raw_image, get_images_in_dir, extract_numbers_from_end
from skew_correction.constants import root_dir
from skew_correction.model import TimmClassifier, get_acc
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix



def df_from_dir(dirpath):
    """
    creates a csv file for testing using filename, path and label are extracted from path_label.jpg
    returns filepath
    """
    filepaths = get_images_in_dir(dirpath, return_path=True)
    csv_data = []
    for filepath in filepaths:
        label = extract_numbers_from_end(filepath.split('.')[-2])
        csv_data.append((filepath, label))

    df = pd.DataFrame(csv_data, columns = ['path', 'label'])
    filepath = os.path.join(root_dir,"data/test.csv")
    df.to_csv(filepath, index=None)

    return filepath
# dataset = DatasetClass(file=filepath, split='predict')



def test(model, dirpath=None, filepath=None, batch_size=4, limit=None):

    if (not dirpath) and (not filepath):
        print("provide one of dirpath or filepath")
        return None
    if dirpath:
        filepath = df_from_dir(dirpath)

    perdict_dataset = DatasetClass(file=filepath)
    dataloader = DataLoader(perdict_dataset, batch_size=2)
    
    labels_list = []
    preds_list = []
    for pixels, labels in dataloader:
        preds = model.predict(pixels)
        labels_list.extend(labels.numpy())
        preds_list.extend(preds.numpy())
        limit -= batch_size
        if limit<=0:
            break

    accuracy = (np.array(labels_list)== np.array(preds_list)).sum()/ len(labels_list)
    cm = confusion_matrix(labels_list, preds_list)
    print("Confusion Matrix:")
    print(cm)
    print("accuracy: \n",accuracy)
    return accuracy


if __name__=='__main__':
    test()



