import re
import cv2
import glob
import numpy as np
import pydicom as dcm

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model

from efficientnet.tfkeras import EfficientNetB3

from settings import BATCH_SIZE, SHAPE, MODEL_PATH

from utils import window


class Model:
    """
    The model class for ai-corona.
    """
    def __init__(self) -> None:
        """
        Initialize the class.
        """
        self.batch_size = BATCH_SIZE
        self.shape = SHAPE
        self.img_input = Input(shape=(SHAPE, SHAPE, 3))
        self.efficient_net_b3 = EfficientNetB3(include_top=False,
                                               input_shape=(self.shape,
                                                            self.shape, 3),
                                               input_tensor=self.img_input,
                                               weights='imagenet',
                                               classes=14,
                                               pooling='avg')
        self.model = load_model(MODEL_PATH)

    def predict(self, case_directory_path: str) -> dict:
        """
        Main prediction (diagnosis) function.

        Args:
            case_directory_path (str): Directory path of the case
                to be diagnosed.

        Returns:
            dict: Prediction (diagnosis) results.
        """
        ls_dire = glob.glob(case_directory_path + '/*.dcm')
        ls_dire = [[s, int(re.findall(r'\d+', s)[-1])] for s in ls_dire]
        ls_dire = sorted(ls_dire, key=lambda x: x[1])

        if len(ls_dire) == 0:
            ls_dire = glob.glob(case_directory_path + '/*.DCM')
            ls_dire = [[s, int(re.findall(r'\d+', s)[-1])] for s in ls_dire]
            ls_dire = sorted(ls_dire, key=lambda x: x[1])

        ls_dire = np.array(ls_dire)[:, 0]

        ct_data = []
        for s in ls_dire:
            slice_0 = dcm.dcmread(s)
            slice_0 = np.array(slice_0.pixel_array, float)
            ct_data.append(cv2.resize(slice_0, (512, 512)))
        ct_data = np.array(ct_data)
        ct_data = window(ct_data)

        p = self.efficient_net_b3.predict(preprocess_input(ct_data),
                                          batch_size=self.batch_size)

        k = 0.5
        x = len(p)
        ss = int(len(p) * k)

        p = np.mean(p[int((x - ss) / 2) + 1:-int((x - ss) / 2) + 1], axis=0)
        pred = np.round(self.model.predict(np.expand_dims(p, axis=0))[0],
                        4) * 100

        diagnosis = {
            "n": pred[0],
            "p": pred[1],
            "c": pred[2],
        }

        return diagnosis
