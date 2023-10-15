import shutil
import cv2
import typing
import numpy as np
import os

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs


def startPrediction():
    current_directory = os.path.dirname(__file__)
    # first path to input_sentences folder to retrieve input csv file
    relative_path = os.path.join("..", "handwritten_to_digit",
                                 "input_sentences")
    path_input = os.path.abspath(os.path.join(current_directory,
                                              relative_path))

    # 2nd path to connectivity folder for output
    output_path = os.path.join("..", "..", "static", "prediction_text")
    path_connect = os.path.abspath(os.path.join(current_directory, output_path))

    # 3rd path to saved_model folder to retrieve config.yaml
    temp_path = os.path.join("saved_model", "configs.yaml")
    path_config = os.path.abspath(os.path.join(current_directory, temp_path))
    # path_config=os.path.abspath(os.path.join(current_directory, conf_path))

    # clear contents in output.txt
    open(os.path.join(path_connect, "output.txt"), "w").close()

    configs = BaseModelConfigs.load(path_config)

    model = ImageToWordModel(model_path=configs.model_path,
                             char_list=configs.vocab)

    df = pd.read_csv(os.path.join(path_input, "input_img_paths.csv")).values.tolist()

    # accum_cer, accum_wer = [], []
    f = open(os.path.join(path_connect, "output.txt"), "w")
    for image_path in tqdm(df):
        new_img = image_path[0]
        print(new_img)
        image = cv2.imread(new_img)
        prediction_text = model.predict(image)
        f.write(prediction_text + '\n')
        print("Prediction: ", prediction_text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # delete jpg files by empty
    shutil.rmtree(os.path.join(path_input, 'input_lines_in_jpg'))
    os.mkdir(os.path.join(path_input, 'input_lines_in_jpg'))
    f.close()
    # print(f"Average CER: {np.average(accum_cer)}, Average WER:
    # {np.average(accum_wer)}")


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image,
                                                             *self.input_shape[
                                                              :2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text



if __name__ == "__main__":
    startPrediction()

