import csv

import numpy as np
import soundfile as sf
from onnxruntime.quantization.calibrate import CalibrationDataReader
from transformers import Speech2TextFeatureExtractor

from utils.converter import convert_model_to_float16
from utils.optimizer import optimize_model
from utils.preprocessing import trim_audio
from utils.quantizer import quantize_model_with_static, quantize_model_with_dynamic


class DataReader(CalibrationDataReader):
    def __init__(self):
        self.extractor = Speech2TextFeatureExtractor()

        with open('data/val_set.csv', 'r') as fp:
            reader = csv.reader(fp)
            self.files = [str(row[0]) for row in reader]
        self.files = iter(self.files[:100])

    def get_next(self):
        file = next(self.files, None)
        if file:
            audio, _ = sf.read('data/' + file)
            audio = trim_audio(audio, trim_length=100_000)
            features = self.extractor(audio, sampling_rate=16_000, return_tensors='np')
            inputs = {'features': features.input_features.astype(np.float32),
                      'attention_mask': features.attention_mask.astype(np.int64)}
            return inputs
        else:
            return None


model_path = 'models/spoken-language-classifier-50/best.onnx'

# Optimize onnx model
optimzed_model_path = optimize_model(model_path, opt_level=99)

# Convert onnx model to float16
model_float = convert_model_to_float16(model_path)

# Quantize onnx model
dr = DataReader()
quantize_model_with_static(model_path, dr, op_types=['MatMul'])
quantize_model_with_dynamic(model_path)
