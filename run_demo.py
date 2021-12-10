import librosa as lr
import numpy as np
import onnxruntime
import yaml
from easydict import EasyDict
from gradio.inputs import Audio
from gradio.interface import Interface
from gradio.outputs import Label
from scipy.special import softmax
from transformers import Speech2TextFeatureExtractor

from utils.preprocessing import trim_audio

with open('config.yml', 'r') as fp:
    config = EasyDict(yaml.load(fp, Loader=yaml.FullLoader))

languages = ['Swiss German', 'English', 'Italian', 'French', 'German']

extractor = Speech2TextFeatureExtractor()
ort_session = onnxruntime.InferenceSession(config['TESTING']['model_path'])


def predict(audio_from_upload=None, audio_from_mic=None):
    if audio_from_mic is not None:
        sr, audio = audio_from_mic
    elif audio_from_upload is not None:
        sr, audio = audio_from_upload
    else:
        return 'Must provide input before submit'

    audio = audio.astype(float)
    if len(audio.shape) == 2:
        if audio.shape[1] == 2:
            audio = np.swapaxes(audio, 0, 1)
        audio = lr.to_mono(audio)

    audio = trim_audio(audio, trim_length=config['DATA']['trim_length'])
    features = extractor(audio, sampling_rate=config['DATA']['sampling_rate'], return_tensors='np')

    inputs = {'features': features.input_features.astype(np.float32),
              'attention_mask': features.attention_mask.astype(np.int64)}
    output = ort_session.run(None, inputs)
    probabilities = softmax(output[0], axis=1).squeeze().tolist()
    print(probabilities)
    return {label: confidence for label, confidence in zip(languages, probabilities)}


if __name__ == "__main__":
    demo = Interface(
        fn=predict,
        title="Spoken Language Classification",
        description='Drop an audio or record one to recognize the language',
        inputs=[Audio(source='upload', optional=True),
                Audio(source='microphone', optional=True)],
        outputs=Label(num_top_classes=len(languages)),
        examples='./demo_data/',
        examples_per_page=3,
        theme='darkhuggingface',
        allow_flagging=True,
        flagging_dir='demo_data/flagged',
        allow_screenshot=False,
        live=True
    )
    demo.launch(debug=False, share=True)
