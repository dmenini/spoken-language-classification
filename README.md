# Spoken Language Classification

This repository contains the code base to train and evaluate a spoken language classifier, as well as the runtime code
to classify the language of input audio files/recordings through a web interface.

## Install Requirements

To install the full project requirements:

1. create and activate your virtual environment
2. run `pip install -r requirements.txt` to install the requirements

In order to load audio files, the project requires Soundfile to be installed. SoundFile depends on the system library
libsndfile. In a modern Python, you can use `pip install soundfile` to download and install the latest release of
SoundFile and its dependencies. On Windows and OS X, this will also install the library libsndfile. On Linux, you need
to install libsndfile using your distribution’s package manager, for example `sudo apt-get install libsndfile1`.

### Download Huggingface Transformer

Our classifier uses Huggingface's Speech2TextModel transformer as encoder. In order to load the official pretrained
configuration, either

- specify `facebook/s2t-medium-mustc-multilingual-st` as `pretrained_encoder_path` in `config.yml` to download the model
  in the cache
- manually download the model from the Huggingface model Hub and then specify its path in `config.yml`. In this case,
  git-lfs is required to pull the model after having downloaded its pointer:

```
git clone https://huggingface.co/facebook/s2t-medium-mustc-multilingual-st
cd s2t-medium-mustc-multilingual-st
git lfs install
git lfs pull
``` 

## Web Interface

To easily demo our machine learning model, we use [Gradio](https://gradio.app) to build a friendly web interface that
runs locally. In order to launch the demo, run `run_demo.py --config config.yml`: after a few seconds, a local link and
a shared link to the web app will be provided. The demo uses ONNX runtime, thus an ONNX model path has to be provided
first in the config file (see Section [ONNX Runtime](#ONNX Runtime)).

The interface allows the user to drop or to record an audio file and to get a predicition of the audio language in a few
milliseconds (~100 ms). There is also the possibility to load some examples and to flag some inputs that resulted in an
unexpected outcome.

## Run Training and Evaluation

Instead of running individual scripts, you can use the training supervisor to train or evaluate the model:

```
# Run training
run.py --config config.yml --mode train

# Run evaluation
run.py --config config.yml --mode eval
```

For that we use a package called [python fire](https://google.github.io/python-fire/guide/)

### Dataset Processing

The dataset is composed of 88k WAV files in english, italian, french, german and swiss german (including different
dialects). The four main languages are part of [VoxLingua107](http://bark.phon.ioc.ee/voxlingua107/), which contains
audios extracted from YouTube videos sampled at 16 kHz, while the swiss german audios are recordings from
the [FHNW Institute for Data Science Datasets](https://www.cs.technik.fhnw.ch/i4ds-datasets), sampled at 44.1 kHz.

When first called, the AudioDataset constructor computes training, validation and test splits, which are saved in csv
files in the `data` dir. When the PyTorch DataLoader calls the AudioDataset getitem method, a batch of audio files is
loaded, resampled to 16kHz and trimmed (or warp-padded) to the specified length (100k samples, i.e. ~6sec). Then,
Huggingface's Speech2TextFeatureExtractor extracts mel-bank filters features of
shape `(batch_size, sequence_length, features_size)`
and the corresponding attention masks of shape `(batch_size, sequence_length)`, which are then fed into the transformer.
See [Speech2Text docs](https://huggingface.co/transformers/model_doc/speech_to_text.html#speech2textfeatureextractor)
for further details.

TODO: Use the feature extractor to also trim the signal, so that masks are actually effective. It didn't work on GPU
with multiple workers.

### Training pipeline

The SpokenLanguageCLassifier is composed of three blocks: a pretrained Speech2Text encoder that encodes the input
mel-bank filters, a convolution block that further encodes the features and reduces the spatial dimension of the hidden
states thanks to Max pooling, and a classifier head composed of linear layers. The model uses lazy layers to be
independent of the specified signal length at training time, i.e. one doesn't have to know the `sequence_length` a
priori to hardcode the input channels/features of the layers. Lazy layers are instantiated as normal Pytorch layers
after a first run of the model with dummy inputs.

The inputs are mel-bank filter features and corresponding attention masks, both obtained with the
Speech2TextFeatureExtractor. The output is the last raw hidden state, which can be turned into a probability by applying
a softmax. We predict one of 5 languages between english, italian, french, german and swiss german.

The model is trained with Categorical cross-entropy loss for 50 epochs. The optimizer is Adam with Decoupled Weight
Decay Regularization ([AdamW](https://arxiv.org/abs/1711.05101)). The scheduler decreases the learning rate linearly
from the initial lr (0.001) set in the optimizer to 0, after a warmup period during which it increases linearly from 0
to the specified lr.

The best model is selected according to the best value of the monitored validation metric, which can be one of `F1`
, `IoU` or `Precision` (accuracy).

### Evaluation

Average scores on the test set:

| Precision  | Recall | F1     | IoU    |
| :--------: | :----: | :----: | :----: |
|   0.9297   | 0.9302 | 0.9296 | 0.8699 |

Per class F1 score:

| Swiss German  | English | Italian | French | German |  
| :-----------: | :-----: | :-----: | :----: | :----: |  
|   0.9845      | 0.8981  | 0.9100  | 0.9373 | 0.9179 |  

## ONNX Runtime

The project allows to export the model to ONNX and to run it with the ONNX runtime. Please, read
the [torch.onnx documentation](https://pytorch.org/docs/stable/onnx.html) to see a tutorial and get more info. In our
case, the model is converted to a static graph by tracing, i.e. the model is executed once with the given inputs and all
operations that happen during that execution are recorded.

In the config file, set `export_to_onnx` and `onnx_runtime` to
`True` (providing the correct model path) to respectively export the best model to ONNX during training and to run ONNX
inference at test time. Our experiments show that ONNX runtime is 2.5 times faster than the standard PyTorch
evaluation (on CPU), while preserving same accuracy and model size.

### ONNX Optimization

ONNX Runtime provides three levels of graph optimizations (graph-level transformations) to improve model performance:

1. Basic (constant-folding, redundant node eliminations, semantics-preserving node fusions), applied before graph
   partitioning and thus independent on the CPU provider
2. Extended (complex node fusions), applied after graph partitioning and thus dependent on the CPU provider
3. Layout Optimizations (change data layout, e.g. NCHWc optimization)

All optimizations can be performed either online (right before inference) or offline (saving the optimized model on
disk). Our experiments show no significant improvements in the runtime by using optimizations.

### ONNX Quantization

Quantization refers to techniques for doing both computations and memory accesses with lower precision data, usually
int8 compared to floating point implementations. In ONNX this is done by mapping the floating point real values to an 8
bit quantization space: VAL_fp32 = Scale * (VAL_quantized - Zero_point). Quantization enables performance gains such as:

- 4x reduction in model size
- 2-4x reduction in memory bandwidth
- 2-4x faster inference due to savings in memory bandwidth and faster compute with int8 arithmetic. The exact speed up
  depends on hardware and model: old machines may have few instruction support for byte computation.

Quantization does not however come without additional cost. Fundamentally quantization means introducing approximations
and the resulting networks have slightly less accuracy.

There are three types of quantization:

1. Dynamic quantization calculates the quantization parameter (scale and zero point) for activations dynamically.
2. Static quantization leverages the calibration data to calculate the quantization parameter of activations.
3. Quantization aware training calculates the quantization parameter of activation while training, and the training
   process can control activation to a certain range.

We experimented with dynamic quantization applied to the whole model, and with static quantization applied to different
subsets of operations, with the goal of finding a good compromise between model size, latency and accuracy. Regarding
runtime, we noticed that inference on a statically quantized model runs 15% faster w.r.t. a dynamically quantized model,
which instead don't show any particular improvement in runtime w.r.t. non quantized models.

In addition to quantization, we also converted all weights to float16 precision, while keeping inputs and outputs to
float32 precision. No loss in accuracy is registered, while the model size is halved (as expected).

|Quantization           | F1     | IoU    | Size (MB) |
|:--------:             | :----: | :----: | :-------: |
|None                   | 0.9313 | 0.8729 |   188.4   |
|Dynamic on all ops     | 0.9309 | 0.8722 |    56.8   |
|Static on all ops      | 0.2821 | 0.1739 |    56.8   |
|Static on MatMul only  | 0.9298 | 0.8703 |    75.2   |
|Convert to float16     | 0.9315 | 0.8732 |    94.2   |

## Tensorboard

After training a model you can check tensorboard by running the following command:

```
tensorboard --logdir workspace/<TIMESTAMP>/logs
```

and then go to [http://localhost:6006/](http://localhost:6006/)

You should see some charts. If you're running training on a remote ssh instance, type the following on your local
machine and then go to [http://localhost:6006/](http://localhost:6006/)

```
ssh -i <YOUR_PEM_AUTH_KEY> -NL 6006:localhost:6006 <REMOTE_USERNAME>@<REMOTE_MACHINE_ADDRESS>
```

To compare several different trainings you can put all model output folders in a folder and tensorboard will recursively
look for logging data in all subfolders :

```
tensorboard --logdir <TRAINING_ROOT_FOLDER>
```

To manually pick several folders to compare and name them, use

```
tensorboard --logdir_spec=experiment_a:/model_path_a,experiment_b:/model_path_b
```