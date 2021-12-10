import os

from onnxruntime.quantization import QuantType, CalibrationMethod
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static


# Only UINT8 activations supported. On AVX2 machines use UINT8 weights instead of signed INT8.
# Indeed, U8U8 kernel can process 6 rows at a time versus 4 rows for the U8S8 kernel,
# and U8U8 sequence is 2 instructions vs 3 instructions for U8S8.

def quantize_model_with_static(model_path, data_reader, op_types=None):
    """
    :param model_path: input model path
    :param data_reader: the DataReader must implement a generator that outputs a sample input from the calibration
                        dataset (usually 100s of samples from the validation set).
    :param op_types: list of operations to quantize. If None is specified, all operations are quantized
    :return: quantized model path

    Calibration methods:
    * MinMax: compute min-max range of activations from calibration data
    * Entropy: compute histogram of activations of intermediate layers by running inference with FP32 precision on
               calibration data, then selects min-max range that minimizes the loss of information (entropy).
    """
    out_path = str(model_path).replace('.onnx', '_s_quantized.onnx')
    quantize_static(
        model_path,
        out_path,
        calibration_data_reader=data_reader,
        calibrate_method=CalibrationMethod.Entropy,
        optimize_model=False,
        nodes_to_exclude=['Concat_1401'],  # Fails because of INT64 dtype
        op_types_to_quantize=op_types,
        weight_type=QuantType.QUInt8  # on AVX2 machine
    )
    os.remove('augmented_model.onnx')

    return out_path


def quantize_model_with_dynamic(model_path):
    """
    :param model_path: input model path
    :return: quantized model path
    """
    out_path = model_path.replace('.onnx', '_d_quantized.onnx')
    quantize_dynamic(
        model_path,
        out_path,
        optimize_model=False,
        weight_type=QuantType.QUInt8  # on AVX2 machine
    )
    os.remove('augmented_model.onnx')
    return out_path
