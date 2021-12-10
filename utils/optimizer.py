from onnxruntime.transformers import optimizer


def optimize_model(model_path, opt_level=99):
    # opt_level: 1 for Basic, 2 for Extended, 99 for Layout optimizations. The optimizations belonging to one level
    # are performed after the optimizations of the previous level have been applied.
    assert opt_level in [1, 2, 99]
    out_path = str(model_path).replace('.onnx', '_optimized.onnx')
    optimizer.optimize_by_onnxruntime(
        model_path,
        optimized_model_path=out_path,
        opt_level=opt_level
    )
    return out_path
