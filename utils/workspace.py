from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.logger import get_logger


class Workspace:

    def __init__(self, output_dir, mode: str):

        if mode == 'train':
            self.workspace_path = Path(output_dir)
            self.model_path = self.workspace_path / datetime.now().strftime('%y%m%d-%H%M%S')
            self.log_path = self.model_path / 'logs'

            self.workspace_path.mkdir(exist_ok=True, parents=True)
            self.model_path.mkdir(exist_ok=True)
            self.log_path.mkdir(exist_ok=True)

            self.logger = get_logger(self.log_path, 'training')

            self.writer = SummaryWriter(str(self.log_path))
            self.step = {}
        else:
            self.model_path = Path(output_dir).parent
            self.log_path = self.model_path / 'logs'
            assert self.log_path.is_dir()

            self.logger = get_logger(self.log_path, 'evaluation')

    def save_model_state(self, state: dict, is_best: bool = False):
        if is_best:
            state = {'model_state': state['model_state']}
            torch.save(state, self.model_path / 'best.pth.tar')
        else:
            torch.save(state, self.model_path / 'ckpt.pth.tar')

    def export_model_to_onnx(self, model: torch.nn.Module, batch):
        input_names = list(batch.keys())
        in_out_names = input_names + ['output']
        dynamic_axes = {name: {0: 'batch_size'} for name in in_out_names}
        model.eval()

        torch.onnx.export(
            model,  # model being run
            batch,  # model input (or a tuple for multiple inputs)
            self.model_path / 'best.onnx',  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=False,  # whether to execute constant folding for optimization
            input_names=input_names,  # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes=dynamic_axes
        )

    def info(self, message: str):
        self.logger.info(message)

    def add_scalar(self, name, scalar):
        try:
            self.step[name] += 1
        except ValueError:
            self.step[name] = 0
        self.writer.add_scalar(name, scalar, global_step=self.step[name])
