import onnxruntime
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from pipeline.audio_dataset import AudioDataset
from pipeline.transformer import SpokenLanguageClassifier
from utils.metrics import RunningScore
from utils.workspace import Workspace


class Trainer:

    def __init__(self, config):
        self.config = config
        self.batch_size = config.TRAINING.batch_size
        self.shuffle = config.TRAINING.shuffle
        self.n_epochs = config.TRAINING.n_epochs
        self.learning_rate = config.TRAINING.learning_rate
        self.start_epoch = 0

        self.languages = config.DATA.languages
        self.feature_size = config.DATA.feature_size
        self.sequence_length = int(config.DATA.trim_length / 160) - 2
        self.input_shape = (self.batch_size, self.sequence_length, self.feature_size)
        self.mask_shape = (self.batch_size, self.sequence_length)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.best_metric_value = -1.
        self.monitored_metric = config.TRAINING.monitored_metric
        self._metrics = RunningScore(n_classes=len(self.languages), monitored_metric=self.monitored_metric)

        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._workspace = None
        self._ort_session = None

    def train(self):
        train_loader = self._get_data_loader('train')
        val_loader = self._get_data_loader('val')

        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler(self._optimizer, train_loader)

        self._workspace = Workspace(self.config.output_dir, mode='train')
        self._workspace.info(
            f'Starting training. Batch size: {self.batch_size}, Epochs: {self.n_epochs}, device: {self.device}.\n'
            f'Total model parameters: {sum(p.numel() for p in self._model.parameters())}')

        if self.config.TRAINING.resume:
            self._load_checkpoint(self.config.TRAINING.checkpoint_path)

        for epoch in range(self.start_epoch, self.n_epochs):
            self._workspace.info(f' Epoch {epoch + 1}/{self.n_epochs} '.center(76, '*'))
            self._workspace.info(f'Training model')
            self._train_one_epoch(train_loader)
            torch.cuda.empty_cache()

            self._workspace.info(f'Evaluating model')
            self.evaluate(val_loader)
            for k, v in self._metrics.scores.items():
                self._workspace.add_scalar(f'eval/{k}', v)

            self.save_checkpoint(epoch)

    def test(self):
        self.batch_size = self.config.TESTING.batch_size
        self.shuffle = self.config.TESTING.shuffle
        test_loader = self._get_data_loader('test')

        if self.config.TESTING.onnx_runtime and not torch.cuda.is_available():
            self._ort_session = onnxruntime.InferenceSession(self.config.TESTING.model_path)
        else:
            self._model = self.get_pretrained_model(self.config.TESTING.model_path)

        self._workspace = Workspace(self.config.TESTING.model_path, mode='test')
        self._workspace.info(f'Starting evaluation')

        self.evaluate(test_loader)

    def evaluate(self, eval_loader: DataLoader):
        if self._ort_session and not torch.cuda.is_available():
            self._run_onnx_inference(eval_loader)
        else:
            self._run_inference(eval_loader)

        score, class_score = self._metrics.get_scores()
        for k, v in score.items():
            self._workspace.info(f'{k:9}:\t{v}')
        if class_score:
            self._workspace.info(f'Per class {self.monitored_metric} score:')
            for k, v in class_score.items():
                self._workspace.info(f'{self.languages[k]:5}:\t{v}')
        self._metrics.reset()

    def _train_one_epoch(self, train_loader: DataLoader):
        criterion = CrossEntropyLoss().to(self.device)
        train_loss_t = 0.
        self._model.train()
        for batch in tqdm(train_loader, total=len(train_loader), mininterval=30):
            self._optimizer.zero_grad()
            inputs, target = self._prepare_batch(batch)
            output = self._model.forward(inputs)
            loss = criterion.forward(output, target)
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            train_loss_t += loss.item()

        train_loss_t /= len(train_loader)
        self._workspace.info(f'Training loss {train_loss_t}')
        self._workspace.add_scalar('Train/loss', train_loss_t)

    def _run_inference(self, eval_loader: DataLoader):
        self._model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_loader, total=len(eval_loader), mininterval=30):
                inputs, target = self._prepare_batch(batch)
                output = self._model.forward(inputs)
                self._metrics.update(output.max(1)[1], target)

    def _run_onnx_inference(self, eval_loader: DataLoader):
        with torch.no_grad():
            for batch in tqdm(eval_loader, total=len(eval_loader), mininterval=30):
                inputs = {'features': batch['features'].cpu().numpy(),
                          'attention_mask': batch['attention_mask'].cpu().numpy()}
                target = batch['label']
                output = self._ort_session.run(None, inputs)

                self._metrics.update(output[0].argmax(axis=1), target)

    def _get_data_loader(self, mode: str):
        mum_workers = self.config.num_workers if torch.cuda.is_available() else 0
        dataset = AudioDataset(mode, self.config.DATA)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=mum_workers
        )

    def _setup_model(self):
        self._model = SpokenLanguageClassifier(
            config=self.config.MODEL,
            n_classes=len(self.languages),
        )
        self._model.to(self.device)
        dummy_batch = {
            'features': torch.randn(*self.input_shape).to(self.device),
            'attention_mask': torch.ones(*self.mask_shape, dtype=torch.int64).to(self.device)
        }
        self._model.forward(dummy_batch)  # turn Lazy layers into normal layers
        SpokenLanguageClassifier.init_weights(self._model.conv)
        SpokenLanguageClassifier.init_weights(self._model.classifier_head)

    def _setup_optimizer(self):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self._optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

    def _setup_scheduler(self, optimizer: torch.optim, train_loader: DataLoader):
        num_training_steps = len(train_loader) * self.n_epochs
        self._scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

    def _prepare_batch(self, batch: dict) -> tuple:
        inputs = {
            'features': batch['features'].to(self.device).float(),
            'attention_mask': batch['attention_mask'].to(self.device).long()
        }
        target = batch['label'].to(self.device).long()
        return inputs, target

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch + 1,
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'scheduler_state': self._scheduler.state_dict(),
            self.monitored_metric: self._metrics.scores[self.monitored_metric]
        }

        self._workspace.save_model_state(state, is_best=False)

        if self._metrics.scores[self.monitored_metric] >= self.best_metric_value:
            self.best_metric_value = self._metrics.scores[self.monitored_metric]
            self._workspace.save_model_state(state, is_best=True)
            self._workspace.info(
                f'Found best model with mean {self.monitored_metric} {self.best_metric_value} at epoch {epoch + 1}')
            if self.config.MODEL.export_to_onnx:
                self._export_to_onnx(self._model)

    def _export_to_onnx(self, model):
        dummy_batch = {
            'features': torch.randn(*self.input_shape, dtype=torch.float32).to(self.device),
            'attention_mask': torch.ones(*self.mask_shape, dtype=torch.int64).to(self.device)
        }
        self._workspace.export_model_to_onnx(model, dummy_batch)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state'])
        self._scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.start_epoch = checkpoint['epoch']
        self.best_metric = checkpoint[self.monitored_metric]
        self._workspace.info(f'Loaded checkpoint from {checkpoint_path}. '
                             f'Starting training from epoch {self.start_epoch}')

    def get_pretrained_model(self, model_path):
        if self._model is None:
            self._setup_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
        return self._model
