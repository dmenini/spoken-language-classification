import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from pipeline.feature_extractor import FeatureExtractor


class AudioDataset(Dataset):

    def __init__(self, mode: str, config):
        super(AudioDataset, self).__init__()

        self.config = config
        self.languages = config.languages
        self.trim_length = config.trim_length
        self.dataset_split_ratio = config.dataset_split_ratio

        self.data_dir = Path(config.data_dir)
        self.set_file = self.data_dir / (mode + '_set.csv')
        if not self.set_file.is_file():
            self._generate_train_val_test_split()

        self.files = []
        self.labels = []
        self.read_csv(self.set_file)

        self._extractor = FeatureExtractor(config)

    def read_csv(self, filename: Path):
        with open(filename, 'r') as fp:
            reader = csv.reader(fp)
            for row in reader:
                self.files.append(str(row[0]))
                self.labels.append(np.array(self.languages.index(row[1])))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        audio, _ = sf.read(self.data_dir / self.files[item])
        audio = self.trim_audio(audio, trim_length=self.trim_length)
        features = self._extractor.extract(audio)
        return {
            'item_id': item,
            'features': features.input_features.squeeze().astype(np.float32),
            'attention_mask': features.attention_mask.squeeze().astype(np.int64),
            'label': self.labels[item].astype(np.int64)
        }

    def _generate_train_val_test_split(self):
        train_set, test_set, val_set = [], [], []
        for language in self.languages:
            directory = self.data_dir / language
            files = [language + '/' + filename.name for filename in directory.iterdir()]
            files.sort()
            train_set.extend(files[:int(len(files) * self.dataset_split_ratio)])
            val_test_set = files[int(len(files) * self.dataset_split_ratio):]
            val_set.extend(val_test_set[:int(len(val_test_set) * 0.5)])
            test_set.extend(val_test_set[int(len(val_test_set) * 0.5):])

            for file in [file for file in val_set if train_set[-1].split('__')[0] in file]:
                val_set.remove(file)
                train_set.append(file)
            for file in [file for file in test_set if val_set[-1].split('__')[0] in file]:
                test_set.remove(file)
                val_set.append(file)

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        AudioDataset.write_csv(self.data_dir / 'train_set.csv', train_set)
        AudioDataset.write_csv(self.data_dir / 'val_set.csv', val_set)
        AudioDataset.write_csv(self.data_dir / 'test_set.csv', test_set)

    @staticmethod
    def _write_csv(out_file: Path, dataset: list):
        with open(out_file, 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',')
            for file in dataset:
                language = file.split('/')[0]
                writer.writerow([file, language])

    @staticmethod
    def trim_audio(signal, trim_length=100_000):
        signal_length = signal.shape[0]
        if signal_length <= trim_length:
            signal = np.pad(signal, (0, trim_length - signal_length), mode='wrap')
        elif trim_length < signal_length <= trim_length * 1.1:
            signal = signal[(signal_length - trim_length):]
        else:
            signal = signal[int(trim_length * 0.1):int(trim_length * 1.1)]
        return signal
