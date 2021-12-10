from transformers import Speech2TextFeatureExtractor


class FeatureExtractor:

    def __init__(self, config):
        self.num_mel_bins = config.feature_size
        self.sampling_rate = config.sampling_rate

        self._extractor = Speech2TextFeatureExtractor(
            sampling_rate=self.sampling_rate,
            num_mel_bins=self.num_mel_bins
        )

    def extract(self, audio):
        return self._extractor(audio,
                               sampling_rate=self.sampling_rate,
                               return_tensors='np')
