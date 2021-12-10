from torch import nn
from transformers import Speech2TextModel


class SpokenLanguageClassifier(nn.Module):

    def __init__(self, config, n_classes):
        super(SpokenLanguageClassifier, self).__init__()

        self.config = config
        self.clf_hidden_states = config.classifier_hidden_states
        self.clf_dropout = config.classifier_dropout
        self.filters = config.conv_filters

        self.encoder = Speech2TextModel.from_pretrained(config.pretrained_encoder_path).get_encoder()

        self.conv = nn.Sequential(
            nn.LazyConv1d(out_channels=self.filters[0], kernel_size=(3,)),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=(2,)),
            nn.Conv1d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=(3,)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.filters[1])
        )

        self.classifier_head = nn.Sequential(
            nn.LazyLinear(self.clf_hidden_states),
            nn.Dropout(p=self.clf_dropout),
            nn.LeakyReLU(),
            nn.Linear(self.clf_hidden_states, n_classes)
        )

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        y = self.encoder(input_features=x['features'], attention_mask=x['attention_mask'])
        y = self.conv(y['last_hidden_state'])

        y = y.view(-1, y.shape[1] * y.shape[2])
        y = self.classifier_head(y)

        return y
