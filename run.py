import fire
import yaml
from easydict import EasyDict

from pipeline.trainer import Trainer


def main(config: str, mode):
    with open(config, 'r') as fp:
        config = EasyDict(yaml.load(fp, Loader=yaml.FullLoader))

    trainer = Trainer(config)
    if mode == 'train':
        trainer.train()
    elif mode == 'eval':
        trainer.test()
    else:
        raise (NotImplementedError, "Expected 'train' or 'eval' as 'mode'")


if __name__ == "__main__":
    fire.Fire(main)
