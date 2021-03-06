import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from experiments.sentence_similarity.util import to_file, sentence_similarity, pearson_correlation, plot_scatter
from experiments.util.training import train_model

logger = logging.getLogger(__name__)

""""
MedSTS Benchmark
Authors: Yanshan Wang, Naveed Afzal, Sunyang Fu, Liwei Wang, Feichen Shen, Majid Rastegar-Mojarad, Hongfang Liu
Paper: https://arxiv.org/ftp/arxiv/papers/1808/1808.09397.pdf 
"""


def prepare_train(in_path, out_path):
    frame = pd.read_csv(in_path, sep='\t', header=None)
    frame = frame.reset_index()
    frame.columns = ['pair_id', 'sentence_1', 'sentence_2', 'score']
    to_file(frame, out_path)
    return frame


def prepare_test(in_path, score_path, out_path):
    frame = pd.read_csv(in_path, sep='\t', header=None)
    frame['score'] = pd.read_csv(score_path, sep='\t', header=None)
    frame = frame.reset_index()
    frame.columns = ['pair_id', 'sentence_1', 'sentence_2', 'score']
    to_file(frame, out_path)
    return frame


@hydra.main('../../config', 'medsts_config.yaml')
def experiment(config: DictConfig):
    train_in_path = hydra.utils.to_absolute_path(config.data.raw_train_path)
    test_in_path = hydra.utils.to_absolute_path(config.data.raw_test_path)
    test_score_path = hydra.utils.to_absolute_path(config.data.raw_test_score_path)
    train_out_path = hydra.utils.to_absolute_path(config.data.train_path)
    test_out_path = hydra.utils.to_absolute_path(config.data.test_path)

    train_df = prepare_train(train_in_path, train_out_path)
    test_df = prepare_test(test_in_path, test_score_path, test_out_path)

    model = train_model(config)

    train_df['similarity'] = sentence_similarity(model, train_df)
    test_df['similarity'] = sentence_similarity(model, test_df)

    train_correlation = pearson_correlation(train_df.similarity, train_df.score)
    test_correlation = pearson_correlation(test_df.similarity, test_df.score)

    train_chart = plot_scatter(train_df, train_correlation)
    train_chart.save('train_chart.json')

    test_chart = plot_scatter(test_df, test_correlation)
    test_chart.save('test_chart.json')

    logger.info('Encoder %s', config.model.encoder)
    logger.info('Pooling %s', config.model.pooling)
    logger.info('Normalizer %s', config.model.normalizer)
    logger.info('MedSTS Train correlation %s', train_correlation)
    logger.info('MedSTS Test correlation %s', test_correlation)


if __name__ == '__main__':
    experiment()
