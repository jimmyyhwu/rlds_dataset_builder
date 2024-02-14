from typing import Iterator, Tuple, Any

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import yaml
from PIL import Image

import benchmark


def load_scenario(scenario_name):
    with open(f'scenarios/{scenario_name}.yml', 'r', encoding='utf8') as f:
        scenario_dict = yaml.safe_load(f)
    scenario = benchmark.Scenario(
        room=None,
        receptacles=[r for r, info in scenario_dict['receptacles'].items() if 'primitive_names' in info],
        seen_objects=scenario_dict['seen_objects'],
        seen_placements=scenario_dict['seen_placements'],
        unseen_objects=scenario_dict['unseen_objects'],
        unseen_placements=scenario_dict['unseen_placements'],
        annotator_notes=scenario_dict['annotator_notes'],
        tags=None)

    return scenario


def get_language_instruction(annotator_notes):
    parts = annotator_notes.split('. ')
    assert len(parts) == 2
    return f'{parts[0]}.'


class Tidybot(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for TidyBot dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Egocentric camera RGB image of current object.',
                        ),
                        'object': tfds.features.Text(
                            doc='Name of current object.',
                        ),
                        'receptacles': tfds.features.Sequence(
                            feature=tfds.features.Text(),
                            doc='Names of available receptacles.',
                        ),
                    }),
                    'action': tfds.features.Text(
                        doc='Name of selected receptacle.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, data, scenario_name):
            # load raw data
            scenario = load_scenario(scenario_name)
            placements = {o: r for o, r in scenario.unseen_placements}
            language_instruction = get_language_instruction(scenario.annotator_notes)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'image': np.array(Image.open(step['image_path'])),
                        'object': step['object'],
                        'receptacles': scenario.receptacles,
                    },
                    'action': placements[step['object']],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return episode_path, sample

        # create list of all examples
        df = pd.read_csv('annotations.csv')
        df['episode_path'] = df.apply(lambda row: f'scenario-{row["Scenario"]:02}-run-{row["Run"]:02}', axis=1)
        df['image_path'] = df.apply(lambda row: f'images/{row["Scenario"]:02}-{row["Run"]:02}/{row["Image"]}', axis=1)
        df = df.rename(columns={'Object': 'object'})
        episode_to_data = df.groupby('episode_path').apply(lambda x: x[['image_path', 'object']].to_dict('records')).to_dict()
        episode_to_scenario_name = {episode_path: f'scenario-{row["Scenario"]:02}' for episode_path, row in df.groupby('episode_path').first().iterrows()}

        # for smallish datasets, use single-thread parsing
        for episode_path in sorted(episode_to_data):
            yield _parse_example(episode_path, episode_to_data[episode_path], episode_to_scenario_name[episode_path])
