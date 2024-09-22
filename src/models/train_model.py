import click
import logging
from pathlib import Path
import scipy.sparse as sp
from implicit.nearest_neighbours import CosineRecommender

@click.command()
@click.argument('interactions_path')
@click.argument('path_to_save')
def train_implicit_model(interactions_path: str, path_to_save: str) -> None:
	logger = logging.getLogger(__name__)
	logger.info('fitting model')
	data = sp.load_npz(interactions_path)
	model = CosineRecommender(K=20)
	model.fit(data)
	model.save(path_to_save)
	logger.info(f'model saved in {path_to_save}')


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	project_dir = Path(__file__).resolve().parents[2]	
	train_implicit_model()