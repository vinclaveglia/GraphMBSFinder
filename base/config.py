import pathlib

class Atom:

    CA = 'CA'
    CB = 'CB'

#SITES_PATH = pathlib.Path("../..").joinpath('data').joinpath('Zn_sites')
#PROTS_PATH = pathlib.Path("../..").joinpath('data').joinpath('proteins')

SITES_PATH = pathlib.Path(__file__).parent.joinpath("../../data/Zn_sites")
PROTS_PATH = pathlib.Path(__file__).parent.joinpath("../../data/proteins")
DATA_PATH = pathlib.Path(__file__).parent.joinpath("../../data")

METADATA_PATH = pathlib.Path(__file__).parent.joinpath("../metadata")

ProtT5_EMBEDDINGS_PATH = DATA_PATH.joinpath('protT5_embeddings')