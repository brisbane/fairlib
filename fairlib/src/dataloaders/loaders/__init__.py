import sys

from.loaders import TestDataset, SampleDataset
from .Moji import DeepMojiDataset
from .Bios import BiosDataset
from .Valence import ValenceDataset
from .FCL_BiosDataset import FCL_BiosDataset
from .Trustpilot import TrustpilotDataset
from .Adult import AdultDataset
from .COMPAS import COMPASDataset
from .imSitu import imSituDataset
from .ColoredMNIST import MNISTDataset
from .bffhq import bffhqDataset
from .celeba import celebaDataset
from .celebMHQ import celebMHQDataset


if sys.platform == "win32":
    data_dir = r'.\data'
    default_dataset_roots = dict(
        Moji= r'.\data',
        Bios_gender= data_dir + r'\bios',
        Bios_economy= data_dir + r'\bios',
        Bios_both= data_dir + r'\bios',
        FCL_Bios= data_dir + r'\bios',
        Trustpilot_gender= data_dir + r'\trustpilot',
        Trustpilot_age= data_dir + r'\trustpilot',
        Trustpilot_country= data_dir + r'\trustpilot',
        Adult_gender= data_dir + r'\adult',
        Adult_race= data_dir + r'\adult',
        COMPAS_gender= data_dir + r'\compas',
        COMPAS_race= data_dir + r'\compas',
        imSitu = data_dir + r'\imsitu',
        MNIST = data_dir + r'\coloredmnist',
        bffhq = data_dir + r'\bffhq',
        celeba = data_dir + r'\celeba',
    )
else:
    data_dir = './data'
    default_dataset_roots = dict(
        Moji= data_dir + '/deepmoji/split2/',
        Bios_gender= data_dir + '/bios_gender_economy',
        Bios_economy= data_dir + '/bios_gender_economy',
        Bios_both= data_dir + '/bios_gender_economy',
        FCL_Bios= data_dir + '/bios_gender_economy',
        MNIST = data_dir + '/coloredmnist',
        bffhq = data_dir + '/bffhq',
        celeba = data_dir + '/celeba',
        celeba_bm = data_dir + '/celeba_bm',
        COMPAS_race= data_dir + '/compas',
        adult = data_dir + '/adult',
    )


loader_map = {
    "moji":DeepMojiDataset,
    "bios":BiosDataset,
    "test":TestDataset,
    "sample":SampleDataset,
    "valence":ValenceDataset,
    "fclbios":FCL_BiosDataset,
    "trustpilot":TrustpilotDataset,
    "adult":AdultDataset,
    "compas":COMPASDataset,
    "imsitu":imSituDataset,
    "mnist":MNISTDataset,
    "bffhq":bffhqDataset,
    "celeba":celebaDataset,
    "celebmhq":celebMHQDataset,
    "adult":AdultDataset,
#    "coco":COCODataset,
}

def name2loader(args):
    print("name2loader", args)    
    dataset_name = args.dataset.split("_")[0].lower()
    print ("dataset_name:", dataset_name)
    if  len(args.dataset.split("_")) > 1:
        args.protected_task = args.dataset.split("_")[1]


    if dataset_name in loader_map.keys():
        return loader_map[dataset_name]
    else:
        raise NotImplementedError
