from . import moji
from . import bios
from . import coloredMNIST
from . import COMPAS
from . import TP_POS
from . import Adult
from . import MSCOCO
from . import imSitu
from . import bffhq
from . import celeba
from . import celeba_bm
from . import Adult

name2class = {
    "moji":moji,
    "bios":bios,
    "coloredmnist":coloredMNIST,
    "compas":COMPAS,
    "tp_pos":TP_POS,
    "adult":Adult,
    "coco":MSCOCO,
    "imsitu":imSitu,
    "bffhq":bffhq,
    "celeba":celeba,
    "celeba_bm":celeba_bm,
    "adult":Adult,
}

def prepare_dataset(name, dest_folder,batch_size=64, seed=42):
    if name in name2class.keys():
        data_class = name2class[name.lower()]
        initialized_class = data_class.init_data_class(dest_folder=dest_folder, batch_size=batch_size, seed=seed)
        initialized_class.prepare_data()
    else:
        print(f"Unknown dataset '{name}', please double check.")
