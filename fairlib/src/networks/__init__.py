import sys
import torch
import logging
from .classifier import MLP, BERTClassifier,ConvNet,ResNet, MLP_decreasing
from . import utils
from . import INLP
from . import FairCL
from . import DyBT
from . import adv
from collections import defaultdict

def get_main_model(args):

    if args.encoder_architecture == "Fixed":
        if not args.decreasing_output_size:
           model = MLP(args)
        else:
           model = MLP_decreasing(args)
    elif args.encoder_architecture == "DecreasingNN":
        model = MLP_decreasing (args)
        print ("init DecreasingNN")
    elif args.encoder_architecture == "BERT":
        model = BERTClassifier(args)
    elif args.encoder_architecture == "MNIST":
        model = ConvNet(args)
    elif args.encoder_architecture == "ResNet":
        model = ResNet(args)
    else:
        print ("Unknown Model")
        raise NotImplementedError
    
    return model
