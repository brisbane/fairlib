import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import logging
import numpy as np

from .utils import BaseModel
from .augmentation_layer import Augmentation_layer

from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights
class MLP(BaseModel):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        assert args.n_hidden >= 0, "n_hidden must be nonnegative"

        if  not args.softmax: 
          self.output_layer = nn.Linear(
            args.emb_size if args.n_hidden == 0 else args.hidden_size, 
            args.num_classes if not args.regression else 1,
            )
        else:
          self.output_layer = nn.Sequential ( nn.Linear(
            args.emb_size if args.n_hidden == 0 else args.hidden_size,
            args.num_classes if not args.regression else 1,
            ), 
            nn.Softmax(dim=0))
        
        # Init batch norm, dropout, and activation function
        self.init_hyperparameters()

        # Init hidden layers
        self.hidden_layers = self.init_hidden_layers()

        self.cls_parameter = self.get_cls_parameter()

        # Augmentation layers
        if self.args.gated:
            if self.args.n_hidden == 0:
                logging.info("Gated component requires at least one hidden layers in the model")
                pass
            else:
                # Init the mapping for the augmentation layer
                if self.args.gated_mapping is None:
                    # For each class init a discriminator component
                    self.mapping = torch.eye(self.args.num_groups, requires_grad=False)
                else:
                    # self.mapping = torch.from_numpy(mapping, requires_grad=False)
                    raise NotImplementedError

                self.augmentation_components = Augmentation_layer(
                    mapping=self.mapping,
                    num_component=self.args.num_groups,
                    device=self.args.device,
                    sample_component=self.hidden_layers
                )
        
        self.cls_parameter = self.get_cls_parameter()
        
        self.init_for_training()

    def forward(self, input_data, group_label = None):
        # main out
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if self.args.gated and self.args.n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentation"

            specific_output = self.augmentation_components(input_data, group_label)

            main_output = main_output + specific_output

        output = self.output_layer(main_output)
        return output
    
    def hidden(self, input_data, group_label = None):
        assert self.args.adv_level in ["input", "last_hidden", "output"]

        if self.args.adv_level == "input":
            return input_data
        else:
            # main out
            main_output = input_data
            for layer in self.hidden_layers:
                main_output = layer(main_output)

            # Augmentation
            if self.args.gated and self.args.n_hidden > 0:
                assert group_label is not None, "Group labels are needed for augmentation"

                specific_output = self.augmentation_components(input_data, group_label)

                main_output = main_output + specific_output
            if self.args.adv_level == "last_hidden":
                return main_output
            elif self.args.adv_level == "output":
                output = self.output_layer(main_output)
                return output
            else:
                raise "not implemented yet"
    
    def init_hidden_layers(self):
        args = self.args

        if args.n_hidden == 0:
            return nn.ModuleList()
        else:
            hidden_layers = nn.ModuleList()
            
            all_hidden_layers = [nn.Linear(args.emb_size, args.hidden_size)] + [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_hidden-1)]

            for _hidden_layer in all_hidden_layers:
                hidden_layers.append(_hidden_layer)
                if self.dropout is not None:
                    hidden_layers.append(self.dropout)
                if self.BN is not None:
                    hidden_layers.append(self.BN)
                if self.AF is not None:
                    hidden_layers.append(self.AF)
            return hidden_layers

    def get_cls_parameter(self):
        parameters = []
        if self.args.adv_level == "output":
            return parameters
        else:
            parameters.append(
                {"params":self.output_layer.parameters(),}
            )
            if self.args.adv_level == "last_hidden":
                return parameters
            elif self.args.adv_level == "input":
                parameters.append(
                    {"params":self.hidden_layers.parameters(),}
                )
                # Augmentation
                if self.args.gated and self.args.n_hidden > 0:
                    parameters.append(
                        {"params":self.augmentation_components.parameters(),}
                    )
                return parameters
            else:
                raise "not implemented yet"
class MLP_decreasing (MLP):
    def __init__(self, args):
        print ("init DecreasingNN")
        super(MLP_decreasing, self).__init__(args)
        self.args = args
    def init_hidden_layers(self):
        args = self.args
        
        if args.n_hidden == 0:
            return nn.ModuleList()
        else:
            hidden_layers = nn.ModuleList()
            hidden_size=args.hidden_size
            all_hidden_layers=[]

            count=0
            for _ in  range(args.n_hidden-1):
               print ("loop :", count, hidden_size*2)
               count+=1
               all_hidden_layers.append(nn.Linear(in_features=int(hidden_size*2), out_features=int(hidden_size) ))
               hidden_size=int(hidden_size*2)
            self.hidden_size=hidden_size
            count=0
            all_hidden_layers.append(nn.Linear(args.emb_size, hidden_size))

            for _hidden_layer in reversed(all_hidden_layers):
                count=count+1
                hidden_layers.append(_hidden_layer)
#                help(_hidden_layer)
                hsize=_hidden_layer.out_features
                if self.dropout is not None:
                    hidden_layers.append(self.dropout)
                if self.BN is not None:
                    hidden_layers.append(
                       nn.BatchNorm1d(hsize)
                       )

                if self.AF is not None:
                    hidden_layers.append(self.AF)
            print ("nhidden is" , count)
            return hidden_layers


class BERTClassifier(BaseModel):
    model_name = 'bert-base-cased'
    n_freezed_layers = 12

    def __init__(self, args):
        super(BERTClassifier, self).__init__()
        self.args = args

        self.bert = BertModel.from_pretrained(self.model_name)

        self.bert_layers = [self.bert.embeddings, 
                                self.bert.encoder.layer[0],
                                self.bert.encoder.layer[1],
                                self.bert.encoder.layer[2],
                                self.bert.encoder.layer[3],
                                self.bert.encoder.layer[4],
                                self.bert.encoder.layer[5],
                                self.bert.encoder.layer[6],
                                self.bert.encoder.layer[7],
                                self.bert.encoder.layer[8],
                                self.bert.encoder.layer[9],
                                self.bert.encoder.layer[10],
                                self.bert.encoder.layer[11],
                                self.bert.pooler]

        self.classifier = MLP(args)

        self.freeze_roberta_layers(self.n_freezed_layers)

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        input_ids, input_masks = input_data
        bert_output = self.bert(input_ids, encoder_attention_mask=input_masks)[1]

        return self.classifier(bert_output, group_label)
    
    def hidden(self, input_data, group_label = None):
        input_ids, input_masks = input_data
        bert_output = self.bert(input_ids, encoder_attention_mask=input_masks)[1]

        return self.classifier.hidden(bert_output, group_label)

    def freeze_roberta_layers(self, number_of_layers):
        "number of layers: the first number of layers to be freezed"
        assert (number_of_layers < 14 and number_of_layers > -14), "beyond the total number of RoBERTa layer groups(14)."
        for target_layer in self.bert_layers[:number_of_layers]:
                for param in target_layer.parameters():
                    param.requires_grad = False
        for target_layer in self.bert_layers[number_of_layers:]:
                for param in target_layer.parameters():
                    param.requires_grad = True
    
    def trainable_parameter_counting(self):
        model_parameters = filter(lambda p: p.requires_grad, self.bert.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


class ResNet(BaseModel):

    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args

        self.conv1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model_ft = self.conv1
        ct = 0
        for child in model_ft.children():
            ct += 1
        #    if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
        #strip ouut the final layer and replace with a fuly connected trinable layer
        num_final_in = self.conv1.fc.in_features
        self.conv1.fc = nn.Linear(num_final_in, args.emb_size)
        self.classifier = MLP(args)

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        #flatten? batch size
        x = x.view(-1, 1000)

        return self.classifier(x, group_label)

    def hidden(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = x.view(-1, 1000)

        return self.classifier.hidden(x, group_label)
class ConvNet(BaseModel):

    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.classifier = MLP(args)

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier(x, group_label)

    def hidden(self, input_data, group_label = None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier.hidden(x, group_label)
