import torch
from models.yolo import Model
import models
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
from nni.compression.pytorch import ModelSpeedup
import os

# nni version==2.6.0

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model("./cfg/training/yolov7-WithAttri-NoDetectLayer.yaml").to(device)  # create
model.load_state_dict(torch.load("./chezai_box_with_attr/train/exp3/weights/best.pt")['model'].type(torch.FloatTensor).to(device), strict=False)

config_list = [{
    'sparsity_per_layer': 0.5,
    'op_types': ['Linear', 'Conv2d']
}]

pruner = L1NormPruner(model, config_list)

# compress the model and generate the masks
_, masks = pruner.compress()

# need to unwrap the model, if the model is wrapped before speedup
pruner.show_pruned_weights()
pruner._unwrap_model()

ModelSpeedup(model, dummy_input=torch.rand([1, 3, 640, 640]).to(device), masks_file=masks, confidence=2).speedup_model()
torch.save(model.state_dict(), './chezai_box_with_attr/train/exp3/weights/pruned0.5_best.pt')
print(model)
