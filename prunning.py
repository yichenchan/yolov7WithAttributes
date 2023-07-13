import torch
from models.yolo import Model
import models
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch import ModelSpeedup

# define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Model("./cfg/training/yolov7-chezai_city_and_highway_without_detect.yaml").to(device)  # create
#whole_model = torch.load("./chezai_box_city_and_highway/yolov7-class23/weights/best.pt")['model'].to(device)
model = torch.load("./vehicleTypeModels/epoch_001.pt")['model'].type(torch.cuda.FloatTensor).to(device)
#torch.save(whole_model.state_dict(), "./chezai_box_city_and_highway/yolov7-class23/weights/best_state_dict.pt")
#print('state dict saved done!')
#model.load_state_dict(torch.load("./chezai_box_city_and_highway/yolov7-class23/weights/best_state_dict.pt"), strict=False)

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

#ModelSpeedup(model, dummy_input=torch.rand([1, 3, 640, 640]).to(device), masks_file=masks, confidence=2).speedup_model()
ModelSpeedup(model, dummy_input=torch.rand([1, 3, 224, 224]).to(device), masks_file=masks, confidence=2).speedup_model()
#torch.save(model.state_dict(), './chezai_box_city_and_highway/yolov7-class23/weights/pruned0.5_best_state_dict_without_detect.pt')
torch.save(model.type(torch.cuda.HalfTensor), './vehicleTypeModels/epoch_001_prunned_1.pt')
print(model)
