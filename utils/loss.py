# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale = 2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result


    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
    
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0] 
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10): 

        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_RS
        relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations=fg_logits-fg_logits[ii] 
            bg_relations=relevant_bg_logits-fg_logits[ii]

            if delta_RS > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)

            # Rank of ii among all examples
            rank=rank_pos+FP_num
                            
            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii]=FP_num/rank      

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

            #Find examples in the target sorted order for example ii         
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            #The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            #Compute target sorting error. (Eq. 8)
            #Since target ranking error is 0, this is also total target error 
            target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target

            #Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error
  
            #Identity Update for Ranking Error 
            if FP_num > eps:
                #For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

            #Find the positives that are misranked (the cause of the error)
            #These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            #Denominotor of sorting pmf 
            sorting_pmf_denom = torch.sum(missorted_examples)

            #Identity Update for Sorting Error 
            if sorting_pmf_denom > eps:
                #For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

        #Normalize gradients by number of positives 
        classification_grads[fg_labels]= (fg_grad/fg_num)
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
    
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None


class ComputeLossWithAttr:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossWithAttr, self).__init__()
        # 获取模型所在gpu号
        device = next(model.parameters()).device  # get model device
        # 获取超参数
        h = model.hyp  # hyperparameters

        # 定义 loss 函数
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.BCEsubCls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['subCls_pw']], device=device))
        self.BCEattrObj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['attrObj_pw']], device=device))

        # 标签平滑（减少误标注带来的影响）
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # 是否使用 focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            self.BCEcls, self.BCEobj = FocalLoss(self.BCEcls, g), FocalLoss(self.BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.gr, self.hyp, self.autobalance = model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, three_head_preds_mat, batch_all_targets):  # predictions, targets, model
        # three_head_preds 是网络的预测输出，是一个长度为 3 的list，单个元素尺寸为：
        # （
        # batch_size, 
        # anchor_num, 
        # feature_map_w, 
        # feature_map_h, 
        # [ # 以下为内容
        # x(0), y(1), w(2), h(3), obj(4), cls_one_hot(5~14),
        # 车辆类别（15、16、17),
        # 左车灯是否可见（18），左车灯状态（19、20），左车灯位置（21～24），
        # 右车灯是否可见（25），右车灯状态（26、27），右车灯位置（28～31），
        # 车屁股是否可见（32），车屁股框位置（33～36）
        # 车头是否可见（37），车头框位置（38～41）
        # 车牌是否可见（42），车牌位置（43～46），
        # 人的类别（47，48），人的状态（49、50）， 是否带头盔（51、52）
        # 人头是否可见（53），人头的位置（54～57），
        # 人骑的车是否可见（58），人骑的车的位置（59～62）
        # 电动车车牌是否可见（63），车牌的位置（64～67），
        # 红绿灯的颜色（68,69,70），
        # 路牌的颜色（71,72），
        # 公交车道第一个点坐标（2），公交车道第二个点坐标（2），公交车道第三个点坐标（2），公交车道第四个点坐标（2），
        # 斑马线第一个点坐标（2），斑马线第二个点坐标（2），斑马线第三个点坐标（2），斑马线第四个点坐标（2），
        # 网格线第一个点坐标（2），网格线第二个点坐标（2），网格线第三个点坐标（2），网格线第四个点坐标（2），
        # 导流线第一个点坐标（2），导流线第二个点坐标（2），导流线第三个点坐标（2），导流线第四个点坐标（2）
        # ]
        # ）

        # batch_all_targets 是数据增强后所有的标注框 bbox，尺寸为：
        #（
        # bbox_num, 
        # [
        # img_index(0), cls_index(1), x(2), y(3), w(4), h(5), 
        # 车辆类别（6），
        # 左车灯是否可见（7），左车灯状态（8），左车灯位置（9~12），
        # 右车灯是否可见（13），右车灯状态（14），右车灯位置（15~18），
        # 车屁股是否可见（19），车屁股框位置（20~23）
        # 车头是否可见（24），车头框位置（25~28）
        # 车牌是否可见（29），车牌位置（30~33），
        # 人的类别（34），人的状态（35）， 是否带头盔（36）
        # 人头是否可见（37），人头的位置（38~41），
        # 人骑的车是否可见（42），人骑的车的位置（43~46）
        # 电动车车牌是否可见（47），车牌的位置（48~51），
        # 红绿灯的颜色（52），
        # 路牌的颜色（53），
        # 公交车道第一个点坐标（2），公交车道第二个点坐标（2），公交车道第三个点坐标（2），公交车道第四个点坐标（2），
        # 斑马线第一个点坐标（2），斑马线第二个点坐标（2），斑马线第三个点坐标（2），斑马线第四个点坐标（2），
        # 网格线第一个点坐标（2），网格线第二个点坐标（2），网格线第三个点坐标（2），网格线第四个点坐标（2），
        # 导流线第一个点坐标（2），导流线第二个点坐标（2），导流线第三个点坐标（2），导流线第四个点坐标（2）
        # ]
        # ）
        
        # 获取设备号
        device = batch_all_targets.device

        # 获取所有正样本的各种对应信息
        total_positive_classes, total_positive_bboxes, total_positive_indices, total_positive_anchors, total_positive_targets = self.build_targets(three_head_preds_mat, batch_all_targets)  # targets

        # 三种 loss 进行初始化
        loss_class, loss_box_position, loss_objection = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        loss_attr_objection = torch.zeros(1, device=device)
        loss_sub_class = torch.zeros(1, device=device)
        loss_attr_position = torch.zeros(1, device=device)

        # 遍历网络输出的三个 head
        for head_index, curr_head_preds_mat in enumerate(three_head_preds_mat): 
            
            # 所有正样本对应的索引：图像 id、合适的anchor索引、y 方向偏移到的格子的索引、x 方向偏移到的格子的索引
            curr_head_targets_imgIds, curr_head_targets_anchor_indices, curr_head_targets_y_offsets_grids_indices, curr_head_targets_x_offsets_grids_indices = total_positive_indices[head_index]  

            # 初始化一个 objection 值的多维矩阵,先全部置0,然后存在匹配的情况就计算那个位置的 obj，未匹配到的仍然是 0
            # 这里注意 curr_head_obj_mat 是被都初始化为 0 的，也就是所有的网格默认为背景（负样本）
            curr_head_obj_mat = torch.zeros_like(curr_head_preds_mat[..., 0], device=device)  
            # 尺寸是:(batch_size, anchor_num, feature_map_w, feature_map_h)

            bbox_num = curr_head_targets_imgIds.shape[0]  
            # 没有标注框的背景不需要计算 loss
            if bbox_num > 0:
                # 获取所有的正样本预测
                # 取出第 curr_head_targets_imgIds 张图中的
                #    第 curr_head_targets_y_offsets_grids_indices 行
                #    第 curr_head_targets_x_offsets_grids_indices 行
                #    第 curr_head_targets_anchor_indices 个 anchor 框对应的预测框
                # 然后排列成 (3*bbox, 6) 的形状,与正样本的排序是对应的，即相同的索引对应匹配预测框和标签框
                curr_head_pred_positive = curr_head_preds_mat[curr_head_targets_imgIds, curr_head_targets_anchor_indices, curr_head_targets_y_offsets_grids_indices, curr_head_targets_x_offsets_grids_indices] 

                # 获取所有对应的正样本
                # 然后排列成 (3*bbox, 7)
                curr_head_targets = total_positive_targets[head_index]
                curr_head_targets_classes = total_positive_classes[head_index]

                #------------------ 进行目标检测框位置回归 ------------------------------
                # 这里需要和推理时候的位置编码一致,以便从特征图尺度放缩回正常尺寸
                # 获取当前 head 所有匹配到预测框的 xy、wh
                curr_head_pred_positive_xy = curr_head_pred_positive[:, :2].sigmoid() * 2. - 0.5
                curr_head_pred_positive_wh = (curr_head_pred_positive[:, 2:4].sigmoid() * 2) ** 2 * total_positive_anchors[head_index]
                # curr_head_pred_positive_xy 尺寸为: (3*bbox, 2)

                # 把 xywh 合并到一起
                curr_head_pred_positive_boxes = torch.cat((curr_head_pred_positive_xy, curr_head_pred_positive_wh), 1) 
                # curr_head_pred_positive_box 尺寸为: (3*bbox, 4)

                # 计算当前 head 的所有预测框和标注框的 Ciou 值矩阵
                curr_head_pred_positive_iou = bbox_iou(curr_head_pred_positive_boxes.T, total_positive_bboxes[head_index], x1y1x2y2=False, CIoU=True)  

                # 把所有的 CIOU 值取平均作为当前 head 当前 batch 的位置 loss
                loss_box_position += (1.0 - curr_head_pred_positive_iou).mean()  

                #------------------ 进行目标检测类别分类----------------------------------
                # 根据标签平滑的正负样本值 self.cn 和 self.cp，得到标签框的 one-hot 类别向量
                # 先初始化填充类负样本的值 self.cn
                curr_head_bbox_cls_onehot = torch.full_like(curr_head_pred_positive[:, 5 : 5 + self.nc - 1], self.cn, device=device)  # targets
                # 再给正样本的位置赋值 self.cp
                curr_head_bbox_cls_onehot[range(bbox_num), curr_head_targets_classes] = self.cp
                # 计算类别 loss
                loss_class += self.BCEcls(curr_head_pred_positive[:, 5 : 5 + self.nc - 1], curr_head_bbox_cls_onehot) 

                # 遍历所有标注框（包括偏移的，bbox_num 是正常的 3 倍）
                curr_head_loss_sub_class = 0
                curr_head_loss_attr_objection = 0
                curr_head_loss_attr_position = 0
                for bbox_index in range(bbox_num):
                    # 获取当前遍历的标注框的类别
                    gt_box_cls = curr_head_targets_classes[bbox_index]

                    # 如果类别等于汽车
                    if(gt_box_cls == 0):
                        # ---------- 进行车辆的 左车灯是否可见|右车灯是否可见|车屁股是否可见|车头是否可见|车牌是否可见 属性回归
                        curr_head_loss_attr_objection += self.BCEattrObj(curr_head_pred_positive[bbox_index, [18, 25, 32, 37, 42]], curr_head_targets[bbox_index, [7, 13, 19, 24, 29]])

                        # ----------- 进行车辆子类别分类 -----------------------
                        curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, [15, 16, 17]], self.cn, device=device)
                        curr_bbox_car_subclass = int(curr_head_targets[bbox_index, 6].item())
                        curr_head_bbox_subclass_onehot[curr_bbox_car_subclass] = self.cp
                        curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, [15, 16, 17]], curr_head_bbox_subclass_onehot)

                        if(curr_head_targets[bbox_index, 7] == 1):
                            #------------ 如果左车灯可见，继续进行左车灯状态分类 --------------------------
                            curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, [19, 20]], self.cn, device=device)
                            curr_car_leftlight_subclass = int(curr_head_targets[bbox_index, 8].item())
                            curr_head_bbox_subclass_onehot[curr_car_leftlight_subclass] = self.cp
                            curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, [19, 20]], curr_head_bbox_subclass_onehot)

                            # ----------- 如果左车灯可见，继续进行左车灯位置回归 --------------------------
                            curr_head_pred_positive_leftlight_boxes = curr_head_pred_positive[bbox_index, 21:25].sigmoid()
                            curr_head_bbox_leftlight_boxes = curr_head_targets[bbox_index, 9:13]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_leftlight_boxes.T, curr_head_bbox_leftlight_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean() 

                        if(curr_head_targets[bbox_index, 13] == 1):
                            #------------ 如果右车灯可见，继续进行右车灯状态分类 --------------------------
                            curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, [26, 27]], self.cn, device=device)
                            curr_car_rightlight_subclass = int(curr_head_targets[bbox_index, 14].item())
                            curr_head_bbox_subclass_onehot[curr_car_rightlight_subclass] = self.cp
                            curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, [26, 27]], curr_head_bbox_subclass_onehot)

                            # ----------- 如果右车灯可见，继续进行右车灯位置回归 --------------------------
                            curr_head_pred_positive_rightlight_boxes = curr_head_pred_positive[bbox_index, 28:32].sigmoid()
                            curr_head_bbox_rightlight_boxes = curr_head_targets[bbox_index, 15:19]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_rightlight_boxes.T, curr_head_bbox_rightlight_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean() 

                        if(curr_head_targets[bbox_index, 19] == 1):
                            # ----------- 如果车屁股可见，继续进行车屁股位置回归 --------------------------
                            curr_head_pred_positive_carbutt_boxes = curr_head_pred_positive[bbox_index, 33:37].sigmoid()
                            curr_head_bbox_carbutt_boxes = curr_head_targets[bbox_index, 20:24]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_carbutt_boxes.T, curr_head_bbox_carbutt_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean() 

                        if(curr_head_targets[bbox_index, 24] == 1):
                            # ----------- 如果车头可见，继续进行车头位置回归 --------------------------
                            curr_head_pred_positive_carhead_boxes = curr_head_pred_positive[bbox_index, 38:42].sigmoid()
                            curr_head_bbox_carhead_boxes = curr_head_targets[bbox_index, 25:29]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_carhead_boxes.T, curr_head_bbox_carhead_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean()

                        if(curr_head_targets[bbox_index, 29] == 1):
                            # ----------- 如果车牌可见，继续进行车牌位置回归 --------------------------
                            curr_head_pred_positive_plate_boxes = curr_head_pred_positive[bbox_index, 43:47].sigmoid()
                            curr_head_bbox_plate_boxes = curr_head_targets[bbox_index, 30:34]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_plate_boxes.T, curr_head_bbox_plate_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean()

                    if(gt_box_cls == 1):
                        # ------- 进行行人的 人头是否可见、人骑的车是否可见、头盔是否可见、骑的车的车牌 属性回归
                        curr_head_loss_attr_objection += self.BCEattrObj(curr_head_pred_positive[bbox_index, [53,58,63]], curr_head_targets[bbox_index, [37,42,47]])

                        # --------- 进行行人类别、人的状态、是否带头盔分类 ---------------
                        curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, 47:53], self.cn, device=device)
                        curr_person_type_subclass = int(curr_head_targets[bbox_index, 34].item())
                        curr_person_state_subclass = int(curr_head_targets[bbox_index, 35].item())
                        curr_person_helmet_subclass = int(curr_head_targets[bbox_index, 36].item())
                        curr_head_bbox_subclass_onehot[curr_person_type_subclass] = self.cp
                        curr_head_bbox_subclass_onehot[2 + curr_person_state_subclass] = self.cp
                        curr_head_bbox_subclass_onehot[4 + curr_person_helmet_subclass] = self.cp
                        curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, 47:53], curr_head_bbox_subclass_onehot)

                        if(curr_head_targets[bbox_index, 42] == 1):
                            # ----------- 如果骑的车可见，继续进行电动车的位置回归 --------------------------
                            curr_head_pred_positive_motor_boxes = curr_head_pred_positive[bbox_index, 59:63].sigmoid()
                            curr_head_bbox_motor_boxes = curr_head_targets[bbox_index, 43:47]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_motor_boxes.T, curr_head_bbox_motor_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean() 

                        if(curr_head_targets[bbox_index, 47] == 1):
                            # ----------- 如果电动车车牌可见，继续进行电动车车牌位置回归 --------------------------
                            curr_head_pred_positive_motorplate_boxes = curr_head_pred_positive[bbox_index, 64:68].sigmoid()
                            curr_head_bbox_motorplate_boxes = curr_head_targets[bbox_index, 48:52]
                            curr_head_pred_positive_attr_iou = bbox_iou(curr_head_pred_positive_motorplate_boxes.T, curr_head_bbox_motorplate_boxes, x1y1x2y2=False, CIoU=True)
                            curr_head_loss_attr_position += (1.0 - curr_head_pred_positive_attr_iou).mean()

                    if(gt_box_cls == 3):
                        # -------------- 进行红绿灯属性分类 -------------------
                        curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, [68, 69, 70]], self.cn, device=device)
                        curr_trafficlight_subclass = int(curr_head_targets[bbox_index, 52].item())
                        curr_head_bbox_subclass_onehot[curr_trafficlight_subclass] = self.cp
                        curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, [68, 69, 70]], curr_head_bbox_subclass_onehot)

                    if(gt_box_cls == 5):
                        # -------------- 进行路牌属性分类 -------------------
                        curr_head_bbox_subclass_onehot = torch.full_like(curr_head_pred_positive[bbox_index, [71,72]], self.cn, device=device)
                        curr_sign_subclass = int(curr_head_targets[bbox_index, 53].item())
                        curr_head_bbox_subclass_onehot[curr_sign_subclass] = self.cp
                        curr_head_loss_sub_class += self.BCEsubCls(curr_head_pred_positive[bbox_index, [71,72]], curr_head_bbox_subclass_onehot)
                    
                    #if(gt_box_cls in [6, 7, 8, 9]):
                        # ------------- 进行各种不规则检测体的四顶点回归 -------------------
                
                # 这里记得除以遍历的 bbox 数量
                curr_head_loss_sub_class /= bbox_num
                curr_head_loss_attr_objection /= bbox_num
                curr_head_loss_attr_position /= bbox_num

                # 这里累积三个 head 的 sub_class_loss
                loss_sub_class += curr_head_loss_sub_class
                loss_attr_objection += curr_head_loss_attr_objection
                loss_attr_position += curr_head_loss_attr_position

                #------------------- 进行目标检测置信度回归 ------------------------------
                # 使用某个网格的某个anchor的预测框和标注框的（使用self.gr修正过） ciou 的值
                # 来表征该 anchor 的真实 Objectness，从而优化网络输出的 objectness 值，
                # 值得注意的是，由于 ciou 的计算与输出的 xywh 有关，为了避免反向传播影响网络输出 xywh 值，
                # 需要在 curr_head_pred_positive_iou 后增加 detach() 操作，让它从计算图中脱离，跟它相关的
                # 的 xywh 都不会在 objcetness loss 的反向传播中得到影响
                curr_head_obj_mat[curr_head_targets_imgIds, \
                                    curr_head_targets_anchor_indices, \
                                    curr_head_targets_y_offsets_grids_indices, \
                                    curr_head_targets_x_offsets_grids_indices] \
                = (1.0 - self.gr) + self.gr * curr_head_pred_positive_iou.detach().clamp(0).type(curr_head_obj_mat.dtype)  

            # 通过所有预测框的 objectness 值和真实标注框通过 iou 计算出的 objectness 值计算 obj_loss
            curr_head_obj_loss_mat = self.BCEobj(curr_head_preds_mat[..., 4], curr_head_obj_mat)
            # 三个头的 objectness 需要加权重后相加
            loss_objection += curr_head_obj_loss_mat * self.balance[head_index]  # obj loss

            # ???
            if self.autobalance:
                self.balance[head_index] = self.balance[head_index] * 0.9999 + 0.0001 / curr_head_obj_loss_mat.detach().item()

        # ???
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 三个 loss 分别乘上自己的权重后相加就是最后的总的 loss
        loss_box_position *= self.hyp['box']
        loss_objection *= self.hyp['obj']
        loss_class *= self.hyp['cls']
        loss_attr_objection *= self.hyp['attr_obj']
        loss_sub_class *= self.hyp['sub_cls']
        loss_attr_position *= self.hyp['attr_box']
        loss = loss_box_position + loss_objection + loss_class + loss_attr_objection + loss_sub_class + loss_attr_position

        # 总的 loss 要乘上 batchs_size
        batch_size = curr_head_obj_mat.shape[0]  

        return loss * batch_size, torch.cat((loss_box_position, loss_objection, loss_class, loss_attr_objection, loss_sub_class, loss_attr_position)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # p 是网络的预测输出，是一个长度为 3 的list，里面每个元素的尺寸为：（batch_size, anchor_num, feature_map_w, feature_map_h, [x y w h obj cls_one_hot]）
        # targets 是数据增强后所有的 bbox，尺寸为：（bbox_num, 
        #                                       [img_index cls_index x y w h 
        #                                       车辆类别（3），左车灯是否可见（1），左车灯状态（2），左车灯位置（4），
        #                                       右车灯是否可见（1），右车灯状态（2），右车灯位置（4），
        #                                       车屁股是否可见（1），车屁股框位置（4）
        #                                       车头是否可见（1），车头框位置（4）
        #                                       车牌是否可见（1），车牌位置（4），
        #                                       人的类别（2），人的状态（2）， 是否带头盔（2）
        #                                       人头是否可见（1），人头的位置（4），
        #                                       人骑的车是否可见（1），人骑的车的位置（4）
        #                                       电动车车牌是否可见（1），车牌的位置（4），
        #                                       红绿灯的颜色（3），
        #                                       路牌的颜色（2），
        #                                       公交车道第一个点坐标（2），公交车道第二个点坐标（2），公交车道第三个点坐标（2），公交车道第四个点坐标（2），
        #                                       斑马线第一个点坐标（2），斑马线第二个点坐标（2），斑马线第三个点坐标（2），斑马线第四个点坐标（2），
        #                                       网格线第一个点坐标（2），网格线第二个点坐标（2），网格线第三个点坐标（2），网格线第四个点坐标（2），
        #                                       导流线第一个点坐标（2），导流线第二个点坐标（2），导流线第三个点坐标（2），导流线第四个点坐标（2）
        #                                       ]）
        # 注意：里面的 x、y 都是归一化之后的值，指占图像长度/宽度的比例

        # 单个 head 的 anchor 的个数
        num_anchor = self.na

        # 当前图片中所有标签框的个数
        bbox_num = targets.shape[0]

        # 使用缩放系数将标签中的原图尺度转换到特征图尺度
        # 这里的 87 个位置是为了与 targets[:,:,87] 这个相乘，用来进行广播
        # 87 个位置的的 2：6 是尺寸信息，需要缩放，其他的都置 1 表示不变
        curr_head_feature_map_size = torch.ones(87, device=targets.device).long()  

        # 和当前图片中所有标签框的个数一样多的anchor索引值
        # 0 0 0 0 0 0 ........ 0 0 0 ...(列数是 bbox_num)
        # 1 1 1 1 1 1 ........ 1 1 1 ...
        # 2 2 2 2 2 2 ........ 2 2 2 ...
        # 尺寸是（anchor_num bbox_num）
        anchors_indices = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, bbox_num) 
        
        # targets 的 shape 从（bbox_num, [img_index cls_index x y w h]）
        #                变成（anchor_num, bbox_num, [img_index cls_index x y w h anchor_index]）
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), anchors_indices[:, :, None]), 2)  # append anchor indices

        # 偏移系数
        abs_offset = 0.5  
        # 偏移量
        offset = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  
                            # [1, 1], [1, -1], [-1, 1], [-1, -1], 
                            ], device=targets.device).float() * abs_offset

        total_positive_classes, total_positive_bboxes, total_positive_indices, total_positive_anchors, total_positive_targets = [], [], [], [], []

        # 遍历三个检测头
        for head_index in range(self.nl):
            # 获取当前 head 的三个 anchor，这里的 anchors 都是缩放到当前特征图尺度过的
            anchors = self.anchors[head_index]
            # anchors 尺寸：(3,2)

            # 获取当前 head 的 feature_map 尺寸（缩放系数),用来将 targets 进行缩放到当前尺度
            curr_head_feature_map_size[2:6] = torch.tensor(p[head_index].shape)[[3, 2, 3, 2]]
            # curr_head_feature_map_size 尺寸：（1, 1,feature_map_w,feature_map_h,feature_map_w,feature_map_h,1）

            # 标注框在特征图上坐标 = 标注框归一化之后的坐标 × 特征图的尺寸（缩放系数）
            curr_head_targets = targets * curr_head_feature_map_size
            # curr_head_targets 尺寸为：（anchor_num, bbox_num, [img_index cls_index x y w h anchor_index]）

            # 如果当前图片有目标框,则不是背景图片
            if bbox_num > 0:

                # 第一遍筛选：高宽比筛选
                # 计算每个标注框和三个anchor的高宽比
                # curr_head_targets[:, :, 4:6] 尺寸是 （anchor_num bbox_num 2）
                # anchors[:, None] 尺寸是（anchor_num 1 2）
                # 两者进行广播操作，得到当前 head 的所有的标注框分别与三个 anchor 的高宽比
                target_anchor_wh_ratio = curr_head_targets[:, :, 4:6] / anchors[:, None] 
                # target_anchor_wh_ratio 尺寸:（anchor_num bbox_num [w_ratio h_ratio]）

                # 这里指每个标注框和三个anchor的高宽比是否小于阈值（例如 4）的 mask
                # 含义是某个标注框尺寸不能比 anchor 框大于 4 倍或者小于 1 / 4
                target_anchor_wh_ratio_mask = torch.max(target_anchor_wh_ratio, 1. / target_anchor_wh_ratio).max(2)[0] < self.hyp['anchor_t']  # compare
                # target_anchor_wh_ratio_mask 尺寸:（anchor_num bbox_num 1）

                # 进行过滤操作，把不符合的 anchor 行都去掉，最后地一个 anchor_num 维度消失
                curr_head_targets = curr_head_targets[target_anchor_wh_ratio_mask]  # filter
                # curr_head_targets 尺寸：（bbox_num, [img_index cls_index x y w h anchor_index]）

                # 第二遍筛选，进行偏移，不仅使用目标中心点所在网格预测目标，还是用距离目标中心最近的两个方格
                # 当前 head 下所有标注框在特征图尺度下的坐标 xy
                curr_head_targets_xy = curr_head_targets[:, 2:4]  
                # curr_head_targets_xy 尺寸为：(bbox_num, [x y])

                # 当前 head 下所有标注框在特征图尺度下的反向坐标 
                curr_head_targets_inverse_xy = curr_head_feature_map_size[[2, 3]] - curr_head_targets_xy 
                # curr_head_targets_inverse_xy 尺寸为：(bbox_num, [invers_x inverse_y])

                # 距离当前格子左上角较近的中心点，并且不是位于边缘格子内的标注框的 mask
                # 分别获取到 x 是否靠近左上角 mask，y 是否靠近左上角 mask
                x_leftup_mask, y_leftup_mask = ((curr_head_targets_xy % 1. < abs_offset) & (curr_head_targets_xy > 1.)).T
                # x_leftup_mask、y_leftup_mask 的尺寸：(bbox_num, 1)

                # 距离当前格子右下角较近的中心点，并且不是位于边缘格子内的标注框的 mask
                x_rightbottom_mask, y_rightbottom_mask = ((curr_head_targets_inverse_xy % 1. < abs_offset) & (curr_head_targets_inverse_xy > 1.)).T
                # x_rightbottom_mask、y_rightbottom_mask 的尺寸：(bbox_num, 1)

                # 用来过滤每个标注框靠近是否靠近5个方向的 mask，“torch.ones_like(x_leftup_mask)” 代表不偏移
                all_direction_mask = torch.stack((torch.ones_like(x_leftup_mask), x_leftup_mask, y_leftup_mask, x_rightbottom_mask, y_rightbottom_mask))
                # all_direction_mask 的尺寸为: (5, bbox_num)

                # 将所有标注框复制五份，使用 all_direction_mask 来过滤
                # 过滤出靠近不同方向的标注框,找出最接近的两个网格，还有自己本网格，一共三个有效网格
                # curr_head_targets.repeat((5, 1, 1)) 尺寸为：(5, bbox_num, 7)
                # 这里注意，标注框还未进行偏移，有三分之二的重复框
                curr_head_targets = curr_head_targets.repeat((5, 1, 1))[all_direction_mask]
                # curr_head_targets尺寸为: (3*bbox_num, 7)

                # 生成有效的偏移矩阵，一共三个方向
                offsets = (torch.zeros_like(curr_head_targets_xy)[None] + offset[:, None])[all_direction_mask]
                # offsets 尺寸为：(3*bbox_num, 2)
            
            # 如果当前图片没有目标框，是背景
            else:
                curr_head_targets = targets[0]
                offsets = 0

            # 获取当前 head 所有标签对应的图片索引和类别
            curr_head_targets_imgIds, curr_head_targets_classes = curr_head_targets[:, :2].long().T 
            # curr_head_targets_imgIds 尺寸为：（1, bbox_num）
            
            # 获取当前 head 所有标注框在特征图尺度下的坐标 
            curr_head_targets_xy = curr_head_targets[:, 2:4]
            # curr_head_targets_xy 尺寸:(3*bbox_num, 2)

            # 获取当前 head 所有标注框在特征图尺度下的长宽
            curr_head_targets_wh = curr_head_targets[:, 4:6]  
            # curr_head_targets_wh 尺寸:(3*bbox_num, 2)

            # 把当前 head 所有标注框在三个有效方向上进行真正的偏移，得到偏移后的三倍数量的标注框
            # curr_head_targets_xy - offsets 得到偏移后格子坐标，long（）取整数得到可以预测正样本的格子的整数索引号
            curr_head_targets_offsets_grids_indices = (curr_head_targets_xy - offsets).long()
            # 分成 x y 方向的偏移到的网格索引值
            curr_head_targets_x_offsets_grids_indices, curr_head_targets_y_offsets_grids_indices = curr_head_targets_offsets_grids_indices.T  # grid xy indices
            # curr_head_targets_x_offsets_grids_indices 尺寸:(3*bbox_num, 1)

            # 获取当前 head 所有标注框在第一步筛选后匹配到的 anchor 序号
            curr_head_targets_anchor_indices = curr_head_targets[:, 6].long()  # anchor indices
            # curr_head_targets_anchor_indices: (3*bbox_num, 1)

            # 所有正样本对应的索引：图像 id、合适的anchor索引、y 方向偏移到的格子的索引、x 方向偏移到的格子的索引
            total_positive_indices.append((curr_head_targets_imgIds, curr_head_targets_anchor_indices, curr_head_targets_y_offsets_grids_indices.clamp_(0, curr_head_feature_map_size[3] - 1), curr_head_targets_x_offsets_grids_indices.clamp_(0, curr_head_feature_map_size[2] - 1)))  # image, anchor, grid indices
            # 所有正样本对应的 bbox  
            total_positive_bboxes.append(torch.cat((curr_head_targets_xy - curr_head_targets_offsets_grids_indices, curr_head_targets_wh), 1))  # box
            # 所有正样本对应的 anchor 
            total_positive_anchors.append(anchors[curr_head_targets_anchor_indices])  # anchors
            # 所有正样本对应的类别 
            total_positive_classes.append(curr_head_targets_classes)  # class
            # 所有正样本
            total_positive_targets.append(curr_head_targets)

        return total_positive_classes, total_positive_bboxes, total_positive_indices, total_positive_anchors, total_positive_targets



class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
    

class ComputeLossBinOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossBinOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #MSEangle = nn.MSELoss().to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride', 'bin_count':
            setattr(self, k, getattr(det, k))

        #xy_bin_sigmoid = SigmoidBin(bin_count=11, min=-0.5, max=1.5, use_loss_regression=False).to(device)
        wh_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0, use_loss_regression=False).to(device)
        #angle_bin_sigmoid = SigmoidBin(bin_count=31, min=-1.1, max=1.1, use_loss_regression=False).to(device)
        self.wh_bin_sigmoid = wh_bin_sigmoid

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2     # x,y, w-bce, h-bce     # xy_bin_sigmoid.get_length()*2

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                
                #pxy = ps[:, :2].sigmoid() * 2. - 0.5
                ##pxy = ps[:, :2].sigmoid() * 3. - 1.
                #pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                #pbox = torch.cat((pxy, pwh), 1)  # predicted box

                #x_loss, px = xy_bin_sigmoid.training_loss(ps[..., 0:12], tbox[i][..., 0])
                #y_loss, py = xy_bin_sigmoid.training_loss(ps[..., 12:24], tbox[i][..., 1])
                w_loss, pw = self.wh_bin_sigmoid.training_loss(ps[..., 2:(3+self.bin_count)], selected_tbox[..., 2] / anchors[i][..., 0])
                h_loss, ph = self.wh_bin_sigmoid.training_loss(ps[..., (3+self.bin_count):obj_idx], selected_tbox[..., 3] / anchors[i][..., 1])

                pw *= anchors[i][..., 0]
                ph *= anchors[i][..., 1]

                px = ps[:, 0].sigmoid() * 2. - 0.5
                py = ps[:, 1].sigmoid() * 2. - 0.5

                lbox += w_loss + h_loss # + x_loss + y_loss

                #print(f"\n px = {px.shape}, py = {py.shape}, pw = {pw.shape}, ph = {ph.shape} \n")

                pbox = torch.cat((px.unsqueeze(1), py.unsqueeze(1), pw.unsqueeze(1), ph.unsqueeze(1)), 1).to(device)  # predicted box

                
                
                
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, (1+obj_idx):], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, (1+obj_idx):], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., obj_idx], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, obj_idx:(obj_idx+1)])
                p_cls.append(fg_pred[:, (obj_idx+1):])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pw = self.wh_bin_sigmoid.forward(fg_pred[..., 2:(3+self.bin_count)].sigmoid()) * anch[i][idx][:, 0] * self.stride[i]
                ph = self.wh_bin_sigmoid.forward(fg_pred[..., (3+self.bin_count):obj_idx].sigmoid()) * anch[i][idx][:, 1] * self.stride[i]
                
                pxywh = torch.cat([pxy, pw.unsqueeze(1), ph.unsqueeze(1)], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]            
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs       

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossAuxOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.nl], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]] 
    

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                #pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i] # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        
        indices, anch = self.find_3_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device="cpu")

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def build_targets2(self, p, targets, imgs):
        
        indices, anch = self.find_5_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device="cpu")

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs              

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 1.0  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch                 

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
