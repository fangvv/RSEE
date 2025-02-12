from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
import ops.parallel_resnet as parallel_models
from .parallel import ModuleParallel, BatchNorm2dParallel
from ops.net_flops_table import feat_dim_dict
from ops.blstm import BLSTM_IRTENet, MaxPooling, AvePooling, STAN_LSTM

from torch.distributions import Categorical


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

def init_hidden_lstm(layers, batch_size, cell_size):
    init_cell = torch.Tensor(layers, batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

class RSEE(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', consensus_type='avg', before_softmax=True, dropout=0.8,
                 crop_num=1, partial_bn=True, pretrain='imagenet', fc_lr5=False, args=None):
        super(RSEE, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.pretrain = pretrain

        self.fc_lr5 = fc_lr5

        # TODO(yue)
        self.args = args
        self.rescale_to = args.rescale_to
        if self.args.ada_reso_skip:
            base_model = self.args.backbone_list[0] if len(self.args.backbone_list) >= 1 else None
        self.base_model_name = base_model
        self.num_class = num_class
        self.multi_models = False
        self.time_steps = self.num_segments

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.args.ada_reso_skip:
            self.reso_dim = self._get_resolution_dimension()
            self.skip_dim = len(self.args.skip_list)
            self.action_dim = self._get_action_dimension()
            self._prepare_policy_net()
            self._extends_to_multi_models()
        if self.args.blstm:
            # TODO set this feat_dim adaptive
            blstm_feat_dim = feat_dim_dict["resnet50"]
            self.blstm = self._get_blstm(
                input_dim=blstm_feat_dim, 
                hidden_dim=self.args.blstm_hidden_dim,
                num_classes=num_class
                ) # [2048, 512, ]
        elif self.args.stan_lstm:
            blstm_feat_dim = feat_dim_dict["resnet50"]
            self.stan_lstm = self._get_stan_lstm(
                input_dim=blstm_feat_dim,
                hidden_dim=self.args.blstm_hidden_dim // 2,
                num_classes=num_class
            )
            self.K = 3
            self.stan_alpha = nn.Parameter(torch.ones(self.num_segments - self.K + 1), requires_grad=True)
        if self.args.frm == 'max':
            self.frm = MaxPooling()
        elif self.args.frm == 'ave':
            self.frm = AvePooling()

        self._prepare_base_model(base_model)
        self._prepare_fc(num_class)

        self.consensus = ConsensusModule(consensus_type, args=self.args)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)


    def _get_blstm(self, **kwargs):
        return BLSTM_IRTENet(**kwargs)
    
    def _get_stan_lstm(self, **kwargs):
        return STAN_LSTM(**kwargs)


    def _extends_to_multi_models(self):
        if len(self.args.backbone_list) >= 1:
            self.multi_models = True
            self.base_model_list = nn.ModuleList()
            self.new_fc_list = nn.ModuleList()

    def _prep_a_net(self, model_name, shall_pretrain):
        if "efficientnet" in model_name:
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
            model.last_layer_name = "_fc"
        else:
            if self.args.irte and not self.args.vanilla or self.args.irte_joint and model_name != 'mobilenet_v2' and model_name != 'resnet18':
                # HACK for irte
                model = parallel_models.__dict__[model_name](num_classes=2048, num_parallel=len(self.args.reso_list))
                print(f"Using model {model_name}")
            else:
                model = getattr(torchvision.models, model_name)(shall_pretrain)
            if "resnet" in model_name:
                model.last_layer_name = 'fc'
            elif "mobilenet_v2" in model_name or 'vgg16' in model_name:
                model.last_layer_name = 'classifier'
        return model

    def _get_resolution_dimension(self):
        # reso_dim = 0
        # for i in range(len(self.args.backbone_list)):
        #     reso_dim += self.args.ada_crop_list[i]
        # if self.args.policy_also_backbone:
        #     reso_dim += 1
        reso_dim = len(self.args.reso_list)
        return reso_dim

    def _get_action_dimension(self):
        action_dim = self.reso_dim + self.skip_dim
        return action_dim

    def _prepare_policy_net(self):
        shall_pretrain = not self.args.policy_from_scratch
        self.lite_backbone = self._prep_a_net(self.args.policy_backbone, shall_pretrain)
        self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
        self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.args.hidden_dim, bias=True)

    def _prepare_base_model(self, base_model):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if self.args.ada_reso_skip:
            shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
            for bbi, backbone_name in enumerate(self.args.backbone_list):
                model = self._prep_a_net(backbone_name, True)
                self.base_model_list.append(model)
        else:
            self.base_model = self._prep_a_net(base_model, True)

    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim, isParallel=True):
            if isParallel:
                # HACK for irte
                linear_model = ModuleParallel(nn.Linear(input_dim, output_dim))
                normal_(linear_model.module.weight, 0, 0.001)
                constant_(linear_model.module.bias, 0)
            else:
                linear_model = nn.Linear(input_dim, output_dim)
                normal_(linear_model.weight, 0, 0.001)
                constant_(linear_model.bias, 0)
            return linear_model

        i_do_need_a_policy_network = True

        if self.args.ada_reso_skip and i_do_need_a_policy_network:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.dropout))
            feed_dim = self.args.hidden_dim if not self.args.frame_independent else self.policy_feat_dim
            self.linear = make_a_linear(feed_dim, self.action_dim, isParallel=False) # selection linear
            self.lite_fc = make_a_linear(feed_dim, num_class, isParallel=False)

        if self.multi_models:
            multi_fc_list = [None]
            for bbi, base_model in enumerate(self.base_model_list):
                for fc_i, exit_index in enumerate(multi_fc_list):
                    last_layer_name = base_model.last_layer_name
                    if self.args.irte_joint:
                        feature_dim = getattr(base_model, last_layer_name).module.in_features
                    else:
                        feature_dim = getattr(base_model, last_layer_name).in_features

                    new_fc = make_a_linear(feature_dim, num_class)
                    self.new_fc_list.append(new_fc)
                    if self.args.baseline or self.args.vanilla:
                        setattr(base_model, last_layer_name, nn.Dropout(p=self.dropout))
                    # elif self.args.irte_joint:
                    #     setattr(base_model, last_layer_name, ModuleParallel(nn.Dropout(p=self.dropout)))

        elif self.base_model_name is not None:
            # TODO get feature_dim
            if "mobilenet_v2" == self.base_model_name:
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
            elif "vgg16" == self.base_model_name:
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[6].in_features
            else:
                if self.args.irte and not self.args.vanilla:
                    # HACK for irte
                    feature_dim = getattr(self.base_model, self.base_model.last_layer_name).module.in_features
                else:
                    feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features

            # TODO make a new classifier
            if self.args.baseline or self.args.vanilla:
                if self.base_model_name == 'vgg16':
                    self.base_model.classifier[6] = nn.Sequential()
                else:
                    setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
                self.new_fc = make_a_linear(feature_dim, num_class, isParallel=False)
            elif self.args.irte_joint:
                # setattr(base_model, last_layer_name, ModuleParallel(nn.Dropout(p=self.dropout)))
                self.new_fc = make_a_linear(feature_dim, num_class, isParallel=True)
            else:
                self.new_fc = make_a_linear(feature_dim, num_class, isParallel=True)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(RSEE, self).train(mode)
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            if self.args.ada_reso_skip:
                models = [self.lite_backbone]
                if self.multi_models:
                    models = models + self.base_model_list
            else:
                models = [self.base_model]

            for the_model in models:
                count = 0
                bn_scale = 1
                for m in the_model.modules():
                    if isinstance(m, nn.BatchNorm2d):  # TODO(yue)
                        count += 1
                        if count >= (2 * bn_scale if self._enable_pbn else bn_scale):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])
            elif isinstance(m, torch.nn.modules.rnn.LSTM):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def backbone(self, input_data, the_base_model, new_fc, signal=-1, indices_list=[], boost=False, b_t_c=False,
                 **kwargs):
        _b, _tc, _h, _w = input_data.shape  # TODO(yue) input (B, T*C, H, W)
        _t, _c = _tc // 3, 3

        if b_t_c:
            input_b_t_c = input_data.view(_b, _t, _c, _h, _w)
        else:
            input_2d = input_data.view(_b * _t, _c, _h, _w)

        if b_t_c:
            feat = the_base_model(input_b_t_c, signal=signal, **kwargs)
        else:
            feat = the_base_model(input_2d)

        _base_out = None
        if b_t_c:
            if new_fc is not None:
                _base_out = new_fc(feat.view(_b * _t, -1)).view(_b, _t, -1)
        else:
            if new_fc is not None:
                _base_out = new_fc(feat).view(_b, _t, -1)
            feat = feat.view(_b, _t, -1)
        return feat, _base_out
    
    def backbone_irte(self, input_data, the_base_model, new_fc, signal=-1, b_t_c=False,
                        **kwargs):
        _base_outs = []
        for i in range(len(input_data)):
            # XXX for every reso
            _b, _tc, _h, _w = input_data[i].shape  # TODO(yue) input (B, T*C, H, W)
            _t, _c = _tc // 3, 3

            if b_t_c:
                input_data[i] = input_data[i].view(_b, _t, _c, _h, _w)
            else:
                input_data[i] = input_data[i].view(_b * _t, _c, _h, _w)

        if self.args.vanilla:
            feats = []
            _base_outs = []
            for i in range(len(input_data)):
                out_feat = the_base_model(input_data[i])
                _base_out = new_fc(out_feat)
                feats.append(out_feat)
                _base_outs.append(_base_out.view(_b, _t, -1))

        else:
            if b_t_c:
                feats = the_base_model(input_data, signal=signal, **kwargs)
            else:
                if self.args.baseline:
                    feats = the_base_model(input_data[0])
                    feats = [feats]
                elif 'spec' in kwargs: # TODO for BLSTM inference
                    feats = the_base_model(input_data, **kwargs)
                else:
                    feats = the_base_model(input_data)

            if b_t_c:
                if new_fc is not None:
                    _base_out = new_fc(feats.view(_b * _t, -1)).view(_b, _t, -1)
            else:
                if new_fc is not None:
                    _base_outs = new_fc(feats)
                    _base_outs = [item.view(_b, _t, -1) for item in _base_outs]
                feats = [item.view(_b, _t, -1) for item in feats]
        return feats, _base_outs


    def get_lite_j_and_r(self, input_list, online_policy, tau):

        feat_lite, _ = self.backbone(input_list[self.args.policy_input_offset], self.lite_backbone, None)
        # feat_lite: shape [b, t, 1280]
        r_list = []
        lite_j_list = []
        batch_size = feat_lite.shape[0]
        hx = init_hidden(batch_size, self.args.hidden_dim)
        cx = init_hidden(batch_size, self.args.hidden_dim)

        remain_skip_vector = torch.zeros(batch_size, 1)
        old_hx = None
        old_r_t = None

        if self.args.use_reinforce:
            log_prob_r_list = []
            prob_r_list = []

        for t in range(self.time_steps):
            if self.args.frame_independent:
                feat_t = feat_lite[:, t]
            else:
                hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
                feat_t = hx
            if self.args.use_reinforce:
                p_t = F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8)
            else:
                p_t = torch.log(F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8)) # [16, 4]
            j_t = self.lite_fc(feat_t) # [16, 51]
            lite_j_list.append(j_t)  # TODO as pred 84x84 action class, 每一帧的[b, classes]

            # TODO (yue) need a simple case to illustrate this
            if online_policy:
                if self.args.use_reinforce:
                    m = Categorical(p_t)

                    prob_r_list.append(p_t)

                    r_t_idx = m.sample()
                    r_t = torch.eye(self.action_dim)[r_t_idx].cuda()
                    log_prob_r_t = m.log_prob(r_t_idx)
                    log_prob_r_list.append(log_prob_r_t)
                else:
                    r_t = torch.cat(
                        # [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])]
                        [self.gumbel_softmax(True, p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])]
                        ) # [16, 4]

                # TODO update states and r_t
                if old_hx is not None:
                    take_bool = remain_skip_vector > 0.5
                    take_old = torch.tensor(take_bool, dtype=torch.float).to(self.device)
                    take_curr = torch.tensor(~take_bool, dtype=torch.float).to(self.device)
                    hx = old_hx * take_old + hx * take_curr
                    r_t = old_r_t * take_old + r_t * take_curr

                # TODO update skipping_vector
                for batch_i in range(batch_size):
                    for skip_i in range(self.action_dim - self.reso_dim):
                        # TODO(yue) first condition to avoid valuing skip vector forever
                        if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][self.reso_dim + skip_i] > 0.5:
                            remain_skip_vector[batch_i][0] = self.args.skip_list[skip_i]
                old_hx = hx
                old_r_t = r_t
                r_list.append(r_t)  # TODO as decision
                remain_skip_vector = (remain_skip_vector - 1).clamp(0)
        if online_policy:
            if self.args.use_reinforce:
                return lite_j_list, torch.stack(r_list, dim=1), torch.stack(log_prob_r_list, dim=1)
            else:
                return lite_j_list, torch.stack(r_list, dim=1)
        else:
            return lite_j_list, None

    def using_online_policy(self):
        if any([self.args.offline_lstm_all, self.args.offline_lstm_last]):
            return False
        elif any([self.args.random_policy, self.args.all_policy]):
            return False
        elif self.args.real_scsampler:
            return False
        else:
            return True

    def input_fusion(self, input_data, r):
        # TODO data: B * TC * H * W
        # TODO r   : B * T * T
        _b, _tc, _h, _w = input_data.shape
        _c = _tc // self.args.num_segments
        fuse_data_list = []

        for bi in range(_b):
            if self.args.identity_prior:
                prior = torch.eye(self.args.num_segments).to(input_data.device)
            else:
                prior = 0
            if self.args.lower_mask:
                mask = torch.tril(torch.ones(self.args.num_segments, self.args.num_segments)).to(input_data.device)
            else:
                mask = 1
            real_r = (r[bi] + prior) * mask
            if self.args.direct_lower_mask:
                real_r = torch.tril(real_r)
            if self.args.row_normalization:
                real_r = real_r / (real_r.sum(dim=1, keepdim=True).clamp_min(1e-6))
            fused_data = torch.matmul(real_r, input_data[bi].view(self.args.num_segments, _c * _h * _w))
            fuse_data_list.append(fused_data)
        return torch.stack(fuse_data_list, dim=0).view(_b, _tc, _h, _w)

    def get_feat_and_pred(self, input_list, r_all, **kwargs):
        if self.args.irte_joint:
            for bb_i, the_backbone in enumerate(self.base_model_list):
                feat_out_list, base_out_list = self.backbone_irte(input_list, the_backbone, self.new_fc_list[bb_i])
            return feat_out_list, base_out_list, []
        else:
            feat_out_list = []
            base_out_list = []
            ind_list = []

            for bb_i, the_backbone in enumerate(self.base_model_list):
                feat_out, base_out = self.backbone(input_list[bb_i], the_backbone, self.new_fc_list[bb_i])
                feat_out_list.append(feat_out)
                base_out_list.append(base_out)
            return feat_out_list, base_out_list, ind_list

    def late_fusion(self, base_out_list, in_matrix, out_matrix):
        return base_out_list

    def forward(self, *argv, **kwargs):
        if self.args.irte:
            # HACK for irte backbone
            _, base_out = self.backbone_irte(kwargs["input"], self.base_model, self.new_fc,
                                            signal=self.args.default_signal)
            outputs = []
            for item in base_out:
                output = self.consensus(item)
                outputs += [output]
            return [item.squeeze(1) for item in outputs]
        elif not self.args.ada_reso_skip:  # TODO simple TSN
            _, base_out = self.backbone(kwargs["input"][0], self.base_model, self.new_fc,
                                        signal=self.args.default_signal)
            output = self.consensus(base_out)
            return output.squeeze(1)


        input_list = kwargs["input"]
        batch_size = input_list[0].shape[0]  # TODO(yue) input[0] B*(TC)*H*W

        if self.args.use_reinforce:
            lite_j_list, r_all, r_log_prob = self.get_lite_j_and_r(input_list, self.using_online_policy(),
                                                                kwargs["tau"])
        else:
            lite_j_list, r_all = self.get_lite_j_and_r(input_list, self.using_online_policy(), kwargs["tau"])
            if self.args.fake_r:
                r_all_left = torch.ones(batch_size, 1, 1)
                r_all_right = torch.zeros(batch_size, 1, self.action_dim - 1)
                r_all = torch.cat((r_all_left, r_all_right), dim=2).cuda()

        if self.multi_models:
            if "tau" not in kwargs:
                kwargs["tau"] = None

            feat_out_list, base_out_list, ind_list = self.get_feat_and_pred(input_list, r_all, tau=kwargs["tau"])
        else:
            feat_out_list, base_out_list, ind_list = [], [], []

        if self.args.policy_also_backbone:
            del base_out_list[-1]
            base_out_list.append(torch.stack(lite_j_list, dim=1))

        if self.args.offline_lstm_last:  # TODO(yue) no policy - use policy net as backbone - just LSTM(last)
            return lite_j_list[-1].squeeze(1), None, None, None

        elif self.args.offline_lstm_all:  # TODO(yue) no policy - use policy net as backbone - just LSTM(average)
            return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None, None, None

        elif self.args.real_scsampler:
            real_pred = base_out_list[0]
            lite_pred = torch.stack(lite_j_list, dim=1)
            output, ind = self.consensus(real_pred, lite_pred)
            return output.squeeze(1), ind, real_pred, lite_pred

        else:
            if self.args.random_policy:  # TODO(yue) random policy
                r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                for i_bs in range(batch_size):
                    for i_t in range(self.time_steps):
                        r_all[i_bs, i_t, torch.randint(self.action_dim, [1])] = 1.0
            elif self.args.all_policy:  # TODO(yue) all policy: take all
                r_all = torch.ones(batch_size, self.time_steps, self.action_dim).cuda()
            output = self.combine_logits(r_all, base_out_list, ind_list) # [B, num_classes]
            if self.args.save_meta and self.args.save_all_preds:
                return output.squeeze(1), r_all, torch.stack(base_out_list, dim=1)
            else:
                if self.args.use_reinforce:
                    return output.squeeze(1), r_all, r_log_prob, torch.stack(base_out_list, dim=1)
                else:
                    return output.squeeze(1), r_all, None, torch.stack(base_out_list, dim=1)
        
    def forward_irte(self, *argv, **kwargs):
        input_list = kwargs["input"]
        tau = kwargs["tau"]
        T = self.args.num_segments
        exit_log = []
        # TODO set this from kwargs
        isTraining = kwargs["isTraining"]
        # isTraining = False
        batch_size = input_list[0].shape[0]
        # TODO hx, cx for drs
        hx_drs = init_hidden(batch_size, self.args.hidden_dim)
        cx_drs = init_hidden(batch_size, self.args.hidden_dim)
        # TODO use different hidden_dim
        if self.args.blstm:
            hx = init_hidden(batch_size, self.args.blstm_hidden_dim) # [N, 512]
            cx = init_hidden(batch_size, self.args.blstm_hidden_dim)

        elif self.args.stan_lstm:
            hx = init_hidden_lstm(4, batch_size, self.args.blstm_hidden_dim // 2) # 4 = D * num_layers
            cx = init_hidden_lstm(4, batch_size, self.args.blstm_hidden_dim // 2)

        outputs = []
        r_list = []
        all_base_out_list = []

        # HACK we split input from [B, T*C, H, W] to [B, T, C, H, W]
        new_input_list = []
        for reso_input in input_list:
            _b, _tc, _h, _w = reso_input.shape
            _t, _c = _tc // 3, 3
            new_input_list.append(reso_input.view(_b, _t, _c, _h, _w))
        
        input_list = new_input_list
        if self.args.frm != 'none': # TODO for frm
            last_t_feat = None
        if self.args.stan_lstm:
            accumulate_k_feat = []
            K_h = []

        if isTraining:  # irte final training mode
            for t in range(T):
                last_reso_input = input_list[self.args.policy_input_offset]
                last_reso_tth_frame = last_reso_input[:, t] # No.t
                lite_j_list, r_all, hx_drs, cx_drs = self.get_lite_j_and_r_irte(
                    hx_drs, cx_drs,
                    last_reso_tth_frame,
                    self.using_online_policy(),
                    tau,
                    isTraining,
                    )
                # XXX fake r_all
                if self.args.fake_r:
                    r_all_mid = torch.ones(batch_size, 1, 1)
                    r_all_right = torch.zeros(batch_size, 1, self.action_dim - 1)
                    r_all = torch.cat((r_all_mid, r_all_right), dim=2).to(self.device)

                if self.multi_models:
                    if "tau" not in kwargs:
                        kwargs["tau"] = None
                    single_frame_input_list = [item[:, t] for item in input_list]
                    feat_out_list, base_out_list, ind_list = self.get_feat_and_pred_irte(
                        single_frame_input_list, r_all, tau=kwargs["tau"], isTraining=isTraining
                    )
                else:
                    feat_out_list, base_out_list, ind_list = [], [], []
                
                if self.args.policy_also_backbone:
                    del base_out_list[-1]
                    base_out_list.append(torch.stack(lite_j_list, dim=1))
                
                if self.args.offline_lstm_last:
                    return lite_j_list[-1].squeeze(1), None, None, None
                
                elif self.args.offline_lstm_all:
                    return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None, \
                    None, None
                else:
                    if self.args.random_policy:  # TODO(yue) random policy
                        r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                        for i_bs in range(batch_size):
                            for i_t in range(self.time_steps):
                                r_all[i_bs, i_t, torch.randint(self.action_dim, [1])] = 1.0
                    elif self.args.all_policy:  # TODO(yue) all policy: take all
                        r_all = torch.ones(batch_size, self.time_steps, self.action_dim).cuda()
                    

                    # TODO here we use feat instead of base_out_list
                    output = self.combine_logits(r_all, feat_out_list, ind_list) # [B, 2048]
                    # TODO use prev feat instead of logits
                    if self.args.blstm:
                        _exit, hx, cx = self.blstm(output, (hx, cx)) # exit: [b, 2]

                        # making temporal exiting decision
                        # _exit_prob = torch.log(F.softmax(_exit, dim=1).clamp(min=1e-8))
                        # _exit_ = torch.cat(
                        #         [F.gumbel_softmax(_exit_prob[b_i:b_i + 1], tau, True) for b_i in range(_exit_prob.shape[0])]
                        #     ) # [B, 2]

                        final_output = self.new_fc_list[0](hx)
                        final_output = torch.stack(final_output, dim=0)
                    elif self.args.stan_lstm:
                        # 1-2-3-4, 2-3-4-5, 3-4-5-6, ...
                        if t < self.K - 1:
                            accumulate_k_feat.append(output)
                            continue
                        else:
                            accumulate_k_feat.append(output)
                            accumulate_k_feat_tensor = torch.stack(accumulate_k_feat, dim=0)
                            _exit, output, hx, cx = self.stan_lstm(accumulate_k_feat_tensor, (hx, cx))
                            # making temporal exiting decision
                            _exit_prob = torch.log(F.softmax(_exit, dim=1).clamp(min=1e-8))
                            _exit_ = torch.cat(
                                    [F.gumbel_softmax(_exit_prob[b_i:b_i + 1], tau, True) for b_i in range(_exit_prob.shape[0])]
                                ) # [B, 2]
                            # final_output = self.blstm_fc(hx)
                            final_output = self.new_fc_list[0](output)
                            final_output = torch.stack(final_output, dim=0)
                            del accumulate_k_feat[0]
                    elif self.args.frm != 'none':
                        if t != 0:
                            output = self.frm((last_t_feat, output))
                        last_t_feat = output
                        final_output = self.new_fc_list[0](output) # [num_classes]
                        final_output = torch.stack(final_output, dim=0)
                        _exit_ = torch.ones(batch_size, 2) # TODO fake data here
                    else:
                        final_output = self.new_fc_list[0](output)
                        final_output = torch.stack(final_output, dim=0)
                    _exit_ = torch.tensor((0, 1)) # fake
                    exit_log.append(_exit_)
                    # XXX Accumulated
                    # if len(outputs) != 0: # 每一帧的输出是之前所有帧的平均
                    #     curr_t_out = torch.cat(
                    #         (outputs[t - 1].unsqueeze(0), final_output.unsqueeze(0)), dim=0
                    #         ).mean(0) 
                    #     outputs.append(curr_t_out)
                    # else:
                    #     outputs.append(final_output)
                    # Accumulated done
                    outputs.append(final_output)
                    r_list.append(r_all)
                    base_out_tensor = torch.stack(base_out_list, dim=1)
                    all_base_out_list.append(base_out_tensor)
            if self.args.stan_lstm:
                outputs_ens = torch.zeros_like(outputs[0])
                for i in range(len(outputs)):
                    outputs_ens += self.stan_alpha[i] * outputs[i]
                return outputs_ens[-1].unsqueeze(1), torch.stack(r_list, dim=1).squeeze(2), exit_log, all_base_out_list
            return torch.stack(outputs, dim=1), torch.stack(r_list, dim=1).squeeze(2), exit_log, all_base_out_list
        else:
            outputs = []
            for t in range(T):
                last_reso_input = input_list[self.args.policy_input_offset]
                last_reso_tth_frame = last_reso_input[:, t]
                lite_j_list, r_all, hx_drs, cx_drs = self.get_lite_j_and_r_irte(
                    hx_drs, cx_drs,
                    last_reso_tth_frame,
                    self.using_online_policy(),
                    kwargs["tau"],
                    isTraining,
                    )

                # XXX fake r_all
                if self.args.fake_r:
                    r_all_mid = torch.ones(batch_size, 1, 1)
                    r_all_right = torch.zeros(batch_size, 1, self.action_dim - 1)
                    r_all = torch.cat((r_all_mid, r_all_right), dim=2).to(self.device)

                if self.multi_models:
                    if "tau" not in kwargs:
                        kwargs["tau"] = None
                    chosen_reso = torch.where(r_all[0, 0] == 1)[0][0].item()
                    single_frame_input_list = [input_list[chosen_reso][:, t]]
                    feat_out_list, base_out_list, ind_list = self.get_feat_and_pred_irte(
                        single_frame_input_list, r_all, tau=kwargs["tau"], spec=chosen_reso
                    )
                if self.args.policy_also_backbone and not self.args.irte_final:
                    base_out_list.append(torch.stack(lite_j_list, dim=1))
                
                if self.args.offline_lstm_last:
                    return lite_j_list[-1].squeeze(1), None, None, None
                
                elif self.args.offline_lstm_all:
                    return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None, \
                    None, None
                elif self.args.blstm:
                    # feat_out_list[0] [B, T, 2048]
                    output = self.combine_logits(r_all, feat_out_list, ind_list) # [B, 2048]
                    _exit, hx, cx = self.blstm(output, (hx, cx))
                    # _exit_prob = torch.log(F.softmax(_exit, dim=1).clamp(min=1e-8))
                    # _exit_ = torch.cat(
                    #         [F.gumbel_softmax(_exit_prob[b_i:b_i + 1], tau, True) for b_i in range(_exit_prob.shape[0])]
                    #     ) # [B, 2]
                    # final_output = self.blstm_fc(hx)
                    final_output = self.new_fc_list[0](hx)
                    final_output = torch.stack(final_output, dim=0)
                    # XXX Accumulated
                    # if len(outputs) != 0:
                    #     curr_t_out = torch.cat(
                    #         (outputs[t - 1].unsqueeze(0), final_output.unsqueeze(0)),
                    #         dim=0).mean(0)
                    #     outputs.append(curr_t_out)
                    # else:
                    #     outputs.append(final_output)
                    # Accumulated done
                    outputs.append(final_output)
                    r_list.append(r_all)
                    confidence, _ = torch.max(nn.Softmax()(final_output), dim=1)
                    if confidence >= self.args.TE or t == T - 1:
                    # if t == T - 1:
                        return outputs[-1], torch.stack(r_list, dim=1).squeeze(2), \
                            t + 1, None
                elif self.args.stan_lstm:
                    output_ext = feat_out_list[0].squeeze(1).unsqueeze(0)
                    _exit, output, hx, cx = self.stan_lstm(output_ext, (hx, cx))
                    _exit_prob = torch.log(F.softmax(_exit, dim=1).clamp(min=1e-8))
                    _exit_ = torch.cat(
                            [F.gumbel_softmax(_exit_prob[b_i:b_i + 1], tau, True) for b_i in range(_exit_prob.shape[0])]
                        ) # [B, 2]
                    final_output = self.new_fc_list[0](output)
                    final_output = torch.stack(final_output, dim=0).squeeze(0)
                    outputs.append(final_output)
                    r_list.append(r_all)
                    if _exit_[:, 1] == 1 or t == T - 1:
                    # if t == T - 1:
                        return outputs[-1], torch.stack(r_list, dim=1).squeeze(2), \
                            t + 1, None
                elif self.args.frm != 'none': # TODO frm
                    _exit_ = torch.ones(batch_size, 2) # fake data
                    if t != 0:
                        output = self.frm((last_t_feat, output))
                    else:
                        output = feat_out_list[0].squeeze(1)
                    last_t_feat = output
                    final_output = self.new_fc_list[0](output)
                    final_output = torch.stack(final_output, dim=0)

                    r_list.append(r_all)
                    if t == T - 1:
                        return final_output, torch.stack(r_list, dim=1).squeeze(2), \
                            t + 1, None
                else:
                    output = self.combine_logits(r_all, feat_out_list, ind_list) # [B, 2048]
                    final_output = self.new_fc_list[0](output)
                    final_output = torch.stack(final_output, dim=0)
                    outputs.append(final_output)
                    r_list.append(r_all)
                    confidence, _ = torch.max(nn.Softmax()(final_output), dim=1)
                    if confidence >= self.args.TE or t == T - 1:
                    # if t == T - 1:
                        return outputs[-1], torch.stack(r_list, dim=1).squeeze(2), \
                            t + 1, None


    def forward_bk_ol(self, *args, **kwargs):
        input_list = kwargs['input']
        tau = kwargs['tau']
        T = self.args.num_segments
        exit_log = []
        batch_size = input_list[0].shape[0]
        reso_num = len(self.args.reso_list)

        outputs = []

        new_input_list = []
        for reso_input in input_list:
            _b, _tc, _h, _w = reso_input.shape
            _t, _c = _tc // 3, 3
            new_input_list.append(reso_input.view(_b, _t, _c, _h, _w))
        input_list = new_input_list
        if self.args.frm != 'none':
            last_t_feat = None
        for r in range(len(self.args.reso_list)):
            hx = init_hidden(batch_size, self.args.blstm_hidden_dim)
            cx = init_hidden(batch_size, self.args.blstm_hidden_dim)
            reso_out_list = []
            for t in range(T):
                single_frame_input_list = [input_list[r][:, t]]
                # single_frame_input_list = [item[:, t] for item in input_list]
                feat_out_list, base_out_list = self.backbone_irte(
                    single_frame_input_list, self.base_model, self.new_fc, spec=r
                )
                cur_reso_feat = feat_out_list[0].squeeze(1) # [B, 2048]
                if self.args.blstm:
                    _exit, hx, cx = self.blstm(cur_reso_feat, (hx, cx))
                    _exit_prob = torch.log(F.softmax(_exit, dim=1).clamp(min=1e-8))
                    _exit_ = torch.cat(
                            [F.gumbel_softmax(_exit_prob[b_i:b_i + 1], tau, True) for b_i in range(_exit_prob.shape[0])]
                        ) # [B, 2]
                    final_output = self.new_fc_list[0](hx)
                else:
                    _exit_ = torch.ones(batch_size, 2) # fake data
                    if t != 0:
                        output = self.frm((last_t_feat, output))
                    else:
                        output = feat_out_list[0].squeeze(1)
                    last_t_feat = output
                    final_output = self.new_fc(output)
                    final_output = torch.stack(final_output, dim=0)
                reso_out_list.append(final_output)
            exit_log.append(_exit_)
            outputs.append(reso_out_list)
        return outputs, exit_log


    def get_feat_and_pred_irte(self, input_list, r_all, **kwargs):
        feat_out_list, base_out_list = self.backbone_irte(input_list, self.base_model_list[0], self.new_fc_list[0], **kwargs)
        return feat_out_list, base_out_list, []


    def get_lite_j_and_r_irte(self, hx, cx, input, online_policy, tau, isTraining=False):
        feat_lite, _ = self.backbone(input, self.lite_backbone, None)  
        r_list = []
        lite_j_list = []
        batch_size = feat_lite.shape[0]
        old_hx = None
        old_r_t = None
        remain_skip_vector = torch.zeros(batch_size, 1)

        # XXX For single frame, i.e., t = 1
        feat_lite = feat_lite.squeeze(1)
        hx, cx = self.rnn(feat_lite, (hx, cx))
        feat_t = hx

        # decision making
        p_t = torch.log(F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8))
        j_t = self.lite_fc(feat_t) # 84x84 logits
        lite_j_list.append(j_t)
        if online_policy:
            r_t = torch.cat(
                [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])],
                dim=0
            )
            # if old_hx is not None:
            #     take_bool = remain_skip_vector > 0.5
            #     take_old = torch.tensor(take_bool, dtype=torch.float).to(self.device)
            #     take_curr = torch.tensor(~take_bool, dtype=torch.float).to(self.device)
            #     hx = old_hx * take_old + hx * take_curr
            #     r_t = old_r_t * take_old + r_t * take_curr
            
            # TODO here we skip the skip frame aciton
            old_hx = hx
            old_r_t = r_t
            r_list.append(r_t)

        if online_policy:
            return lite_j_list, torch.stack(r_list, dim=1), hx, cx
        else:
            return lite_j_list, None, hx, cx

    def combine_logits(self, r, base_out_list, ind_list):
        # TODO r                N, T, K  [B, T, decision]
        # TODO base_out_list  < K * (N, T, C)
        pred_tensor = torch.stack(base_out_list, dim=2) # [B, T, decision, num_classes]
        r_tensor = r[:, :, :self.reso_dim].unsqueeze(-1) # [B, T, decision, 1]
        t_tensor = torch.sum(r[:, :, :self.reso_dim], dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame [B, 1]
        return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor # [B, num_class]

    def gumbel_softmax(self, training, x, tau = 1.0, hard=False):
        if training:
            eps = 1e-20
            U = torch.rand(x.size()).cuda()
            U = -torch.log(-torch.log(U + eps) + eps)
            r_t = x + 0.5*U
            r_t = F.softmax(r_t / tau, dim=-1)

            if not hard:
                return r_t
            else:
                shape = r_t.size()
                _, ind = r_t.max(dim=-1)
                r_hard = torch.zeros_like(r_t).view(-1, shape[-1])
                r_hard.scatter_(1, ind.view(-1, 1), 1)
                r_hard = r_hard.view(*shape)
                return (r_hard - r_t).detach() + r_t
        else:
            selected = torch.zeros_like(x)
            Q_t = torch.argmax(x, dim=1).unsqueeze(1)
            selected = selected.scatter(1, Q_t, 1)
            return selected.float()
            
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
