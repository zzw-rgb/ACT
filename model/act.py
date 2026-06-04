import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .builder import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        # ACT 主体是一个条件 VAE：训练时用动作序列编码隐变量 z，推理时从先验路径预测动作块。
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        # KL 权重控制隐变量分布约束的强度，越大越鼓励 z 接近标准正态先验。
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """训练时返回 loss 字典；推理时直接返回长度为 num_queries 的动作块。"""
        env_state = None
        # backbone 使用 ImageNet 预训练权重时，需要沿用对应的图像归一化方式。
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            # ACT 一次只监督/预测 num_queries 个未来动作，也就是配置里的 chunk_size。
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # a_hat: 预测动作块；mu/logvar: 由真实动作序列编码出的隐变量分布参数。
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # padding 位置不是有效监督信号，需要从动作重建损失中排除。
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            # ACT 的训练目标：动作重建 L1 + 加权 KL，兼顾模仿精度和潜变量先验约束。
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            # 推理时没有真实动作可编码，DETRVAE 内部会走零向量先验路径生成动作块。
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        # CNNMLP baseline 与 ACT 使用同样的图像归一化，便于公平比较。
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            # baseline 只预测单步动作，因此监督序列中的第 0 个动作。
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    """计算 q(z|a,qpos) 与标准正态先验之间的 KL 散度。"""
    batch_size = mu.size(0)
    assert batch_size != 0
    # 兼容某些模型把 mu/logvar 额外带成 4D 的情况，统一压回 (batch, latent_dim)。
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # 对角高斯 KL：-0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)。
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
