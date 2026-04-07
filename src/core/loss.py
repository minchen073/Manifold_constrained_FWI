import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..torch_utils import persistence
from torch.func import vmap
from geomloss import SamplesLoss

class PerSampleRMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PerSampleRMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss(logits, labels)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss
    
class relative_MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(relative_MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        relative_errors = torch.norm(predictions - targets, dim = (1,2)) / torch.norm(targets, dim = (1,2))
        return relative_errors.mean()

class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
    def __call__(self, cond_feature, sol_feature, temperature):
        cond_feature = cond_feature.view(cond_feature.size(0), -1)
        sol_feature = sol_feature.view(sol_feature.size(0), -1)
        cond_feature = nn.functional.normalize(cond_feature, dim=-1)
        sol_feature = nn.functional.normalize(sol_feature, dim=-1)
        similarity_matrix = torch.matmul(cond_feature, sol_feature.T)
        similarity_matrix = torch.clamp(similarity_matrix, min=0)
        similarity_matrix = similarity_matrix * temperature.exp()
        batch_size = cond_feature.size(0)
        labels = torch.arange(batch_size).to(cond_feature.device)
        loss = (F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels))/2
        return loss

class DistanceContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        super(DistanceContrastiveLoss, self).__init__()
        self.tau = tau

    def __call__(self, cond_feature, sol_feature, temperature):
        cond_feature = cond_feature.view(cond_feature.size(0), -1)
        sol_feature = sol_feature.view(sol_feature.size(0), -1)
        cond_exp = cond_feature.unsqueeze(1)
        sol_exp = sol_feature.unsqueeze(0)
        distance_matrix = torch.norm(cond_exp - sol_exp, dim=-1)
        similarity_matrix = -distance_matrix * temperature.exp()
        labels = torch.arange(cond_feature.size(0), device=cond_feature.device)
        loss = (F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels))/2
        return loss

class ContrastiveCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ContrastiveCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, similarity_matrix, temperature):
        similarity_matrix = similarity_matrix * temperature.exp()
        labels = torch.zeros(similarity_matrix.size(0), dtype=torch.long, device=similarity_matrix.device)
        return F.cross_entropy(similarity_matrix, labels)
    
class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def __call__(self, input, target):
        if not torch.is_complex(input) or not torch.is_complex(target):
            raise ValueError("Input and target must be complex tensors.")
        real_loss = F.mse_loss(input.real, target.real)
        imag_loss = F.mse_loss(input.imag, target.imag)
        total_loss = real_loss + imag_loss
        return total_loss

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class EDMLoss_CrossBatch:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def __call__(self, net, images1, images2, labels2, loss_scaling=1):
        batch_size = images1.shape[0]
        
        # 生成噪声参数
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=images1.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        y1 = images1
        y2 = images2
        n = torch.randn_like(y1) * sigma

        # 计算y1和y2之间的对应距离（因为batch size相同）
        y1_flat = y1.view(batch_size, -1)  # [batch_size, features]
        y2_flat = y2.view(batch_size, -1)  # [batch_size, features]
        
        # 计算对应元素的距离 [batch_size]
        dist = torch.norm(y1_flat - y2_flat, dim=1)  # [batch_size]
        weight_matrix = torch.exp(-dist**2 / (self.epsilon ** 2))  # [batch_size]
        # weight_matrix = torch.ones_like(dist)  # [batch_size]
        
        # 网络前向传播
        D_yn = net(y1 + n, sigma, labels2)
        reconstruction_loss = torch.mean((D_yn - y1) ** 2, dim=(1,2,3))  # [batch_size]
        # 应用权重
        loss = loss_scaling * weight.squeeze() * weight_matrix * reconstruction_loss  # [batch_size]
        
        return loss.mean()  # 返回标量
    
@persistence.persistent_class
class EDMLoss_CrossBatch_bayesian:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def __call__(self, net, images1, images2, labels2, loss_scaling=1):
        batch_size = images1.shape[0]
        
        # 生成噪声参数
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=images1.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        y1 = images1
        y2 = images2
        n = torch.randn_like(y1) * sigma
        
        # 网络前向传播
        D_yn = net(y1 + n, sigma, labels2)
        reconstruction_loss = torch.mean((D_yn - (y1 - y2)) ** 2, dim=(1,2,3))  # [batch_size]
        # 应用权重
        loss = loss_scaling * weight.squeeze() * reconstruction_loss  # [batch_size]
        
        return loss.mean()  # 返回标量
    

@persistence.persistent_class
class EDMLoss_loosen:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def __call__(self, net, images, labels=None, augment_pipe=None):
        batch_size = images.shape[0]
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        
        if labels is not None:
            # 计算图像之间的距离矩阵（和之前一样）
            y_flat = y.view(batch_size, -1)  # [batch_size, image_features]
            
            # 计算图像之间的欧氏距离
            y_i = y_flat.unsqueeze(1)  # [batch_size, 1, image_features]
            y_j = y_flat.unsqueeze(0)  # [1, batch_size, image_features]
            dist_matrix = torch.sum((y_i - y_j) ** 2, dim=2)  # [batch_size, batch_size]
            
            # 将距离转换为概率 exp(-dist/epsilon^2)
            similarity = torch.exp(-dist_matrix / (self.epsilon ** 2))
            
            # 对每行进行归一化，得到每个样本选择其他图像的概率
            prob_matrix = similarity / similarity.sum(dim=1, keepdim=True)  # [batch_size, batch_size]
            # print(prob_matrix.shape)
            # print(prob_matrix.max(dim=1))
            
            # 调试：验证对角线概率最大
            # diagonal_probs = torch.diag(prob_matrix)  # 每个样本选择自己的概率
            # max_probs = prob_matrix.max(dim=1)[0]     # 每个样本的最大概率
            # print(f"对角线概率: {diagonal_probs[:5]}")
            # print(f"最大概率: {max_probs[:5]}")
            # print(f"对角线是否为最大: {torch.allclose(diagonal_probs, max_probs)}")
            
            # 为每个标签基于概率随机选择一个图像的索引
            selected_image_indices = torch.multinomial(prob_matrix, 1).squeeze()  # [batch_size]
            
            # 调试：验证独立随机选择
            # self_selection = (selected_image_indices == torch.arange(batch_size, device=images.device)).float().mean()
            # print(f"选择自己图像的比例: {self_selection:.3f}")
            
            # 只选择图像，噪声和sigma保持独立
            selected_images = y[selected_image_indices]  # [batch_size, C, H, W]
            
            # 使用原始标签、选择的图像、独立的噪声和sigma计算网络输出
            D_yn = net(selected_images + n, sigma, labels, augment_labels=augment_labels)
            
            # 计算损失：网络应该预测选择的干净图像
            loss = weight * ((D_yn - selected_images) ** 2)
        else:
            # 如果没有提供标签，保持原有逻辑
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
            loss = weight * ((D_yn - y) ** 2)

        return loss
    
@persistence.persistent_class
class EDMLoss_RandomYReplace:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, replace_prob=0.2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.replace_prob = replace_prob

    def __call__(self, net, images, labels=None):
        batch_size = images.shape[0]
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma

        # 随机替换y
        if self.replace_prob > 0:
            device = y.device
            mask = torch.rand(batch_size, device=device) < self.replace_prob
            rand_idx = torch.randint(0, batch_size, (batch_size,), device=device)
            y_replaced = y.clone()
            y_replaced[mask] = y[rand_idx[mask]]
        else:
            y_replaced = y

        D_yn = net(y_replaced + n, sigma, labels)
        loss = weight * ((D_yn - y_replaced) ** 2)
        return loss

class LossFunction(nn.Module):
    def __init__(self, loss_type: str, normalize: bool = False, reduce_mean: bool = True, normalize_eps: float = 0.01):
        super(LossFunction, self).__init__()
        self.normalize = normalize
        self.reduce_mean = reduce_mean
        self.normalize_eps = normalize_eps
        if self.normalize_eps <= 0:
            raise ValueError(f"'normalize_eps' should be a positive float, but got '{normalize_eps}'.")
        if loss_type == "MSE":
            self.sample_loss_fn = nn.MSELoss(reduction='mean' if reduce_mean else 'none')
        elif loss_type == "RMSE":
            self.sample_loss_fn = relative_MSELoss(reduction='mean' if reduce_mean else 'none')
        elif loss_type == "MAE":
            self.sample_loss_fn = nn.L1Loss(reduction='mean' if reduce_mean else 'none')
        elif loss_type == "INFONCE":
            self.sample_loss_fn = InfoNCELoss(temperature=0.07)
        else:
            raise ValueError(f"'loss_type' should be one of ['MSE', 'RMSE', 'MAE', 'INFONCE'], but got '{loss_type}'.")

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = self.sample_loss_fn(pred, label)
        if self.normalize:
            label_norm = self.sample_loss_fn(label, torch.zeros_like(label))
            loss = loss / (label_norm + self.normalize_eps)
        return loss
    
class Wasserstein_ADFWI(nn.Module):
    def __init__(self, method='linear'):
        super(Wasserstein_ADFWI, self).__init__()
        self.method = method
    
    def searchsorted(self, a, v, side='left'):
        right = (side != 'left')
        return torch.searchsorted(a, v, right=right)
    
    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices)

    def quantile_function(self, qs, cws, xs):
        n = xs.shape[0]
        cws = cws.contiguous()
        qs = qs.contiguous()
        idx = self.searchsorted(cws, qs)
        return self.take_along_axis(xs, torch.clip(idx, 0, n - 1), axis=0)

    def quantile_function_with_interpolation(self, qs, cws, xs):
        idx = torch.searchsorted(cws, qs)
        
        result = torch.zeros_like(qs)
        
        mask_left = qs <= cws[0]
        result[mask_left] = xs[0]

        mask_right = qs >= cws[-1]
        result[mask_right] = xs[-1]

        mask_mid = ~(mask_left | mask_right)
        q_mid = qs[mask_mid]
        idx_mid = idx[mask_mid]
        idx_left = idx_mid - 1
        cw_left = torch.gather(cws, 0, idx_left)
        cw_right = torch.gather(cws, 0, idx_mid)
        x_left = torch.gather(xs, 0, idx_left)
        x_right = torch.gather(xs, 0, idx_mid)
        alpha = (q_mid - cw_left) / (cw_right - cw_left)
        result[mask_mid] = x_left + alpha * (x_right - x_left)
        
        return result

    def zero_pad(self, a, pad_width, value=0):
        from torch.nn.functional import pad
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad, value=value)

    def wasserstein_1d(self, t, u_weights=None, v_weights=None, p=1):
        assert p >= 1, f"OT损失仅对p>=1有效，给定了{p}"

        u_cumweights = torch.cumsum(u_weights, 0)
        v_cumweights = torch.cumsum(v_weights, 0)

        qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
        
        u_quantiles = self.quantile_function(qs, u_cumweights, t)
        v_quantiles = self.quantile_function(qs, v_cumweights, t)
        # u_quantiles = self.quantile_function_with_interpolation(qs, u_cumweights, t)
        # v_quantiles = self.quantile_function_with_interpolation(qs, v_cumweights, t)
        
        qs = self.zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
        delta = qs[1:, ...] - qs[:-1, ...]
        diff_quantiles = torch.abs(u_quantiles - v_quantiles)

        if p == 1:
            return torch.sum(delta * diff_quantiles, axis=0)
        return torch.sum(delta * torch.pow(diff_quantiles, p), axis=0)

    def transform_nonnegative(self, x, y, method='linear'):
        if method == 'abs':
            return torch.abs(x), torch.abs(y)
        elif method == 'square':
            return x**2, y**2
        elif method == 'sqrt':
            return torch.sqrt(x**2), torch.sqrt(y**2)
        elif method == 'linear':
            min_value = torch.min(torch.min(x), torch.min(y))
            min_value = min_value if min_value < 0 else 0
            return x - 1.1*min_value, y - 1.1*min_value
        elif method == 'softplus':
            beta = 0.2
            return F.softplus(beta*x), F.softplus(beta*y)
        elif method == 'exp':
            beta = 1.
            return torch.exp(beta*x), torch.exp(beta*y)
        else:
            raise ValueError(f'无效的方法: {method}')
            
    def normalize(self, x, dim=2, ntype='sumis1'):
        if ntype == 'sumis1':
            return x / (torch.sum(x, dim=dim, keepdim=True) + 1e-10)
        elif ntype == 'max1':
            return x / (torch.max(torch.abs(x), dim=dim, keepdim=True)[0] + 1e-10)
        else:
            raise ValueError(f'无效的归一化类型: {ntype}')

    def forward(self, x, y):
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
            
        if len(x.shape) == 4:
            batch_size, n_sources, n_time, n_space = x.shape
        else:
            batch_size = 1
            n_sources, n_time, n_space = x.shape
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x, y = self.transform_nonnegative(x, y, method=self.method)
        x = self.normalize(x, dim=2, ntype='sumis1')
        y = self.normalize(y, dim=2, ntype='sumis1')

        loss = torch.zeros(batch_size, device=x.device)
        dt = 0.001
        t = torch.linspace(0, n_time-1, n_time, dtype=x.dtype, device=x.device) * dt

        for b in range(batch_size):
            for s in range(n_sources):
                _x = x[b, s]
                _y = y[b, s]
                _x = _x.permute(1, 0).contiguous()
                _y = _y.permute(1, 0).contiguous()

                for i in range(n_space):
                    if torch.sum(_x[i]) == 0 or torch.sum(_y[i]) == 0:
                        continue
                    else:
                        loss[b] += self.wasserstein_1d(t, _x[i], _y[i], p=2)
                    
        return loss
    
class Wasserstein_POT(nn.Module):
    def __init__(self, method='linear'):
        super(Wasserstein_POT, self).__init__()
        self.method = method
    
    def searchsorted(self, a, v):
        return torch.searchsorted(a, v)
    
    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices.clamp(0, arr.shape[axis]-1))
    
    def zero_pad(self, a, pad_width, value=0):
        if pad_width[0][0] == 1 and pad_width[0][1] == 0:
            shape = list(a.shape)
            shape[0] = 1
            zeros = torch.full(shape, value, dtype=a.dtype, device=a.device)
            return torch.cat([zeros, a], dim=0)
        return a
    
    def quantile_function(self, qs, cws, xs):
        """计算分位数函数"""
        idx = self.searchsorted(cws, qs)
        return self.take_along_axis(xs, idx, axis=0)
    
    def transform_nonnegative(self, x, y, method='linear'):
        if method == 'abs':
            return torch.abs(x), torch.abs(y)
        elif method == 'square':
            return x**2, y**2
        elif method == 'sqrt':
            return torch.sqrt(x**2), torch.sqrt(y**2)
        elif method == 'linear':
            min_value = torch.min(torch.min(x), torch.min(y))
            min_value = min_value if min_value < 0 else 0
            return x - 1.1*min_value, y - 1.1*min_value
        elif method == 'softplus':
            beta = 0.2
            return F.softplus(beta*x), F.softplus(beta*y)
        elif method == 'exp':
            beta = 1.
            return torch.exp(beta*x), torch.exp(beta*y)
        else:
            raise ValueError(f'无效的方法: {method}')
            
    def normalize(self, x, dim=2, ntype='sumis1'):
        if ntype == 'sumis1':
            return x / (torch.sum(x, dim=dim, keepdim=True))
        elif ntype == 'max1':
            return x / (torch.max(torch.abs(x), dim=dim, keepdim=True)[0])
        else:
            raise ValueError(f'无效的归一化类型: {ntype}')
    
    # def wasserstein_1d_torch(self, u_values, v_values, u_weights=None, v_weights=None, p=2):
    #     n = u_values.shape[0]
    #     m = v_values.shape[0]
        
    #     if u_weights is None:
    #         u_weights = torch.full_like(u_values, 1.0 / n)
    #     if v_weights is None:
    #         v_weights = torch.full_like(v_values, 1.0 / m)
        
    #     u_cumweights = torch.cumsum(u_weights, 0)
    #     v_cumweights = torch.cumsum(v_weights, 0)
        
    #     qs = torch.sort(torch.cat([u_cumweights, v_cumweights], 0))[0]
    #     u_quantiles = self.quantile_function(qs, u_cumweights, u_values)
    #     v_quantiles = self.quantile_function(qs, v_cumweights, v_values)
        
    #     qs_padded = torch.cat([torch.zeros(1, device=qs.device, dtype=qs.dtype), qs])
    #     delta = qs_padded[1:] - qs_padded[:-1]
    #     print(delta)
    #     diff_quantiles = torch.abs(u_quantiles - v_quantiles)
        
    #     if p == 1:
    #         return torch.sum(delta * diff_quantiles)
    #     return torch.sum(delta * torch.pow(diff_quantiles, p))

    def wasserstein_1d_torch(self, u_density, v_density, p=2):
        u_density = u_density / (torch.sum(u_density) + 1e-10)
        v_density = v_density / (torch.sum(v_density) + 1e-10)
        
        u_cdf = torch.cumsum(u_density, 0)
        v_cdf = torch.cumsum(v_density, 0)
        
        qs = torch.sort(torch.cat([u_cdf, v_cdf], 0))[0]

        n = u_density.shape[0]
        m = v_density.shape[0]
        u_positions = torch.arange(n, dtype=u_density.dtype, device=u_density.device) / n
        v_positions = torch.arange(m, dtype=v_density.dtype, device=v_density.device) / m
        
        u_quantiles = self.quantile_function(qs, u_cdf, u_positions)
        v_quantiles = self.quantile_function(qs, v_cdf, v_positions)
        
        qs_padded = torch.cat([torch.zeros(1, device=qs.device, dtype=qs.dtype), qs])
        delta = qs_padded[1:] - qs_padded[:-1]
        diff_quantiles = torch.abs(u_quantiles - v_quantiles)
        
        if p == 1:
            return torch.sum(delta * diff_quantiles)
        else:
            return torch.sum(delta * torch.pow(diff_quantiles, p))
    
    def forward(self, x, y):
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
            
        if len(x.shape) == 4:
            batch_size, n_sources, n_time, n_space = x.shape
        else:
            batch_size = 1
            n_sources, n_time, n_space = x.shape
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x, y = self.transform_nonnegative(x, y, method=self.method)
        x = self.normalize(x, dim=2, ntype='sumis1')
        y = self.normalize(y, dim=2, ntype='sumis1')
        loss = torch.zeros(batch_size, device=x.device)
        
        for b in range(batch_size):
            channel_distances = 0
            
            for s in range(n_sources):
                _x = x[b, s].permute(1, 0).contiguous()  # [n_space, n_time]
                _y = y[b, s].permute(1, 0).contiguous()  # [n_space, n_time]
                
                for i in range(n_space):
                    if torch.sum(_x[i]) == 0 or torch.sum(_y[i]) == 0:
                        continue
                    print(_x[i])
                    w2_squared = self.wasserstein_1d_torch(_x[i], _y[i], p=2)
                    channel_distances += w2_squared
            
            loss[b] = channel_distances
                    
        return loss

class Wasserstein_Sinkhorn(torch.nn.Module):
    def __init__(self, dt=0.001, p=2, blur=0.01, scaling=0.5, sparse_sampling=1):
        super().__init__()
        self.dt = dt
        self.sparse_sampling = sparse_sampling
        self.p = p
        self.blur = blur
        self.scaling = scaling
        
        # 导入geomloss库
        from geomloss import SamplesLoss
    
    def forward(self, syn_data, obs_data):
        device = syn_data.device
        
        # 自动检测并调整维度
        if syn_data.dim() == 3:
            # 如果是3维 [n_shots, nt, nr]，添加batch维度
            syn_data = syn_data.unsqueeze(0)
            obs_data = obs_data.unsqueeze(0)
        
        batch_size, n_shots, nt, nr = syn_data.shape
        
        misfit_fun = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, scaling=self.scaling)
        
        loss = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            batch_loss = torch.tensor(0.0, device=device)
            
            for s in range(n_shots):
                mask1 = torch.sum(torch.abs(obs_data[b, s]), dim=0) == 0
                mask2 = torch.sum(torch.abs(syn_data[b, s]), dim=0) == 0
                mask = ~(mask1 & mask2)
                trace_idx = torch.nonzero(mask).squeeze(-1)
                
                if len(trace_idx) == 0:
                    continue
                
                obs_shot = obs_data[b, s, ::self.sparse_sampling, trace_idx].T  # [trace, time]
                syn_shot = syn_data[b, s, ::self.sparse_sampling, trace_idx].T  # [trace, time]
                
                t = torch.arange(obs_shot.shape[1], device=device) * self.dt
                t = t.unsqueeze(0).expand(obs_shot.shape[0], -1)
                
                obs_with_time = torch.stack((t, obs_shot), dim=-1)  # [trace, time, 2]
                syn_with_time = torch.stack((t, syn_shot), dim=-1)  # [trace, time, 2]
                
                w2_dist = misfit_fun(obs_with_time, syn_with_time)
                
                if w2_dist.dim() > 0:
                    w2_dist = w2_dist.sum()
                
                batch_loss = batch_loss + w2_dist
            
            loss[b] = batch_loss
        
        return loss

def w2_distance_from_discretized_pdf(pdf_values1: torch.Tensor,
                                     pdf_values2: torch.Tensor,
                                     x_coords: torch.Tensor,
                                     normalize_type: str = 'linear',
                                     b: float = 30.0) -> torch.Tensor:
    """
    计算一个或多个离散 PDF 对之间的 W2 距离。
    这个函数是完全可微分的，并且支持对多维输入进行批处理。

    Args:
        pdf_values1 (torch.Tensor): 第一个分布的 PDF 值。
                                    支持的形状: (N,), (B, N), (B, R, N)
        pdf_values2 (torch.Tensor): 第二个分布的 PDF 值。形状必须与 pdf_values1 相同。
        x_coords (torch.Tensor): PDF 值对应的 x 坐标。形状: (N,)
        normalize_type (str): 归一化类型，可选 'linear', 'exponential', 'softplus'。默认为 'linear'。
        b (float): 归一化参数。
                   - 对于 'linear': 偏移量，默认为 30.0
                   - 对于 'exponential': 指数缩放因子，默认为 30.0
                   - 对于 'softplus': softplus 缩放因子，默认为 30.0

    Returns:
        torch.Tensor: W2 距离。
                      如果输入是 (N,), 返回标量。
                      如果输入是 (B, N), 返回 (B,)。
                      如果输入是 (B, R, N), 返回 (B, R)。
    """
    # 记录原始形状以便恢复
    original_shape = pdf_values1.shape
    
    if original_shape != pdf_values2.shape:
        raise ValueError(f"Input tensor shapes must be identical. pdf1: {original_shape}, pdf2: {pdf_values2.shape}")
    
    if original_shape[-1] != x_coords.shape[0]:
        raise ValueError(f"The last dimension of input tensors ({original_shape[-1]}) must match the size of x_coords ({x_coords.shape[0]}).")

    # 将输入重塑为 (B_eff, N) 以进行批处理
    # B_eff 是所有批次维度的乘积
    num_points = original_shape[-1]
    reshaped_pdf1 = pdf_values1.reshape(-1, num_points)
    reshaped_pdf2 = pdf_values2.reshape(-1, num_points)
    
    device = reshaped_pdf1.device
    effective_batch_size = reshaped_pdf1.shape[0]
    
    # --- 1. 将PDF值置为正并进行归一化 ---
    # 根据归一化类型选择不同的转换方法
    if normalize_type == 'linear':
        # 线性归一化: f_tilde = (f + b) / (f + b)的积分
        # 这里简化为先加上偏移量b，使所有值非负
        non_negative_pdf1 = reshaped_pdf1 + b
        non_negative_pdf2 = reshaped_pdf2 + b
    
    elif normalize_type == 'exponential':
        # 指数归一化: f_tilde = exp(b*f) / (exp(b*f)的积分)
        non_negative_pdf1 = torch.exp(b * reshaped_pdf1)
        non_negative_pdf2 = torch.exp(b * reshaped_pdf2)
    
    elif normalize_type == 'softplus':
        # Softplus归一化: f_tilde = log(exp(b*f) + 1) / (log(exp(b*f) + 1)的积分)
        non_negative_pdf1 = torch.log(torch.exp(b * reshaped_pdf1) + 1)
        non_negative_pdf2 = torch.log(torch.exp(b * reshaped_pdf2) + 1)
    
    else:
        raise ValueError(f"无效的归一化类型: {normalize_type}。可选值为 'linear', 'exponential', 'softplus'。")

    # 辅助函数，使用梯形法则计算积分（总面积）
    def _integrate(pdf, x):
        dx = torch.diff(x)
        # pdf shape: (B_eff, N), dx shape: (N-1)
        areas = (pdf[..., :-1] + pdf[..., 1:]) / 2.0 * dx
        return torch.sum(areas, dim=-1)

    total_area1 = _integrate(non_negative_pdf1, x_coords)
    total_area2 = _integrate(non_negative_pdf2, x_coords)

    # 归一化，并添加 epsilon 以避免除以零
    # total_area shapes are (B_eff,), need to be (B_eff, 1) for broadcasting
    reshaped_pdf1 = non_negative_pdf1 / (total_area1.unsqueeze(-1) + 1e-9)
    reshaped_pdf2 = non_negative_pdf2 / (total_area2.unsqueeze(-1) + 1e-9)
    
    # 辅助函数: 将离散 PDF 转换为离散 CDF
    def _pdf_to_cdf(pdf_values, x):
        dx = torch.diff(x)
        areas = (pdf_values[..., :-1] + pdf_values[..., 1:]) / 2.0 * dx
        zeros = torch.zeros((effective_batch_size, 1), device=device, dtype=pdf_values.dtype)
        cdf_values = torch.cat([zeros, torch.cumsum(areas, dim=-1)], dim=-1)
        # 确保 CDF 值是连续的，以提高 searchsorted 的性能
        return cdf_values.contiguous()

    # 辅助函数: 将离散 CDF 转换为离散分位数函数
    def _cdf_to_quantile(cdf_values, x, t):
        # cdf_values: (B_eff, N), t: (N,)
        # 为了让 searchsorted 正确处理批次，需要将 t 扩展为 (B_eff, N)
        # 使用 contiguous() 确保张量是连续的，避免性能警告
        t_expanded = t.expand(cdf_values.shape[0], -1).contiguous()
        right_indices = torch.searchsorted(cdf_values, t_expanded)
        
        right_indices = torch.clamp(right_indices, 1, num_points - 1)
        left_indices = right_indices - 1

        cdf_left = torch.gather(cdf_values, 1, left_indices)
        cdf_right = torch.gather(cdf_values, 1, right_indices)
        
        x_expanded = x.expand(effective_batch_size, -1)
        x_left = torch.gather(x_expanded, 1, left_indices)
        x_right = torch.gather(x_expanded, 1, right_indices)

        dcdf = cdf_right - cdf_left
        # Add epsilon to avoid division by zero
        dcdf = torch.where(dcdf < 1e-8, torch.tensor(1.0, device=device, dtype=dcdf.dtype), dcdf)
        slope = (x_right - x_left) / dcdf
        # 使用扩展后的 t_expanded 进行插值
        quantile_values = x_left + (t_expanded - cdf_left) * slope
        
        return quantile_values

    # --- 主要计算流程 ---
    
    # 1. 计算两个分布的 CDF
    cdf1 = _pdf_to_cdf(reshaped_pdf1, x_coords)
    cdf2 = _pdf_to_cdf(reshaped_pdf2, x_coords)

    # 2. 计算两个分布的分位数函数
    t = torch.linspace(0, 1, num_points, device=device, dtype=reshaped_pdf1.dtype)
    
    quantile1 = _cdf_to_quantile(cdf1, x_coords, t)
    quantile2 = _cdf_to_quantile(cdf2, x_coords, t)

    # 3. 计算 W2 距离
    integrand = (quantile1 - quantile2) ** 2
    
    # Integrate along the last dimension (the N points)
    w2_squared = torch.trapezoid(integrand, t, dim=-1)
    
    result_flat = torch.sqrt(w2_squared) # Shape: (B_eff,)
    
    # 恢复到原始批次形状
    if len(original_shape) == 1: # Input was (N,)
        return result_flat.squeeze(0) # Return scalar
    else:
        output_shape = original_shape[:-1] # (B,) or (B, R)
        return result_flat.view(output_shape)


def w2_distance_velocity_field(
    pred: torch.Tensor,
    target: torch.Tensor,
    trace_type: str = 'both',
    x_coords: torch.Tensor = None,
    normalize_type: str = 'linear',
    b: float = 30.0,
    reduction: str = 'sum',
) -> torch.Tensor:
    """
    计算两个速度场之间的 W2 距离（逐道/逐行/逐列）。
    将 70x70 速度场视为多道 1D 分布，每道与 w2_distance_from_discretized_pdf 一致。

    Args:
        pred (torch.Tensor): 预测速度场，形状 (H, W) 或 (B, H, W)，如 (70, 70)
        target (torch.Tensor): 目标速度场，形状与 pred 相同
        trace_type (str): 道类型，'row'=每行一道, 'column'=每列一道, 'both'=行+列
        x_coords (torch.Tensor): 空间坐标，形状 (W,) 或 (H,)。None 时使用 arange
        normalize_type (str): 归一化类型，'linear', 'exponential', 'softplus'
        b (float): 归一化参数
        reduction (str): 'sum' 或 'mean'

    Returns:
        torch.Tensor: 标量 W2 距离
    """
    device = pred.device
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    batch_size, H, W = pred.shape
    if target.shape != pred.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

    w2_list = []

    if trace_type in ('row', 'both'):
        # 每行一道: (B, H, W) -> (B*H, W)
        pred_rows = pred.reshape(batch_size * H, W)
        target_rows = target.reshape(batch_size * H, W)
        x_row = x_coords if x_coords is not None and x_coords.numel() == W else torch.arange(W, device=device, dtype=torch.float32)
        w2_rows = w2_distance_from_discretized_pdf(
            pred_rows, target_rows, x_row, normalize_type=normalize_type, b=b
        )
        w2_list.append(w2_rows)

    if trace_type in ('column', 'both'):
        # 每列一道: (B, H, W) -> (B*W, H)
        pred_cols = pred.permute(0, 2, 1).reshape(batch_size * W, H)
        target_cols = target.permute(0, 2, 1).reshape(batch_size * W, H)
        x_col = x_coords if x_coords is not None and x_coords.numel() == H else torch.arange(H, device=device, dtype=torch.float32)
        w2_cols = w2_distance_from_discretized_pdf(
            pred_cols, target_cols, x_col, normalize_type=normalize_type, b=b
        )
        w2_list.append(w2_cols)

    w2_all = torch.cat([w.view(-1) for w in w2_list])
    if reduction == 'sum':
        return w2_all.sum()
    return w2_all.mean()


class WavefieldLoss(torch.nn.Module):
    """
    统一的波场损失函数接口，支持多种损失函数类型。
    
    使用方法:
        # 初始化
        loss_func = WavefieldLoss(loss_type='w2_sinkhorn', dt=0.001, p=2, blur=0.01, scaling=0.5)
        
        # 计算损失
        loss = loss_func(syn_wavefield, obs_wavefield)
    
    支持的损失函数类型:
        - 'l1': 平均绝对误差 mean(|syn-obs|)（与逐元素 L1 一致）
        - 'mse': 均方误差 mean((syn-obs)^2)，即逐元素平方后 **平均**（与 ``nn.MSELoss`` 默认一致）
        - 'l2_sq': 残差 **平方 L2 范数** sum((syn-obs)^2)，不对元素数归一；对应论文中
          :math:`\\|u_{data}-f_{PDE}(x)\\|_2^2` 的离散形式（无 1/N 因子）
        - 'w2_sinkhorn': 基于Sinkhorn算法的Wasserstein距离
        - 'w2_per_trace': 逐道计算的精确W2距离
    """
    def __init__(self, loss_type='mse', dt=0.001, p=2, blur=0.01, scaling=0.5, sparse_sampling=1, 
                 normalize_type='linear', b=30.0):
        """
        Args:
            loss_type (str): 损失函数类型，可选 'l1', 'mse', 'l2_sq', 'w2_sinkhorn', 'w2_per_trace'
            dt (float): 时间采样间隔（用于W2损失）
            p (int): W2距离的阶数（用于W2 Sinkhorn）
            blur (float): Sinkhorn算法的模糊参数（用于W2 Sinkhorn）
            scaling (float): Sinkhorn算法的缩放参数（用于W2 Sinkhorn）
            sparse_sampling (int): 稀疏采样率（用于W2 Sinkhorn）
            normalize_type (str): 归一化类型，可选 'linear', 'exponential', 'softplus'（用于W2 per trace）
            b (float): 归一化参数（用于W2 per trace）
        """
        super().__init__()
        self.loss_type = loss_type
        self.dt = dt
        self.normalize_type = normalize_type
        self.b = b
        
        if loss_type == 'l1':
            self.loss_fn = None
        elif loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == 'l2_sq':
            self.loss_fn = None
        elif loss_type == 'w2_sinkhorn':
            self.loss_fn = Wasserstein_Sinkhorn(
                dt=dt,
                p=p,
                blur=blur,
                scaling=scaling,
                sparse_sampling=sparse_sampling
            )
        elif loss_type == 'w2_per_trace':
            self.loss_fn = None  # 使用函数计算
        else:
            raise ValueError(
                f"不支持的损失函数类型: {loss_type}. 请选择 "
                "'l1', 'mse', 'l2_sq', 'w2_sinkhorn' 或 'w2_per_trace'"
            )
    
    def forward(self, syn_wavefield, obs_wavefield):
        """
        计算波场损失。
        
        Args:
            syn_wavefield (torch.Tensor): 合成波场，形状为 [n_shots, nt, nr] 或 [batch, n_shots, nt, nr]
            obs_wavefield (torch.Tensor): 观测波场，形状与 syn_wavefield 相同
        
        Returns:
            torch.Tensor: 损失值（标量）
        """
        if self.loss_type == 'l1':
            return self._compute_l1_loss(syn_wavefield, obs_wavefield)
        if self.loss_type == 'mse':
            return self._compute_mse_loss(syn_wavefield, obs_wavefield)
        if self.loss_type == 'l2_sq':
            return self._compute_l2_sq_loss(syn_wavefield, obs_wavefield)
        if self.loss_type == 'w2_sinkhorn':
            return self._compute_w2_sinkhorn_loss(syn_wavefield, obs_wavefield)
        if self.loss_type == 'w2_per_trace':
            return self._compute_w2_per_trace_loss(syn_wavefield, obs_wavefield)
        raise RuntimeError(f"WavefieldLoss: unhandled loss_type {self.loss_type!r}")

    def _compute_l1_loss(self, syn_wavefield, obs_wavefield):
        """逐元素绝对误差，默认 reduction=mean（全体元素平均）。"""
        return F.l1_loss(syn_wavefield, obs_wavefield)

    def _compute_l2_sq_loss(self, syn_wavefield, obs_wavefield):
        """残差平方和 sum_i (syn-obs)_i^2；与 MSE 差一个 1/N 因子（N 为元素个数）。"""
        diff = syn_wavefield - obs_wavefield
        return (diff * diff).sum()

    def _compute_mse_loss(self, syn_wavefield, obs_wavefield):
        """均方误差：``nn.MSELoss`` 对全体元素取平均。"""
        return self.loss_fn(syn_wavefield, obs_wavefield)
    
    def _compute_w2_sinkhorn_loss(self, syn_wavefield, obs_wavefield):
        """计算W2 Sinkhorn损失"""
        # 确保输入是4D张量 [batch, n_shots, nt, nr]
        if syn_wavefield.dim() == 3:
            syn_wavefield = syn_wavefield.unsqueeze(0)
            obs_wavefield = obs_wavefield.unsqueeze(0)
        
        loss_tensor = self.loss_fn(syn_wavefield, obs_wavefield)
        
        # 确保返回标量
        if loss_tensor.dim() > 0:
            return loss_tensor.mean()
        return loss_tensor
    
    def _compute_w2_per_trace_loss(self, syn_wavefield, obs_wavefield):
        """
        逐道计算W2距离（批量优化版本）
        
        Args:
            syn_wavefield (torch.Tensor): 合成波场，形状为 [n_shots, nt, nr] 或 [batch, n_shots, nt, nr]
            obs_wavefield (torch.Tensor): 观测波场，形状与 syn_wavefield 相同
        
        Returns:
            torch.Tensor: W2距离的总和（标量）
        
        Note:
            - 使用 self.normalize_type 控制归一化类型（'linear', 'exponential', 'softplus'）
            - 使用 self.b 控制归一化参数
            - 相比逐道循环，批量计算大幅提升了效率
        """
        # 确保输入是4D张量 [batch, n_shots, nt, nr]
        device = syn_wavefield.device
        if syn_wavefield.dim() == 3:
            # 如果是3维输入，添加batch维度
            syn_wavefield = syn_wavefield.unsqueeze(0)
            obs_wavefield = obs_wavefield.unsqueeze(0)
        
        batch_size, n_shots, nt, nr = syn_wavefield.shape
        
        # 创建时间坐标
        x_coords = torch.arange(nt, device=device, dtype=torch.float32) * self.dt
        
        # 批量计算：将所有道重塑为 [batch * n_shots * nr, nt]
        # 对于4维输入 [batch, n_shots, nt, nr]，使用permute(0, 1, 3, 2)转换为 [batch, n_shots, nr, nt]
        syn_all_traces = syn_wavefield.permute(0, 1, 3, 2).reshape(batch_size * n_shots * nr, nt)  # [batch*n_shots*nr, nt]
        obs_all_traces = obs_wavefield.permute(0, 1, 3, 2).reshape(batch_size * n_shots * nr, nt)  # [batch*n_shots*nr, nt]
        
        # 批量计算W2距离，返回 [n_shots*nr] 的向量
        w2_distances = w2_distance_from_discretized_pdf(
            syn_all_traces, 
            obs_all_traces, 
            x_coords,
            normalize_type=self.normalize_type,
            b=self.b
        )
        
        # 返回所有W2距离的总和
        return w2_distances.sum()
    
    def __call__(self, syn_wavefield, obs_wavefield):
        """使损失函数可以直接调用"""
        return self.forward(syn_wavefield, obs_wavefield)


def postprocess_model(padded_model):

    if padded_model.dim() == 3:
        padded_model = padded_model.unsqueeze(1)
        
    if padded_model.shape[2] == 72 and padded_model.shape[3] == 72:
        cropped_model = padded_model[:, :, :70, :70]
        return cropped_model
    
    return padded_model
