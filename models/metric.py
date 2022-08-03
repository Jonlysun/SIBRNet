import torch
import torch.nn as nn
import math
import numpy as np
import skimage.measure


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def add(self, es, ta, ma=None):
        pass

    def get(self):
        return {}

    def items(self):
        return self.get().items()

    def __str__(self):
        return ", ".join(
            ["%s=%.5f" % (key, value) for key, value in self.get().items()]
        )

class MultipleMetric(Metric):
    def __init__(self, metrics, prefix="", **kwargs):
        self.metrics = metrics
        super(MultipleMetric, self).__init__(**kwargs)
        self.prefix = prefix

    def reset(self):
        for m in self.metrics:
            m.reset()

    def add(self, es, ta, ma=None):
        for m in self.metrics:
            m.add(es, ta, ma)

    def get(self):
        ret = {}
        for m in self.metrics:
            vals = m.get()
            for k in vals:
                ret["%s%s" % (self.prefix, k)] = vals[k]
        return ret

    def __str__(self):
        lines = []
        for m in self.metrics:
            line = ", ".join(
                [
                    "%s%s=%.5f" % (self.prefix, key, value)
                    for key, value in m.get().items()
                ]
            )
            lines.append(line)
        return "\n".join(lines)


class BaseDistanceMetric(Metric):
    def __init__(self, name="", stats=None, **kwargs):
        super(BaseDistanceMetric, self).__init__(**kwargs)
        self.name = name
        if stats is None:
            self.stats = {"mean": np.mean}
        else:
            self.stats = stats

    def reset(self):
        self.dists = []

    def add(self, es, ta, ma=None):
        pass

    def get(self):
        dists = np.hstack(self.dists)
        return {"%s_%s" % (self.name, k): f(dists) for k, f in self.stats.items()}

class BaseDepthMetric(Metric):
    def __init__(self, name="", stats=None, **kwargs):
        super(BaseDepthMetric, self).__init__(**kwargs)
        self.name = name
        self.t_valid = 0.0001
        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
        ]
        if stats is None:
            self.stats = {"mean": np.mean}
        else:
            self.stats = stats

    def reset(self):
        self.dists = []

    def add(self, es, ta, ma=None):
        pass

    def get(self):
        dists = np.hstack(self.dists)
        return {"%s_%s" % (self.name, k): f(dists) for k, f in self.stats.items()}

class RMSE(BaseDepthMetric):
    def __init__(self, **kwargs):
        super().__init__(name='RMSE', **kwargs)

    def add(self, pred, gt, ma=None):
        # For numerical stability
        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        # RMSE / MAE
        diff = pred - gt
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)
        self.dists.append(rmse)

class MAE(BaseDepthMetric):
    def __init__(self, **kwargs):
        super().__init__(name='MAE', **kwargs)

    def add(self, pred, gt, ma=None):
        # For numerical stability
        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        # RMSE / MAE
        diff = pred - gt
        diff_abs = torch.abs(diff)

        mae = diff_abs.sum() / (num_valid + 1e-8)
        self.dists.append(mae)

class iRMSE(BaseDepthMetric):
    def __init__(self, **kwargs):
        super().__init__(name='iRMSE', **kwargs)

    def add(self, pred, gt, ma=None):
    
        pred_inv = 1.0 / (pred + 1e-8)
        gt_inv = 1.0 / (gt + 1e-8)

        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        pred_inv = pred_inv[mask]
        gt_inv = gt_inv[mask]

        pred_inv[pred <= self.t_valid] = 0.0
        gt_inv[gt <= self.t_valid] = 0.0

        diff_inv = pred_inv - gt_inv
        diff_inv_sqr = torch.pow(diff_inv, 2)

        irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
        irmse = torch.sqrt(irmse)
        self.dists.append(irmse)


class iMAE(BaseDepthMetric):
    def __init__(self, **kwargs):
        super().__init__(name='iMAE', **kwargs)

    def add(self, pred, gt, ma=None):

        pred_inv = 1.0 / (pred + 1e-8)
        gt_inv = 1.0 / (gt + 1e-8)

        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        pred_inv = pred_inv[mask]
        gt_inv = gt_inv[mask]

        pred_inv[pred <= self.t_valid] = 0.0
        gt_inv[gt <= self.t_valid] = 0.0

        diff_inv = pred_inv - gt_inv
        diff_inv_abs = torch.abs(diff_inv)

        imae = diff_inv_abs.sum() / (num_valid + 1e-8)
        self.dists.append(imae)

class REL(BaseDepthMetric):
    def __init__(self, **kwargs):
        super().__init__(name='REL', **kwargs)

    def add(self, pred, gt, ma=None):
        # For numerical stability
        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        # RMSE / MAE
        diff = pred - gt
        diff_abs = torch.abs(diff)

        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)
        self.dists.append(rel)

class DistanceMetric(BaseDistanceMetric):
    def __init__(self, vec_length=1, p=2, **kwargs):
        super(DistanceMetric, self).__init__(name=str(p), **kwargs)
        self.vec_length = vec_length
        self.p = p

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape or es.shape[-1] != self.vec_length:
            print(es.shape, ta.shape)
            raise Exception(
                "es and ta have to be of shape N x vec_length(={self.vec_length})"
            )
        es = es.reshape(-1, self.vec_length)
        ta = ta.reshape(-1, self.vec_length)
        if ma is not None:
            ma = ma.ravel()
            es = es[ma != 0]
            ta = ta[ma != 0]
        dist = np.linalg.norm(es - ta, ord=self.p, axis=1)
        self.dists.append(dist)

class PSNRMetric(BaseDistanceMetric):
    def __init__(self, max=1, **kwargs):
        super(PSNRMetric, self).__init__(name="psnr", **kwargs)
        # distance between minimum and maximum possible value
        self.max = max

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape:
            raise Exception("es and ta have to be of shape Nxdim")
        if es.ndim == 3:
            es = es[..., None]
            ta = ta[..., None]
        if es.ndim != 4 or es.shape[3] not in [1, 3]:
            raise Exception(
                "es and ta have to be of shape bs x height x width x 0, 1, or 3"
            )
        if ma is not None:
            es = ma * es
            ta = ma * ta
        
        mse = np.mean((es - ta) ** 2)
        if mse == 0:
            psnr = 100
        PIXEL_MAX = 1.0
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        self.dists.append(psnr)


class SSIMMetric(BaseDistanceMetric):
    def __init__(self, data_range=None, mode="default", **kwargs):
        super(SSIMMetric, self).__init__(name="ssim", **kwargs)
        # distance between minimum and maximum possible value
        self.data_range = data_range
        self.mode = mode

    def add(self, es, ta, ma=None):
        if es.shape != ta.shape:
            raise Exception("es and ta have to be of shape Nxdim")
        if es.ndim == 3:
            es = es[..., None]
            ta = ta[..., None]
        if es.ndim != 4 or es.shape[3] not in [1, 3]:
            raise Exception(
                "es and ta have to be of shape bs x height x width x 0, 1, or 3"
            )
        if ma is not None:
            es = ma * es
            ta = ma * ta
        for bidx in range(es.shape[0]):
            if self.mode == "default":
                ssim = skimage.measure.compare_ssim(
                    es[bidx], ta[bidx], multichannel=True, data_range=self.data_range
                )
            elif self.mode == "dv":
                ssim = 0
                for c in range(3):
                    ssim += skimage.measure.compare_ssim(
                        es[bidx, ..., c],
                        ta[bidx, ..., c],
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                        data_range=1.0,
                    )
                ssim /= 3
            else:
                raise Exception("invalid mode")
            self.dists.append(ssim)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = None
        self.clip = True
        
    def forward(self, es, ta):
        if self.mod is None:
            import lpips

            self.mod = lpips.LPIPS(net='alex')
        
        if self.clip:
            es = torch.clamp(es, 0, 1)
        out = self.mod(es, ta, normalize=False)
    
        return out.mean()