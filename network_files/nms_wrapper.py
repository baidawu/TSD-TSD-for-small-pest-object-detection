# from nms.cpu_nms import cpu_nms
# from cpu_nms import cpu_soft_nms
import cpu_nms
cpu_nms.install()
import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.05, method=1):

    keep = cpu_nms.cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)
