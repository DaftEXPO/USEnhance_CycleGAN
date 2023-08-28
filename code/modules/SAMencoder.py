from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F


def regist_sam_image_encoder(checkpoint_path, net_type='vit_b'):
    sam_model = sam_model_registry[net_type](checkpoint=checkpoint_path)
    encoder = sam_model.image_encoder
    transform = ResizeLongestSide(encoder.img_size)
    mean, std = sam_model.pixel_mean, sam_model.pixel_std
    return encoder, transform, (mean, std)


def tensor_transform(img_tensor, transform):
    img_tensor = transform.apply_image_torch(img_tensor)
    # mean, std = stats
    # img_tensor = (img_tensor - mean) / std
    h, w = img_tensor.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    img_tensor = F.pad(img_tensor, (0, padw, 0, padh))
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    return img_tensor