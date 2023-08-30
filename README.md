# USEnhance_CycleGAN

Pytorch implementation of proposed Cycle-GAN based method in USEnhance Challenge.

## Usage
### train:
```bash
cd codes
python train.py --config config.yaml
```

### inference:
```bash
cd codes
python inference.py --config config.yaml
```

## Acknowledgments
Our code borrows heavily from [RegGAN](https://github.com/Kid-Liet/Reg-GAN), [EGEUNet](https://github.com/JCruan519/EGE-UNet). SAM model checkpoint used for perceptual loss can be downloaded from [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).