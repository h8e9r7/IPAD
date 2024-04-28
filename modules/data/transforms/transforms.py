from torchvision import transforms
import numpy as np
import cv2

from .random_augment import RandomAugment

class GlobalTransform(object):
    def __init__(self, cfg, is_train=False):
        # print('++++++++++++++random augmentation [ON]')
        if cfg.MODEL.SINGLE.ENABLE:
            print("flip+++++")
            
            self.t = transforms.Compose(
                [
                    transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
                    transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
                    transforms.RandomHorizontalFlip(),
                    ####
                    
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=cfg.INPUT.PIXEL_MEAN,
                        std=cfg.INPUT.PIXEL_STD
                    )
                ]
            )
        else:
            self.t = transforms.Compose(
                [
                    transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
                    transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=cfg.INPUT.PIXEL_MEAN,
                        std=cfg.INPUT.PIXEL_STD
                    )
                ]
            )

        self.aug1 = transforms.Compose([
            transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
            # transforms.RandomPerspective(distortion_scale=0.2, p=1),
            transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD
            )
        ])
        self.aug2 = transforms.Compose([
            transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
            # transforms.RandomPerspective(distortion_scale=0.4, p=1),
            transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
            # transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness','ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD
            )
        ])

        self.t1 = transforms.Compose(
            [
                transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
                transforms.RandomPerspective(distortion_scale=0.2, p=1),
                transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD
                )
            ]
        )

        self.t2 = transforms.Compose(
            [
                transforms.Resize(cfg.INPUT.GLOBAL_SIZE),
                transforms.RandomPerspective(distortion_scale=0.4, p=1),
                transforms.CenterCrop(cfg.INPUT.GLOBAL_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD
                )
            ]
        )

    def __call__(self, img, transform_type=0, is_train=False):   
        if transform_type == 0:
            return self.t(img)
        if transform_type == 1:
            return self.t1(img)
        if transform_type == 2:
            return self.t2(img)





class LocalTransform(object):
    def __init__(self, cfg):
        self.t = transforms.Compose(
            [
                transforms.Resize(cfg.INPUT.LOCAL_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD
                )
            ]
        )
        
        self.global_size = cfg.INPUT.GLOBAL_SIZE
        self.threshold = cfg.INPUT.THRESHOLD

    def __call__(self, img, mask):
        mask = cv2.resize(mask, (self.global_size,self.global_size), interpolation=cv2.INTER_LINEAR)
        # min-max normalization
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        # map to char
        mask = np.uint8(mask * 255)
        # binarization
        ret, mask = cv2.threshold(mask, 255 * self.threshold, 255, cv2.THRESH_BINARY)
        # bounding box, got (left, top, width, height)
        x, y, w, h = cv2.boundingRect(mask)
        # box center
        xc = x + w//2
        yc = y + h//2
        # align to larger edge
        d = max(w, h)

        # handle case that patch out of image border
        if xc + d//2 > self.global_size:
            x = self.global_size - d
        else:
            x = max(0, xc - d//2)
        if yc + d//2 > self.global_size:
            y = self.global_size - d
        else:
            y = max(0, yc - d//2)

        # short edge is width or height
        short_edge = 0 if img.size[0] < img.size[1] else 1
        x = int(x / self.global_size * img.size[short_edge])
        y = int(y / self.global_size * img.size[short_edge])
        d = int(d / self.global_size * img.size[short_edge])
        if short_edge == 0:
            y += (img.size[1] - img.size[0]) // 2
        else:
            x += (img.size[0] - img.size[1]) // 2
            
        img_p = img.crop((x, y, x+d, y+d))
        img_p = self.t(img_p)

        return img_p


