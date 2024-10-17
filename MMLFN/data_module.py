### Import
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import pytorch_lightning as pl
from torchvision.transforms import v2
import albumentations as A
from PIL import Image
import os
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

###


# @title CVUSADataset class definition

# Expected dataset structure: the input_dir contains the split cvs files and a
# subdirectory named 'data' with the CVUSA dataset

class CVUSADataset(Dataset):

    def __init__(self, input_dir, split = 'train', polar = False):
        self.split = split
        self.polar = polar
        self.data = self.load_data(input_dir + f'/{split}.csv')


    def load_data(self, csv_path):
        data = []
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) #skip header
            for row in csv_reader:
                grd_path = row[1]
                grd_seg_path = row[5]
                grd_dep_path = row[6]
                if self.polar: #If we want to use polar
                   sat_path = row[3]
                   sat_seg_path = row[4]
                else:
                  sat_path = row[0]
                  sat_seg_path = row[2]
                data.append({"grd_path": grd_path, "grd_seg_path": grd_seg_path, "grd_dep_path": grd_dep_path, "sat_path": sat_path, "sat_seg_path": sat_seg_path})

        return data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        dictionary = self.data[index]
        grd_path = dictionary['grd_path']
        grd_seg_path = dictionary['grd_seg_path']
        grd_dep_path = dictionary['grd_dep_path']
        sat_path = dictionary['sat_path']
        sat_seg_path = dictionary['sat_seg_path']
        return grd_path, grd_seg_path, grd_dep_path, sat_path, sat_seg_path


    def __str__(self):
        return f"CVUSA-Dataset-{self.split}: {len(self.data)} samples"
    


# @title CVUSADataModule class definition

class CVUSADataModule(pl.LightningDataModule):

    def __init__(
        self,
        input_dir,
        polar = False,
        batch_size = 8,
        grd_resize = None,
        grd_seg_resize = None,
        grd_dep_resize = None,
        sat_resize = None,
        sat_seg_resize = None,
        augmentations = False,
        fov = 360,
        all_rgb = False
    ):
        super(CVUSADataModule, self).__init__()
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.data_dir = input_dir  # + '/data' I made this choice for semplicity!!! """IMPORTANT!!!"
        self.polar = polar
        self.augmentations = augmentations
        self.fov = fov
        self.all_rgb = all_rgb

        assert fov >= 1 and fov <= 360

        self.original_size = {'grd': None, 'grd_seg': None, 'grd_dep': None, 'sat': None, 'sat_seg': None}
        self.resize = {'grd': grd_resize, 'grd_seg': grd_seg_resize, 'grd_dep': grd_dep_resize, 'sat': sat_resize, 'sat_seg': sat_seg_resize}
        self.size = {'grd': None, 'grd_seg': None, 'grd_dep': None, 'sat': None, 'sat_seg': None}
        self.mean = {'grd': [0, 0, 0], 'grd_seg': [0, 0, 0], 'grd_dep': [0, 0, 0], 'sat': [0, 0, 0], 'sat_seg': [0, 0, 0]}
        self.std = {'grd': [1, 1, 1], 'grd_seg': [1, 1, 1], 'grd_dep': [1, 1, 1], 'sat': [1, 1, 1], 'sat_seg': [1, 1, 1]}
        self.transform = {'grd': None, 'grd_seg': None, 'grd_dep': None, 'sat': None, 'sat_seg': None}
        self.train_transform = {'grd': None, 'grd_seg': None, 'grd_dep': None, 'sat': None, 'sat_seg': None}


        if not polar:
            mean_std = {'grd_mean': [0.4691, 0.4821, 0.4603],'grd_std': [0.2202, 0.2191, 0.2583],
                        'grd_seg_mean': [0.2976, 0.7013, 0.3604],'grd_seg_std': [0.2777, 0.3306, 0.4343],
                        'grd_dep_mean': [0.3874, 0.166 , 0.1971],'grd_dep_std': [0.3763, 0.2308, 0.171 ],
                        'sat_mean': [0.3833, 0.3964, 0.3434],'sat_std': [0.1951, 0.1833, 0.1934],
                        'sat_seg_mean': [0.2861, 0.8014, 0.8299],'sat_seg_std': [0.4468, 0.3955, 0.3707]
                        }
        else:
            mean_std = {'grd_mean': [0.4691, 0.4821, 0.4603],'grd_std': [0.2202, 0.2191, 0.2583],
                        'grd_seg_mean': [0.2976, 0.7013, 0.3604],'grd_seg_std': [0.2777, 0.3306, 0.4343],
                        'grd_dep_mean': [0.3874, 0.166 , 0.1971],'grd_dep_std': [0.3763, 0.2308, 0.171 ],
                        'sat_mean': [0.4   , 0.4128, 0.3647],'sat_std': [0.1966, 0.1862, 0.1993],
                        'sat_seg_mean': [0.3349, 0.7968, 0.8518],'sat_seg_std': [0.4638, 0.3969, 0.3478]
                        }

        self.set_mean_std(mean_std)


    def setup(self):
        # load the datasets
        self.train_dataset = CVUSADataset(input_dir=self.input_dir, split='train_updated', polar=self.polar)
        self.val_dataset = CVUSADataset(input_dir=self.input_dir, split='val_updated', polar=self.polar)

        print(self.train_dataset)
        print(self.val_dataset)

        # find image sizes
        self.__compute_image_sizes()

        # compute transforms
        self.__compute_transforms()


    def __compute_image_sizes(self):
        grd_sample, grd_seg_sample, grd_dep_sample, sat_sample, sat_seg_sample = self.train_dataset[0]

        grd_image = v2.ToImage()(Image.open(os.path.join(self.data_dir, grd_sample)))
        grd_seg_image = v2.ToImage()(Image.open(os.path.join(self.data_dir, grd_seg_sample)))
        grd_dep_image = v2.ToImage()(Image.open(os.path.join(self.data_dir, grd_dep_sample)))
        sat_image = v2.ToImage()(Image.open(os.path.join(self.data_dir, sat_sample)))
        sat_seg_image = v2.ToImage()(Image.open(os.path.join(self.data_dir, sat_seg_sample)))

        self.original_size['grd'] = grd_image.size()[1:3]
        self.original_size['grd_seg'] = grd_seg_image.size()[1:3]
        self.original_size['grd_dep'] = grd_dep_image.size()[1:3]
        self.original_size['sat'] = sat_image.size()[1:3]
        self.original_size['sat_seg'] = sat_seg_image.size()[1:3]

        self.size['grd'] = grd_image.size()[1:3]
        self.size['grd_seg'] = grd_seg_image.size()[1:3]
        self.size['grd_dep'] = grd_dep_image.size()[1:3]
        self.size['sat'] = sat_image.size()[1:3]
        self.size['sat_seg'] = sat_seg_image.size()[1:3]

        # compute image new sizes
        for key in self.resize:
            if self.resize[key]:
                proportion = self.original_size[key][1] / self.original_size[key][0]
                width_resize = int(proportion * self.resize[key])
                self.size[key] = (self.resize[key], width_resize)
                # image = v2.ToImage()(Image.open(os.path.join(self.data_dir, locals()[f"{key}_sample"])))
                # self.size[key] = v2.Resize((self.resize[key]))(image).size()[1:3]

        # compute sizes with fov
        if self.fov != 360:
            for key in self.size:
                if 'grd' in key:
                    self.size[key] = (self.size[key][0], int(self.size[key][1] * (self.fov / 360)))


    def __compute_transforms(self):
        for key in self.transform:
            self.transform[key] = A.Compose([
                A.Resize(self.size[key][0], self.size[key][1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                A.Normalize(self.mean[key], self.std[key]),
                ToTensorV2()
            ])

        self.train_transform = self.transform.copy()

        if self.augmentations:
          # Applied to everything now!
          for key in self.train_transform:
              grid_dropout_ratio = 0.4 if 'sat' in key else 0.5

              self.train_transform[key] = A.Compose([
                  A.Resize(self.size[key][0], self.size[key][1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                  # New Transforms
                  A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                  A.OneOf([
                      A.AdvancedBlur(p=1.0),
                      A.Sharpen(p=1.0),
                  ], p=0.3),
                  A.OneOf([
                      A.GridDropout(ratio=grid_dropout_ratio, p=1.0),
                      A.CoarseDropout(
                          max_holes=25,
                          max_height=int(0.2 * self.size[key][0]),
                          max_width=int(0.2 * self.size[key][0]),
                          min_holes=10,
                          min_height=int(0.1 * self.size[key][0]),
                          min_width=int(0.1 * self.size[key][0]),
                          p=1.0
                      ),
                  ], p=0.3),
                  ###
                  A.Normalize(self.mean[key], self.std[key]),
                  ToTensorV2()
              ])


    def train_collate_fn(self, batch):
        return self.collate_fn(batch, 'train')


    def val_collate_fn(self, batch):
        return self.collate_fn(batch, 'val')


    def collate_fn(self, batch, dataset):
        grd_path, grd_seg_path, grd_dep_path, sat_path, sat_seg_path = zip(*batch)

        # load and transform each image in the batch
        if not self.all_rgb:

            grd_ids, grd_images = self.__compute_images(grd_path, 'grd', dataset)
            grd_seg_ids, grd_seg_images = self.__compute_images(grd_seg_path, 'grd_seg', dataset)
            grd_dep_ids, grd_dep_images = self.__compute_images(grd_dep_path, 'grd_dep', dataset)
            sat_ids, sat_images = self.__compute_images(sat_path, 'sat', dataset)
            sat_seg_ids, sat_seg_images = self.__compute_images(sat_seg_path, 'sat_seg', dataset)

            grd_samples = {'imgs': grd_images, 'imgs_id': grd_ids}
            grd_seg_samples = {'imgs': grd_seg_images, 'imgs_id': grd_seg_ids}
            grd_dep_samples = {'imgs': grd_dep_images, 'imgs_id': grd_dep_ids}
            sat_samples = {'imgs': sat_images, 'imgs_id': sat_ids}
            sat_seg_samples = {'imgs': sat_seg_images, 'imgs_id': sat_seg_ids}

        # instead of segmentation and depth use the RGB image
        elif self.all_rgb:

            grd_ids, grd_images = self.__compute_images(grd_path, 'grd', dataset)
            sat_ids, sat_images = self.__compute_images(sat_path, 'sat', dataset)

            grd_samples = {'imgs': grd_images, 'imgs_id': grd_ids}
            grd_seg_samples = grd_samples
            grd_dep_samples = grd_samples
            sat_samples = {'imgs': sat_images, 'imgs_id': sat_ids}
            sat_seg_samples = sat_samples

        return grd_samples, grd_seg_samples, grd_dep_samples, sat_samples, sat_seg_samples


    def __compute_images(self, paths, img_type, dataset):
        images = []
        ids = []

        for img_path in paths:
            img = cv2.imread(os.path.join(self.data_dir, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # fov
            if 'grd' in img_type and self.fov != 360:
                old_width = self.original_size[img_type][1]
                new_width = int(old_width * (self.fov / 360))
                j = np.arange(0, old_width)
                random_shift = int(np.random.rand() * old_width)
                # assuming h,w,c
                img = img[:, ((j - random_shift) % old_width)[:new_width], :]

            if dataset == 'train':
                img = self.train_transform[img_type](image = img)['image']
            else:
                img = self.transform[img_type](image = img)['image']

            images.append(img)

            ids.append(int(img_path[-11:-4]))

        # Stack the image tensors along the batch dimension
        images_tensor = torch.stack(images)
        ids_tensor = torch.tensor(ids, dtype=int)
        return ids_tensor, images_tensor


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.train_collate_fn, shuffle=True, num_workers=4, pin_memory = torch.cuda.is_available())


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.val_collate_fn, shuffle=False, num_workers=4, pin_memory = torch.cuda.is_available())


    def compute_mean_std(self):
        mean_std = {}

        for key in self.mean:
            mean, std = self.__compute_mean_std_for_key(key)
            mean_std[key + '_mean'] = mean
            mean_std[key + '_std'] = std

        return mean_std


    def __compute_mean_std_for_key(self, key):
        mean = np.array([0., 0., 0.])
        std = np.array([0., 0., 0.])

        for i in self.train_dataset.data:
            img_path = os.path.join(self.data_dir, i[f'{key}_path'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(float) / 255.
            img_size = img.shape[0] * img.shape[1]
            mean += np.mean(img[:, :, :], axis=(0, 1))
            std += ((img[:, :, :] - mean) ** 2).sum(axis=(0, 1)) / img_size

        mean /= len(self.train_dataset.data)
        std = np.sqrt(std / len(self.train_dataset.data))

        return mean, std


    def set_mean_std(self, mean_std):
        for key in self.mean:
            self.mean[key] = mean_std[f'{key}_mean']
            self.std[key] = mean_std[f'{key}_std']
