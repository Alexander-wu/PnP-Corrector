
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
# import cv2
from my_utils.norm import reshape_fields_atmos, reshape_fields_ocean, reshape_fields_land, reshape_fields_sst
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
climate_mean_path = os.path.join(current_dir, '../../data/ocean/climate_mean_s_t_ssh.npy')
climate_mean_path = os.path.abspath(climate_mean_path)


def get_data_loader(params, files_pattern, files_pattern_ocean, files_pattern_land, distributed, train):
    dataset = GetDataset(params, files_pattern, files_pattern_ocean, files_pattern_land, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    dataloader = DataLoader(dataset,
                            batch_size  = int(params.batch_size),
                            num_workers = params.num_data_workers,
                            shuffle     = False,  # (sampler is None),
                            sampler     = sampler if train else None,
                            drop_last   = True,
                            pin_memory  = True)

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, location_ocean, location_land, train):
        self.params = params
        self.location = location
        self.location_ocean = location_ocean
        self.location_land = location_land
        self.train = train
        self.orography = params.orography
        self.normalize = params.normalize
        self.dt = params.dt  # 需要预测的时间步
        self.n_history = params.n_history # 输入的时间步
        self.sst_channels = np.array(params.sst_channels)
        self.ocean_channels = np.array(params.ocean_channels)
        self.atmos_channels = np.array(params.atmos_channels)
        self.land_channels = np.array(params.land_channels)
        self.n_ocean_channels = len(self.ocean_channels)
        self.n_atmos_channels = len(self.atmos_channels)
        self.n_land_channels = len(self.land_channels)
        self.n_sst_channels = len(self.sst_channels)

        self._get_files_stats()
        self.add_noise = params.add_noise if train else False
        self.fusion_3d_2d = params.fusion_3d_2d
        self.climate_mean = np.load(climate_mean_path, mmap_mode='r')


    # 获取文件统计信息
    def _get_files_stats(self):
        self.files_paths = glob.glob(f"{self.location}/19[9][3-9].h5") + glob.glob(f"{self.location}/20*.h5")
        self.files_paths_ocean = glob.glob(self.location_ocean + "/*.h5")
        self.files_paths_land = glob.glob(self.location_land + "/*.h5")
        self.files_paths.sort()
        self.files_paths_ocean.sort()
        self.files_paths_land.sort()
        self.n_years = len(self.files_paths)
        logging.info('files_paths: %s' % self.files_paths)
        logging.info('files_paths_ocean: %s' % self.files_paths_ocean)
        logging.info('files_paths_land: %s' % self.files_paths_land)
        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))

            # self.n_samples_per_year = _f['fields'].shape[0] - 1  
            self.n_samples_per_year = _f['fields'].shape[0] - self.params.multi_steps_finetune 

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] - 1 # just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        self.files_ocean = [None for _ in range(self.n_years)]
        self.files_land = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Atmos Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_atmos_channels))
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(
            1 * self.dt * self.n_history, 1 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        _file_ocean = h5py.File(self.files_paths_ocean[year_idx], 'r')
        _file_land = h5py.File(self.files_paths_land[year_idx], 'r')
        self.files[year_idx] = _file['fields']
        self.files_ocean[year_idx] = _file_ocean['fields'] 
        self.files_land[year_idx] = _file_land['fields']

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        step = self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
            # logging.info(f'orog: {orog.shape}')
        else:
            orog = None
        

        if self.params.multi_steps_finetune == 1:
            if local_idx == 365:
                local_idx = 364
            
            inp_climate_mean_ocean = self.climate_mean[(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :180, :360]
            inp_ocean = reshape_fields_ocean( 
                    self.files_ocean[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :180, :360] - inp_climate_mean_ocean, 
                    'ocean', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            inp_ocean = np.nan_to_num(inp_ocean, nan=0)

            
            inp_sst = reshape_fields_sst( 
                    self.files_ocean[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.sst_channels, :180, :360], 
                    'sst', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            inp_sst = np.nan_to_num(inp_sst, nan=0)
            inp_sst = np.expand_dims(inp_sst, axis=0)

            
            inp_atmos = reshape_fields_atmos( 
                    np.nan_to_num(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.atmos_channels, :180, :360], nan=0), 
                    'atmos', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )

            inp_land = reshape_fields_land( 
                    np.nan_to_num(self.files_land[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.land_channels, :180, :360], nan=0), 
                    'land', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            

            tar_climate_mean = self.climate_mean[local_idx+step, self.ocean_channels, :180, :360]
            tar_ocean = reshape_fields_ocean(
                    self.files_ocean[year_idx][local_idx+step, self.ocean_channels, :180, :360] - tar_climate_mean, 
                    'ocean', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog
            )
            tar_ocean = np.nan_to_num(tar_ocean, nan=0)

            
            tar_atmos = reshape_fields_atmos(
                    np.nan_to_num(self.files[year_idx][local_idx+step, self.atmos_channels, :180, :360], nan=0), 
                    'atmos', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )

            tar_land = reshape_fields_land(
                    np.nan_to_num(self.files_land[year_idx][local_idx+step, self.land_channels, :180, :360], nan=0), 
                    'land', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )
      
        
        return np.concatenate((inp_atmos, inp_sst), axis=0), inp_ocean, inp_land, np.concatenate((tar_atmos, tar_ocean, tar_land), axis=0)
