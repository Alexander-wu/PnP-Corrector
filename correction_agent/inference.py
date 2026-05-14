import os
import sys
import time
import glob
import h5py
import logging
import argparse
import numpy as np
from icecream import ic
from datetime import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from my_utils.YParams import YParams
from my_utils.data_loader import get_data_loader
from my_utils import logging_utils
logging_utils.config_logger()


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # get data loader
    # valid_data_loader, valid_dataset = get_data_loader(params, params.test_data_path, dist.is_initialized(), train=False)

    img_shape_x = 180
    img_shape_y = 360
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    atmos_channels = np.array(params.atmos_channels)
    ocean_channels = np.array(params.ocean_channels)
    n_atmos_channels = len(atmos_channels)
    n_ocean_channels = len(ocean_channels)

    if params.normalization == 'zscore': 
        params.means_atmos = np.load(params.global_means_path)
        params.stds_atmos = np.load(params.global_stds_path)

        params.means_ocean = np.load(params.global_means_path_ocean)
        params.stds_ocean = np.load(params.global_stds_path_ocean)

        params.means_sst = np.load(params.global_means_path_sst)
        params.stds_sst = np.load(params.global_stds_path_sst)

    if params.nettype == 'DSLCast':
        from networks.DSLCast_atmos import DSLCast as model
        from networks.DSLCast_ocean import DSLCast as model2
        from networks.DSLCast_coupler import DSLCast as model3
    else:
        raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    checkpoint_file2  = params['best_checkpoint_path2']
    checkpoint_file3  = params['best_checkpoint_path3']
    logging.info('Loading trained model checkpoint from {}'.format(checkpoint_file))
    logging.info('Loading trained model2 checkpoint from {}'.format(checkpoint_file2))
    logging.info('Loading trained model3 checkpoint from {}'.format(checkpoint_file3))
    
    model = model(params).to(device) 
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    print('model is ok')

    model2 = model2(params).to(device) 
    model2 = load_model(model2, params, checkpoint_file2)
    model2 = model2.to(device)

    print('model2 is ok')

    model3 = model3(params).to(device) 
    model3 = load_model(model3, params, checkpoint_file3)
    model3 = model3.to(device)

    print('model3 is ok')
    
    files_paths_atmos = glob.glob(params.test_data_path + "/*.h5")
    files_paths_atmos.sort()

    files_paths_ocean = glob.glob(params.test_data_path_ocean + "/*.h5")
    files_paths_ocean.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(files_paths_atmos[yr]))
    logging.info('Inference data_ocean from {}'.format(files_paths_ocean[yr]))
    climate_mean = np.load('../data/ocean/climate_mean_s_t_ssh.npy', mmap_mode='r')
    valid_data_full_atmos = h5py.File(files_paths_atmos[yr], 'r')['fields'][:365, :, :, :]
    valid_data_full_ocean = h5py.File(files_paths_ocean[yr], 'r')['fields'][:365, :, :, :]
    valid_data_full_ocean = valid_data_full_ocean - climate_mean
    valid_data_full_sst = h5py.File(files_paths_ocean[yr], 'r')['fields'][:365, 69:70, :, :]

    return valid_data_full_atmos, valid_data_full_ocean, valid_data_full_sst, model, model2, model3, climate_mean

    
def autoregressive_inference(params, init_condition, valid_data_full_atmos, valid_data_full_ocean, valid_data_full_sst, model, model2, model3, climate_mean): 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
    icd = int(init_condition) 
    
    exp_dir = params['experiment_dir'] 
    dt                = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history      = params.n_history
    img_shape_x    = params.img_shape_x
    img_shape_y    = params.img_shape_y
    atmos_channels = np.array(params.atmos_channels)
    ocean_channels = np.array(params.ocean_channels)
    n_atmos_channels  = len(atmos_channels)
    n_ocean_channels  = len(ocean_channels)
    
    seq_real        = torch.zeros((prediction_length, n_atmos_channels + n_ocean_channels, img_shape_x, img_shape_y))
    seq_pred        = torch.zeros((prediction_length, n_atmos_channels + n_ocean_channels, img_shape_x, img_shape_y))

    valid_data_atmos = valid_data_full_atmos[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.atmos_channels][:,:,0:180]
    valid_data_ocean = valid_data_full_ocean[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.ocean_channels][:,:,0:180]
    valid_data_sst = valid_data_full_sst[icd:(icd+prediction_length*dt+n_history*dt):dt][:, :][:,:,0:180]
    valid_data_climate_mean = climate_mean[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.ocean_channels][:,:,0:180]
    logging.info(f'valid_data_full_atmos: {valid_data_full_atmos.shape}')
    logging.info(f'valid_data_atmos: {valid_data_atmos.shape}')
    logging.info(f'valid_data_full_ocean: {valid_data_full_ocean.shape}')
    logging.info(f'valid_data_ocean: {valid_data_ocean.shape}')
    logging.info(f'valid_data_full_sst: {valid_data_full_sst.shape}')
    logging.info(f'valid_data_sst: {valid_data_sst.shape}')
    logging.info(f'climate_mean: {climate_mean.shape}')
    logging.info(f'valid_data_climate_mean: {valid_data_climate_mean.shape}')
    
    # normalize
    if params.normalization == 'zscore': 
        valid_data_atmos = (valid_data_atmos - params.means_atmos[:,params.atmos_channels])/params.stds_atmos[:,params.atmos_channels]
        valid_data_atmos = np.nan_to_num(valid_data_atmos, nan=0)

        valid_data_ocean = (valid_data_ocean - params.means_ocean[:,params.ocean_channels])/params.stds_ocean[:,params.ocean_channels]
        valid_data_ocean = np.nan_to_num(valid_data_ocean, nan=0)

        valid_data_sst = (valid_data_sst - params.means_sst[:, 69:70, :, :])/params.stds_sst[:, 69:70, :, :]
        valid_data_sst = np.nan_to_num(valid_data_sst, nan=0)
        
    valid_data_atmos = torch.as_tensor(valid_data_atmos)
    valid_data_ocean = torch.as_tensor(valid_data_ocean)
    valid_data_sst = torch.as_tensor(valid_data_sst)
    valid_data_climate_mean = torch.as_tensor(valid_data_climate_mean)

    # autoregressive inference
    logging.info('Begin autoregressive inference')
    
    
    with torch.no_grad():
        for i in range(valid_data_atmos.shape[0]): 
            if i==0: # start of sequence, t0 --> t0'
                first_atmos = valid_data_atmos[0:n_history+1]
                first_ocean = valid_data_ocean[0:n_history+1]
                first_sst = valid_data_sst[0:n_history+1]
                first = torch.cat((first_atmos, first_ocean), axis=1)
                ic(first_atmos.shape, first_ocean.shape, first.shape)
                future_atmos = valid_data_atmos[n_history+1]
                future_ocean = valid_data_ocean[n_history+1]
                future = torch.cat((future_atmos, future_ocean), axis=0)
                ic(future.shape)

                for h in range(n_history+1):
                    seq_real[h] = first[:, :, :, :]
                    seq_pred[h] = seq_real[h]

                first_atmos = first_atmos.to(device, dtype=torch.float)
                first_ocean = first_ocean.to(device, dtype=torch.float)
                first_sst = first_sst.to(device, dtype=torch.float)
                model_input = torch.cat((first_atmos, first_sst), axis=1)
                ic(first_atmos.shape, first_ocean.shape, first_sst.shape, model_input.shape)
                model_future_pred = model(model_input)

                atmos_forcing0 = first_atmos[:, [65, 66, 67, 68], :, :]
                atmos_forcing1 = model_future_pred[:, [65, 66, 67, 68], :, :]
                model2_input = torch.cat((first_ocean, atmos_forcing0, atmos_forcing1), axis=1)
                model2_future_pred = model2(model2_input)
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:,:, :180, :360], dtype=bool).to(device, dtype=torch.bool)
                model2_future_pred = torch.masked_fill(input=model2_future_pred, mask=~mask_data, value=0)

                model3_input = torch.cat((model_future_pred, model2_future_pred), axis=1)
                model3_future_pred = model3(model3_input)
                model3_future_pred[:, 69:, :, :] = torch.masked_fill(input=model3_future_pred[:, 69:, :, :], mask=~mask_data, value=0)

                future_pred = model3_future_pred


            else:
                if i < prediction_length-1:
                    future_atmos = valid_data_atmos[n_history+i+1]
                    future_ocean = valid_data_ocean[n_history+i+1]
                    future = torch.cat((future_atmos, future_ocean), axis=0)
                    ic(future.shape)

                inf_one_step_start = time.time()
                    
                climate_mean_input = valid_data_climate_mean[i:i+1, 69:70, :, :]
                atmos_input = future_pred[:, :69, :, :]
                ssta_norm = future_pred[:, 138:139, :, :]
                ssta = ssta_norm * params.stds_ocean[:, 69:70, :, :] + params.means_ocean[:, 69:70, :, :]
                sst = ssta + climate_mean_input
                sst_norm = (sst - params.means_sst[:, 69:70, :, :]) / params.stds_sst[:, 69:70, :, :]
                sst_norm = torch.nan_to_num(sst_norm, nan=0.0)
                model_input = torch.cat((atmos_input.to(device, dtype=torch.float), sst_norm.to(device, dtype=torch.float)), axis=1)
                model_future_pred = model(model_input)

                atmos_forcing0 = atmos_input[:, [65, 66, 67, 68], :, :]
                atmos_forcing1 = model_future_pred[:, [65, 66, 67, 68], :, :]
                ocean_input = future_pred[:, 69:, :, :]
                model2_input = torch.cat((ocean_input.to(device, dtype=torch.float), atmos_forcing0.to(device, dtype=torch.float), atmos_forcing1), axis=1).to(device, dtype=torch.float)
                model2_future_pred = model2(model2_input)
                model2_future_pred = torch.masked_fill(input=model2_future_pred, mask=~mask_data, value=0)

                model3_input = torch.cat((model_future_pred, model2_future_pred), axis=1).to(device, dtype=torch.float)
                model3_future_pred = model3(model3_input)
                model3_future_pred[:, 69:, :, :] = torch.masked_fill(input=model3_future_pred[:, 69:, :, :], mask=~mask_data, value=0)

                future_pred = model3_future_pred

                inf_one_step_time = time.time() - inf_one_step_start

                logging.info(f'inference one step time: {inf_one_step_time}')
    

            if i < prediction_length - 1: # not on the last step
                # with h5py.File(params.land_mask_path, 'r') as _f: 
                #     mask_data = torch.as_tensor(_f['fields'][:,out_channels, :180, :360], dtype=bool)
                seq_pred[n_history+i+1] = future_pred
                seq_real[n_history+i+1] = future[:]
                history_stack = seq_pred[i+1:i+2+n_history]

            future_pred = history_stack

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar  = torch.unsqueeze(seq_real[i], 0)


            print(torch.mean((pred-tar)**2))

    
    seq_real[:, :69, :, :] = seq_real[:, :69, :, :] * params.stds_atmos[:,params.atmos_channels] + params.means_atmos[:,params.atmos_channels]
    seq_real[:, 69:, :, :] = seq_real[:, 69:, :, :] * params.stds_ocean[:,params.ocean_channels] + params.means_ocean[:,params.ocean_channels]
    seq_real = seq_real.numpy()
    seq_pred[:, :69, :, :] = seq_pred[:, :69, :, :] * params.stds_atmos[:,params.atmos_channels] + params.means_atmos[:,params.atmos_channels]
    seq_pred[:, 69:, :, :] = seq_pred[:, 69:, :, :] * params.stds_ocean[:,params.ocean_channels] + params.means_ocean[:,params.ocean_channels]
    seq_pred = seq_pred.numpy()
   

    return (np.expand_dims(seq_real[n_history:], 0), 
            np.expand_dims(seq_pred[n_history:], 0), 
           )     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='../exp_15_levels', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--prediction_length", default=61, type=int)
    parser.add_argument("--finetune_dir", default='', type=str)
    parser.add_argument("--ics_type", default='default', type=str)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, args.config, args.run_num, 'config.yaml')
    params = YParams(config_path, args.config)

    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['global_batch_size']  = params.batch_size
    params['prediction_length']  = args.prediction_length
    params['multi_steps_finetune'] = 1

    torch.cuda.set_device(0)
    # torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.finetune_dir == '':
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num), args.finetune_dir)
    logging.info(f'expDir: {expDir}')
    params['experiment_dir']       = expDir 
    params['best_checkpoint_path'] = os.path.join(expDir, 'atmos/training_checkpoints/ckpt.tar')
    params['best_checkpoint_path2'] = os.path.join(expDir, 'ocean/training_checkpoints/best_ckpt.tar')
    params['best_checkpoint_path3'] = os.path.join(expDir, 'coupler/training_checkpoints/best_ckpt.tar')

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        ics = np.arange(0, 55, 1)
        n_ics = len(ics)
        print('init_condition:', ics)

    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    valid_data_full_atmos, valid_data_full_ocean, valid_data_full_sst, model, model2, model3, climate_mean = setup(params)


    seq_pred = []
    seq_real = []

    # run autoregressive inference for multiple initial conditions
    for i, ic_ in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        seq_real, seq_pred = autoregressive_inference(params, ic_, valid_data_full_atmos, valid_data_full_ocean, valid_data_full_sst, model, model2, model3, climate_mean)

        prediction_length = seq_real[0].shape[0]
        n_out_channels = seq_real[0].shape[1]
        img_shape_x = seq_real[0].shape[2]
        img_shape_y = seq_real[0].shape[3]

        # save predictions and loss
        save_path = os.path.join(params['experiment_dir'], 'results_forecasting.h5')
        logging.info("Saving to {}".format(save_path))
        print(f'saving to {save_path}')
        if i==0:
            f = h5py.File(save_path, 'w')
            f.create_dataset(
                    "ground_truth",
                    data=seq_real,
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.create_dataset(
                    "predicted",       
                    data=seq_pred, 
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.close()
        else:
            f = h5py.File(save_path, 'a')

            f["ground_truth"].resize((f["ground_truth"].shape[0] + 1), axis = 0)
            f["ground_truth"][-1:] = seq_real 

            f["predicted"].resize((f["predicted"].shape[0] + 1), axis = 0)
            f["predicted"][-1:] = seq_pred 
            f.close()

