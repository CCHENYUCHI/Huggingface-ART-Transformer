
import os
import numpy as np
import h5py
import mne
import json
import torch
import time
import random
from scipy.signal import decimate, resample_poly, firwin, lfilter
from scipy.signal import resample
from torch.utils.data import Dataset

def txt_to_dict_with_list(txt_file):
    try:
        with open(txt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        result = {}
        for line in lines:
            line = line.strip()  # 移除空白與換行
            if not line:  # 跳過空行
                continue
            parts = line.split(",")
            filename = parts[0].strip(".set")
            if len(parts) > 1:
                # 將 index 切割為 list
                index = [int(x) for x in parts[1].strip().split()]
            else:
                index = None
            result[filename] = index

        return result
    except Exception as e:
        print(f"發生錯誤: {e}")
        return None
    
class EEGVoxelDataset(Dataset):
    def __init__(self, roi_folder, eeg_folder, group_file , group_index, overlap=0.5, window_size=500):
        """
        Args:
            roi_folder (str): Path to the folder containing ROI .set files.
            eeg_folder (str): Path to the folder containing EEG .set files.
            overlap (float): Fraction of overlap between consecutive windows (0 <= overlap < 1).
            window_size (int): Number of samples in each window.
        """
        self.roi_folder = roi_folder
        self.eeg_folder = eeg_folder
        self.group_file = group_file
        self.group_index = group_index
        self.overlap = overlap
        self.window_size = window_size
        self.subjects = self._get_subject_list()
        self.segment_file = "G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\test_data\\roi_removal_segment.txt"
        self.segment_index = txt_to_dict_with_list(self.segment_file)

        self.eeg_data = []  # Will store tuples of (ROI segment, EEG segment)
        self.roi_data = []
        self._prepare_dataset()

    def _get_subject_list(self):
        """Gets the list of subjects based on file names in the ROI folder."""
        with open(self.group_file, 'r') as f:
            groups = json.load(f)

        subject_indices = groups.get(str(self.group_index), [])
        print(subject_indices)
        return subject_indices 

    def _prepare_dataset(self):
        """Reads and processes data for all subjects."""
        for subject in self.subjects:
            # start_time = time.time()
            roi_path = os.path.join(self.roi_folder, f"processed_{subject}_ICA_DLtrain.set")
            eeg_path = os.path.join(self.eeg_folder, f"{subject}_ICA_DLtrain.set")

            # Load ROI data
            with h5py.File(roi_path, 'r') as f:
                if 'roi' in f:
                    roi_data = f['roi']['source_voxel_data'][:]
                    print(f"ROI Times: {roi_data.shape[2] / 100}")

            # Load EEG data
            eeg_data = mne.io.read_raw_eeglab(eeg_path, preload=True).get_data()
            # end_time = time.time()
            # print(f"Load Data time: {end_time - start_time}")
            # Verify dimensions
            assert roi_data.shape[0] == 3, f"Unexpected ROI shape: {roi_data.shape}"
            assert roi_data.shape[1] == 5003, f"Unexpected ROI shape: {roi_data.shape}"
            assert eeg_data.shape[0] == 30, f"Unexpected EEG shape: {eeg_data.shape}"

            # Process and overlap data
            # start_time = time.time()
            self._process_subject_data(roi_data, eeg_data)
            # end_time = time.time()
            # print(f"Overlapping time: {end_time - start_time}")
            
    def _process_subject_data(self, roi_data, eeg_data):
        """Segments and overlaps data for a single subject."""
        time_len = int(int(eeg_data.shape[1] / 256) / 2)*2
        # print(time_len)
        eeg_window_size = 256 * 2
        roi_window_size = 100 * 2
        nan_count=0
        for start_idx in range(0, time_len, 2):
            eeg_step = start_idx * 256
            eeg_segment = torch.tensor(eeg_data[:, eeg_step:eeg_step+eeg_window_size], dtype=torch.float32)
            eeg_mean = torch.mean(eeg_segment, dim=0, keepdim=True)
            eeg_std = torch.std(eeg_segment, dim=0, keepdim=True)
            eeg_segment = (eeg_segment - eeg_mean) / (eeg_std + 1e-10)

            roi_step = start_idx * 100
            roi_segment = roi_data[:, :, roi_step:roi_step+roi_window_size]
            
            if roi_segment.shape[2] == roi_window_size:
                roi_segment_reshape = torch.tensor(roi_segment.reshape(-1, roi_window_size), dtype=torch.float32)
                # print(roi_segment_reshape.shape)
                if not (torch.isnan(eeg_segment).any() or torch.isinf(eeg_segment).any()):
                    if not (torch.isnan(roi_segment_reshape).any() or torch.isinf(roi_segment_reshape).any()):
                        self.roi_data.append(roi_segment_reshape)
                        self.eeg_data.append(eeg_segment)
                    else:
                        nan_count+=1

                else:
                    nan_count+=1
            else:
                
                break
        print(f"Total Nan of the subject: {nan_count}")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            "src": self.eeg_data[idx], 
            "tgt": self.eeg_data[idx], 
            "src_mask": None,
            "tgt_mask": None,
            "label": self.roi_data[idx]
        }
    
    
class EEGROIDataset(Dataset):
    def __init__(self, roi_folder, eeg_folder, group_file , group_index, overlap=0.5, window_size=512, 
                 segment_file= "G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\test_data\\ROI\\Desikan_Kilianny_with_3pca\\roi_removal_segment.txt"):
        """
        Args:
            roi_folder (str): Path to the folder containing ROI .set files.
            eeg_folder (str): Path to the folder containing EEG .set files.
            overlap (float): Fraction of overlap between consecutive windows (0 <= overlap < 1).
            window_size (int): Number of samples in each window.
        """
        self.roi_folder = roi_folder
        self.eeg_folder = eeg_folder
        self.group_file = group_file
        self.group_index = group_index
        self.overlap = overlap
        self.window_size = window_size
        self.subjects = self._get_subject_list()
        self.segment_file = segment_file
        self.segment_index = txt_to_dict_with_list(self.segment_file)

        self.eeg_data = []  # Will store tuples of (ROI segment, EEG segment)
        self.roi_data = []
        self._prepare_dataset()

    def _get_subject_list(self):
        """Gets the list of subjects based on file names in the ROI folder."""
        with open(self.group_file, 'r') as f:
            groups = json.load(f)

        subject_indices = groups.get(str(self.group_index), [])
        print(subject_indices)
        return subject_indices 

    def _prepare_dataset(self):
        """Reads and processes data for all subjects."""
        for subject in self.subjects:
            # start_time = time.time()
            roi_path = os.path.join(self.roi_folder, f"processed_{subject}_ICA_DLtrain.set")
            eeg_path = os.path.join(self.eeg_folder, f"processed_{subject}_ICA_DLtrain.set")

            # Load ROI data
            with h5py.File(roi_path, 'r') as f:
                if 'roi' in f:
                    roi_data = f['roi']['source_roi_data'][:]


            # Load EEG data
            eeg_data = mne.io.read_raw_eeglab(eeg_path, preload=True).get_data()
            # end_time = time.time()
            # print(f"Load Data time: {end_time - start_time}")
            # Verify dimensions
            # assert roi_data.shape[0] == 3, f"Unexpected ROI shape: {roi_data.shape}"
            assert roi_data.shape[1] == 200, f"Unexpected ROI shape: {roi_data.shape}"
            assert eeg_data.shape[0] == 30, f"Unexpected EEG shape: {eeg_data.shape}"

            # Process and overlap data
            # start_time = time.time()
            self._process_subject_data(roi_data, eeg_data, subject_name=f"{subject}_ICA_DLtrain")
            # end_time = time.time()
            # print(f"Overlapping time: {end_time - start_time}")
            
    def _process_subject_data(self, roi_data, eeg_data, subject_name):
        """Segments and overlaps data for a single subject."""

        segment_len = roi_data.shape[0]
        nan_count=0
        rand_count = 0

        for start_idx in range(0, segment_len):
            # EEG [0~511] [512~1023]
            if start_idx not in self.segment_index[subject_name]:
                if random.random() < 1/3:
                    continue

                eeg_segment_start = 256 * (start_idx)
                eeg_segment_end = 256 * (start_idx+2)
                
                eeg = eeg_data[:, eeg_segment_start:eeg_segment_end]
                # EEG resample 
                EEG_resample = []
                for ch in eeg:
                    x_down = resample(ch, 200)
                    # print(ch.shape, x_down.shape)
                    EEG_resample.append(x_down)
                EEG = np.array(EEG_resample)
                # print(EEG.shape)
                # Normalizer
                eeg_raw = torch.tensor(EEG, dtype=torch.float32)
                eeg_mean = torch.mean(eeg_raw, dim=0, keepdim=True)
                eeg_std = torch.std(eeg_raw, dim=0, keepdim=True)
                EEG = (eeg_raw - eeg_mean) / (eeg_std + 1e-10)

                ROI = torch.tensor(roi_data[start_idx, :, :], dtype=torch.float32).squeeze(0).transpose(0, 1)

                # print(EEG.shape, ROI.shape)
                assert ROI.shape[0] == 204, f"Unexpected ROI shape: {ROI.shape}"
                assert ROI.shape[1] == 200, f"Unexpected ROI shape: {ROI.shape}"
                assert EEG.shape[0] == 30, f"Unexpected EEG shape: {EEG.shape}"
                assert EEG.shape[1] == 200, f"Unexpected EEG shape: {EEG.shape}"

                if not (torch.isnan(EEG).any() or torch.isinf(EEG).any()):
                    if not (torch.isnan(ROI).any() or torch.isinf(ROI).any()):
                        self.roi_data.append(ROI)
                        self.eeg_data.append(EEG)
                    else:
                        nan_count+=1

                else:
                    nan_count+=1
                rand_count += 1

        print(f"Total Nan of the subject: {nan_count}")
        print(f"Total Random of the subject: {rand_count}")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            "src": self.eeg_data[idx], 
            "tgt": self.eeg_data[idx], 
            "src_mask": None,
            "tgt_mask": None,
            "label": self.roi_data[idx]
        }
   

# For Huggingface Trainer
class SignalDataCollator:
    def __call__(self, features):
        inputs = torch.stack([f["src"] for f in features])
        masks = None # torch.stack([f["src_mask"] for f in features])
        labels = torch.stack([f["label"] for f in features])
        return_dict = True
        return {"src": inputs, 
                "tgt":inputs, 
                "src_mask": masks, 
                "tgt_mask": masks, 
                "labels": labels, 
                "return_dict": return_dict}