
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
import bisect
import scipy.io
import glob

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
    
class EEG_Dataset(Dataset):
    def __init__(self, eeg_folder, group_file , group_index, window_size=200, 
                 segment_file= "G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\test_data\\ROI\\Desikan_Kilianny_with_3pca\\roi_removal_segment.txt"):
        """
        Args:
            roi_folder (str): Path to the folder containing ROI .set files.
            eeg_folder (str): Path to the folder containing EEG .set files.
            overlap (float): Fraction of overlap between consecutive windows (0 <= overlap < 1).
            window_size (int): Number of samples in each window.
        """
        self.eeg_folder = eeg_folder
        self.group_file = group_file
        self.group_index = group_index
        self.window_size = window_size
        self.subjects = self._get_subject_list()
        self.segment_file = segment_file
        self.segment_index = txt_to_dict_with_list(self.segment_file)

        self.eeg_data = []  # Will store tuples of (ROI segment, EEG segment)
        self._prepare_dataset()

    def _get_subject_list(self):
        """Gets the list of subjects based on file names in the ROI folder."""
        with open(self.group_file, 'r') as f:
            groups = json.load(f)

        subject_indices = groups.get(str(self.group_index), [])
        print(subject_indices)
        return subject_indices 

    def _prepare_dataset(self):
        """Reads and processes data for all subjects and segments them."""
        all_segments = []  # 最後儲存所有切好的 EEG 段落

        for subject in self.subjects:
            eeg_path = os.path.join(self.eeg_folder, f"processed_{subject}_ICA_DLtrain.set")

            # Load EEG data: shape = (channels=30, timepoints)
            eeg_data = mne.io.read_raw_eeglab(eeg_path, preload=True).get_data() * 1e6

            # Step 1: 確保可以整除512
            num_segments = eeg_data.shape[1] // 512
            eeg_data = eeg_data[:, :num_segments * 512]  # Trim data to be divisible

            # Step 2: reshape 為 (num_segments, 30, 512)
            eeg_segments = eeg_data.reshape(30, num_segments, 512).transpose(1, 0, 2)  # (num_segments, 30, 512)
            
            # Step 3: Downsample 每個 segment from 512 to 200
            eeg_downsampled = resample(eeg_segments, 200, axis=2)  # Output: (num_segments, 30, 200)

            all_segments.append(eeg_downsampled)

        # Step 3: 堆疊所有 subject 的段落 (total_segments, 30, 512)
        self.eeg_data = np.concatenate(all_segments, axis=0)
        print(f"Total EEG segments: {self.eeg_data.shape}")  # Should be (N, 30, 512)
        np.save(f"G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\test_data\\ROI\\EEG\\{self.group_index}.npy", self.eeg_data)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            "src": self.eeg_data[idx], 
            "label": self.eeg_data[idx]
        }

class EEGDatasetFromNPY(Dataset):
    def __init__(self, npy_file_list, fft_length=None):
        """
        Args:
            npy_file_list (list of str): List of paths to .npy files containing EEG data.
                                         Each file is assumed to have shape (N_i, C, T)
        """
        self.data = []
        for file in npy_file_list:
            eeg = np.load(file)  # shape: (N_i, C, T)
            self.data.append(eeg)
        self.data = np.concatenate(self.data, axis=0)  # shape: (N_total, C, T)
        self.fft_length = fft_length if fft_length is not None else self.data.shape[2]
        self.freq_bins = self.fft_length // 2 + 1  # Keep only positive frequencies
        self.channel_loc =  [
                    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                    "FT7", "FC3", "FCz", "FC4", "FT8",
                    "T3", "C3", "Cz", "C4", "T4",
                    "TP7", "CP3", "CPz", "CP4", "TP8",
                    "T5", "P3", "Pz", "P4", "T6",
                    "O1", "Oz", "O2"
                ]

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = self.data[idx] * 1e6  # shape: (C, T)
        # Apply FFT over time axis for each channel
        fft_result = np.fft.rfft(eeg, n=self.fft_length, axis=-1)  # shape: (C, F)
        magnitude = np.abs(fft_result)  # 取 magnitude
        mean = np.mean(magnitude, axis=(0, 1), keepdims=True)
        std = np.std(magnitude, axis=(0, 1), keepdims=True)
        src_ = (magnitude - mean) / (std)
        return {
            "src": torch.tensor(src_, dtype=torch.float32),  # shape: (C, F)
            "montage": self.channel_loc,
        }

# class TshingHwa_Dataset(Dataset):
#     def __init__(self, eeg_folder, sampling_rate=250, save_path=None):
#         """
#         Args:
#             eeg_folder (str): Path to the folder containing .mat EEG files.
#             sampling_rate (int): EEG sampling rate, default = 250Hz.
#             save_path (str): Optional path to save the merged .npy dataset.
#         """
#         self.eeg_folder = eeg_folder
#         self.sampling_rate = sampling_rate
#         self.channel_loc = [
#             "Fp1", "Fpz", "Fp2", "Af3", "Af4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
#             "Ft7", "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "Ft8", "T7", "C5", "C3", "C1",
#             "Cz", "C2", "C4", "C6", "T8", "M1", "Tp7", "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
#             "Cp6", "Tp8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "Po7", "Po5",
#             "Po3", "Poz", "Po4", "Po6", "Po8", "Cb1", "O1", "Oz", "O2", "Cb2"
#         ]


#         self.eeg_data = self._load_and_process_all()

#         if save_path:
#             np.save(save_path, self.eeg_data)

#     def _load_and_process_all(self):
#         all_fft_segments = []

#         for filename in os.listdir(self.eeg_folder):
#             if filename.endswith(".mat"):
#                 file_path = os.path.join(self.eeg_folder, filename)
#                 print(f"Processing: {filename}")
#                 fft_data = self._process_single_mat(file_path)  # (720, 64, 101)
#                 all_fft_segments.append(fft_data)

#         # Concatenate across subjects along the 0th axis
#         merged = np.concatenate(all_fft_segments, axis=0)  # (N, 64, 101)
#         print(f"Merged FFT data shape: {merged.shape}")
#         np.save(f"G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\test_data\\tshinghwa\\train_dataset.npy", merged)
#         return merged.astype(np.float32)

#     def _process_single_mat(self, file_path):
#         mat = scipy.io.loadmat(file_path)
#         data = mat['data']  # shape: (64, 1500, 40, 6)

#         data = np.transpose(data, (2, 3, 0, 1))  # (40, 6, 64, 1500)

#         segments = []
#         for i in range(3):
#             seg = data[:, :, :, i*500:(i+1)*500]  # (40, 6, 64, 500)
#             segments.append(seg)

#         data_split = np.concatenate(segments, axis=1)  # (40, 18, 64, 500)
#         return data_split

#     def __len__(self):
#         return self.eeg_data.shape[0]  # total segments

#     def __getitem__(self, idx):
#         sample = self.eeg_data[idx]  # (64, 101)
#         return {
#             "src": torch.from_numpy(sample),
#             "montage": self.channel_loc,
#         }
class TshingHwa_Dataset(Dataset):
    def __init__(self, npy_file_list, sampling_rate=250):
        """
        Args:
            eeg_folder (str): Path to the folder containing .mat EEG files.
            sampling_rate (int): EEG sampling rate, default = 250Hz.
            save_path (str): Optional path to save the merged .npy dataset.
        """
        self.sampling_rate = sampling_rate
        self.channel_loc = [
            "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
            "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
            "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5",
            "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"
        ]
        self.data = []
        for file in npy_file_list:
            eeg = np.load(file)  # shape: (N_i, C, T)
            self.data.append(eeg)
        self.data = np.concatenate(self.data, axis=0)  # shape: (N_total, C, T)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        magnitude = self.data[idx]  # shape: (C, T)
        mean = np.mean(magnitude, axis=(0, 1), keepdims=True)
        std = np.std(magnitude, axis=(0, 1), keepdims=True)
        src_ = (magnitude - mean) / (std)
        return {
            "src": torch.tensor(src_, dtype=torch.float32),  # shape: (C, F)
            "montage": self.channel_loc,
        }

class EEGROI_fft_Dataset(Dataset):
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

        self.eeg_power_data = []  # Will store tuples of (ROI segment, EEG segment)
        self.roi_power_data = []
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
            roi_path = os.path.join(self.roi_folder, f"{subject}_roi.npy")
            eeg_path = os.path.join(self.roi_folder, f"{subject}_eeg.npy")

            # 讀取 ROI & EEG
            roi_data = np.load(roi_path)
            eeg_data = np.load(eeg_path)

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

        print(f"EEG Power shape: {self.eeg_power_data[0].shape}")
        print(f"Source Power shape: {self.roi_power_data[0].shape}")


    def _process_subject_data(self, roi_data, eeg_data, subject_name):
        """Segments, computes EEG power spectrum, and predicts ROI power spectrum for a single subject."""

        segment_len = roi_data.shape[0]
        nan_count = 0
        rand_count = 0

        for start_idx in range(0, segment_len):
            if start_idx not in self.segment_index[subject_name]:
                # if random.random() < 1/3:
                #     continue

                eeg_segment_start = 256 * start_idx
                eeg_segment_end = 256 * (start_idx + 2)

                eeg = eeg_data[:, eeg_segment_start:eeg_segment_end]

                # EEG Resampling
                # 更快的批次 resample 方法
                EEG = resample(eeg, 200, axis=1)  # 直接沿時間軸取樣到 200 點

                # Convert to Torch Tensor
                EEG = torch.tensor(EEG, dtype=torch.float32)

                ROI = torch.tensor(roi_data[start_idx, :, :], dtype=torch.float32).squeeze(0).transpose(0, 1)

                # FFT to get Power Spectrum
                # FFT and Extract 0~50Hz
                eeg_fft = torch.fft.rfft(EEG, dim=1)[:, :100]  # Keep only 0~50Hz
                roi_fft = torch.fft.rfft(ROI, dim=1)[:, :100]  # Keep only 0~50Hz

                eeg_power = torch.abs(eeg_fft) ** 2  # Compute Power Spectrum
                roi_power = torch.abs(roi_fft) ** 2  # Compute Power Spectrum

                # Normalize Power Spectrum (Z-score)
                eeg_mean = torch.mean(eeg_power, dim=(0, 1), keepdim=True)
                eeg_std = torch.std(eeg_power, dim=(0, 1), keepdim=True)
                eeg_power = (eeg_power - eeg_mean) / (eeg_std)

                roi_mean = torch.mean(roi_power, dim=(0, 1), keepdim=True)
                roi_std = torch.std(roi_power, dim=(0, 1), keepdim=True)
                roi_power = (roi_power - roi_mean) / (roi_std)

                assert eeg_power.shape == (30, 100), f"Unexpected EEG Power shape: {eeg_power.shape}"
                assert roi_power.shape == (204, 100), f"Unexpected ROI Power shape: {roi_power.shape}"

                if not (torch.isnan(eeg_power).any() or torch.isinf(eeg_power).any()):
                    if not (torch.isnan(roi_power).any() or torch.isinf(roi_power).any()):
                        # Store Power Spectrum as training data
                        self.eeg_power_data.append(eeg_power)
                        self.roi_power_data.append(roi_power)
                    else:
                        nan_count += 1
                else:
                    nan_count += 1
                rand_count += 1

        # print(f"Total NaN count for subject: {nan_count}")
        # print(f"Total Random count for subject: {rand_count}")

    def __len__(self):
        return len(self.eeg_power_data)

    def __getitem__(self, idx):
        return {
            "src": self.eeg_power_data[idx], 
            "label": self.roi_power_data[idx]
        }
        

class EEGROI_Power_Dataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            roi_folder (str): Path to the folder containing ROI .set files.
            eeg_folder (str): Path to the folder containing EEG .set files.
            overlap (float): Fraction of overlap between consecutive windows (0 <= overlap < 1).
            window_size (int): Number of samples in each window.
        """
        self.data_path = data_path

        self.eeg_power_data = []  # Will store tuples of (ROI segment, EEG segment)
        self.roi_power_data = []
        self._prepare_dataset()

    # def _prepare_dataset(self):

        
    #     # 讀取全部資料
    #     all_eeg_data = np.load(os.path.join(self.data_path, "all_eeg_data.npy"), allow_pickle=True)
    #     all_source_data = np.load(os.path.join(self.data_path, "all_source_data.npy"), allow_pickle=True)

    #     EEG = torch.tensor(np.stack(all_eeg_data), dtype=torch.float32)  # (N, 64, 51)
    #     ROI = torch.tensor(np.stack(all_source_data), dtype=torch.float32)  # (N, 204, 51)

    #     # 建立 channel map (只做一次)
    #     ch_idx_map = get_channel_index()

    #     # 選出 32 channels
    #     EEG = EEG[:, ch_idx_map, :]  # (N, 30, 51)

    #     # 做 Z-score normalization
    #     eeg_mean = EEG.mean(dim=2, keepdim=True)
    #     eeg_std = EEG.std(dim=2, keepdim=True)
    #     eeg_power = (EEG - eeg_mean) / (eeg_std + 1e-10)

    #     roi_mean = ROI.mean(dim=2, keepdim=True)
    #     roi_std = ROI.std(dim=2, keepdim=True)
    #     roi_power = (ROI - roi_mean) / (roi_std + 1e-10)

    #     # 篩掉含 NaN 的 sample
    #     valid_mask = ~(torch.isnan(eeg_power).any(dim=(1, 2)) | torch.isinf(eeg_power).any(dim=(1, 2)) |
    #                 torch.isnan(roi_power).any(dim=(1, 2)) | torch.isinf(roi_power).any(dim=(1, 2)))


    #     self.eeg_power_data = eeg_power[valid_mask]
    #     self.roi_power_data = roi_power[valid_mask]

    #     print(f"EEG Power shape: {self.eeg_power_data.shape}")
    #     print(f"Source Power shape: {self.roi_power_data.shape}")

    def _prepare_dataset(self):
        # 🔍 找出所有 .npy 檔案，並依照名稱排序
        eeg_files = sorted(glob.glob(os.path.join(self.data_path, "*all_eeg_data_*.npy")))
        src_files = sorted(glob.glob(os.path.join(self.data_path, "*all_source_data_*.npy")))

        assert len(eeg_files) == len(src_files), "EEG 與 Source 資料數量不一致"
        assert len(eeg_files) > 0, "no data"

        all_eeg_data = []
        all_source_data = []

        # 📦 依序讀取所有檔案並收集
        for eeg_file, src_file in zip(eeg_files, src_files):
            eeg = np.load(eeg_file, allow_pickle=True)
            src = np.load(src_file, allow_pickle=True)
            # print(f"{eeg_file}: {eeg.shape} - {src.shape}")
            all_eeg_data.extend(eeg)
            all_source_data.extend(src)

        # ➕ Stack 成 Tensor
        EEG = torch.tensor(np.stack(all_eeg_data), dtype=torch.float32)  # (N, 64, T)
        ROI = torch.tensor(np.stack(all_source_data), dtype=torch.float32)  # (N, 204, T)

        # Channel map: 選 32 channels
        ch_idx_map = get_channel_index()
        EEG = EEG[:, ch_idx_map, :]  # (N, 32, T)

        # Z-score normalization
        eeg_mean = EEG.mean(dim=(1, 2), keepdim=True)
        eeg_std = EEG.std(dim=(1, 2), keepdim=True)
        eeg_power = (EEG - eeg_mean) / (eeg_std)

        roi_mean = ROI.mean(dim=(1, 2), keepdim=True)
        roi_std = ROI.std(dim=(1, 2), keepdim=True)
        roi_power = (ROI - roi_mean) / (roi_std)
        # print(f"eeg mean:{eeg_mean}, eeg std:{eeg_std}\nsource mea:{roi_mean}, source std:{roi_std}")

        # ✅ 移除 NaN / inf
        valid_mask = ~(torch.isnan(eeg_power).any(dim=(1, 2)) | torch.isinf(eeg_power).any(dim=(1, 2)) |
                    torch.isnan(roi_power).any(dim=(1, 2)) | torch.isinf(roi_power).any(dim=(1, 2)))

        self.eeg_power_data = eeg_power[valid_mask]
        self.roi_power_data = roi_power[valid_mask]

        print(f"EEG Power shape: {self.eeg_power_data.shape}")
        print(f"Source Power shape: {self.roi_power_data.shape}")

    def __len__(self):
        return len(self.eeg_power_data)

    def __getitem__(self, idx):
        return {
            "src": self.eeg_power_data[idx], 
            "label": self.roi_power_data[idx]
        }
    
class EEGROI_Merge_Dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.cumulative_lengths = [0]
        for dataset in dataset_list:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
        self.len_ = self.cumulative_lengths[-1]

    def map_idx(self, idx):
        list_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        item_idx = idx - self.cumulative_lengths[list_idx]
        return list_idx, item_idx

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        list_idx, item_idx = self.map_idx(idx)
        return self.dataset_list[list_idx][item_idx]

def get_channel_index():
    # 建立 channel map (只做一次)
    ch_names_64 = mne.io.read_raw('G:\\共用雲端硬碟\\CNElab_陳昱祺\\source localization\\simulate_data\\test_raw.fif', verbose='Warning').resample(100).info['ch_names']
    ch_names_32 = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
                'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 
                'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
    ch_idx_map = [ch_names_64.index(ch) for ch in ch_names_32]

    return ch_idx_map

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
    
# class RandonMaskDataCollator:
#     def __init__(self, mask_prob=0.15, tgt_type="simple"):
#         self.mask_prob = mask_prob

#     def __call__(self, features):
#         inputs = torch.stack([f["src"] for f in features])  # (batch, 204, 100)
#         labels = torch.stack([f["label"] for f in features])

#         tgt = labels.clone()
#         batch_size, seq_len, feature_dim = tgt.shape

#         mask = torch.rand(batch_size, seq_len) < self.mask_prob  # (batch, 204)
#         # print(f"Mask: {mask[0]}")

#         # 準備三種mask策略
#         mask_rand = torch.rand(batch_size, seq_len)

#         # 80% 變0
#         zero_mask = (mask_rand < 0.8) & mask
#         # print(f"Zero Mask: {zero_mask[0]}")
#         tgt[zero_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = 0.0

#         # 10% 變 random noise
#         random_mask = (mask_rand >= 0.8) & (mask_rand < 0.9) & mask
#         # print(f"Random Mask: {random_mask[0]}")

#         noise = torch.randn_like(tgt)
#         tgt[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = noise[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)]

#         # print(f"tgt: {tgt[0]}")

#         # 10% 保持原樣（不需要動）

#         # tgt_type 是否要加上EEG在tgt上。
        

#         return_dict = True
#         return {
#             "src": inputs,
#             "tgt": tgt,
#             "tgt_mask": None, 
#             "src_mask": None,
#             "tgt_token_mask": mask, # 提供 mask 給 loss function使用
#             "labels": labels,
#             "return_dict": return_dict
#         }
    


class RandonMaskDataCollator:
    def __init__(self, mask_prob=0.15, tgt_type="simple"):
        self.mask_prob = mask_prob
        self.tgt_type = tgt_type

    def __call__(self, features):
        inputs = torch.stack([f["src"] for f in features])  # (batch, 30, 100)
        labels = torch.stack([f["label"] for f in features])  # (batch, 204, 100)

        tgt = labels.clone()
        batch_size, seq_len, feature_dim = tgt.shape  # (batch, 204, 100)

        # 產生 mask
        mask = torch.rand(batch_size, seq_len) < self.mask_prob  # (batch, 204)

        # Mask策略
        mask_rand = torch.rand(batch_size, seq_len)

        zero_mask = (mask_rand < 0.8) & mask
        random_mask = (mask_rand >= 0.8) & (mask_rand < 0.9) & mask

        # Apply masking to tgt
        tgt[zero_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = 0.0
        noise = torch.randn_like(tgt)
        tgt[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = noise[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)]

        # ➡️ 加入 EEG data 到 tgt
        if self.tgt_type == "add_eeg":
            # 1. 把 src 貼到 tgt 後面 (在 token dimension)
            tgt = torch.cat([tgt, inputs], dim=1)  # (batch, 204+30, 100)


        return_dict = True
        return {
            "src": inputs,
            "tgt": tgt,
            "tgt_mask": None,
            "src_mask": None,
            "tgt_token_mask": mask,  # (batch, 204+30) if add_eeg
            "labels": labels,        # 原始 label 還是204個
            "return_dict": return_dict
        }

class RandonMaskEEGDataCollator:
    def __init__(self, mask_prob=0.15):
        self.mask_prob = mask_prob

    def __call__(self, features):
        inputs = torch.stack([f["src"] for f in features])  # (batch, 30, 200)
        labels = inputs.clone()

        batch_size, seq_len, feature_dim = inputs.shape  # (batch, 30, 100)

        # 產生 mask
        mask = torch.rand(batch_size, seq_len) < self.mask_prob  # (batch, 30)

        # Mask策略
        mask_rand = torch.rand(batch_size, seq_len)

        zero_mask = (mask_rand < 0.8) & mask
        random_mask = (mask_rand >= 0.8) & (mask_rand < 0.9) & mask

        # Apply masking to tgt
        inputs[zero_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = 0.0
        noise = torch.randn_like(inputs)
        inputs[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = noise[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)]


        return_dict = True
        return {
            "src": inputs,
            "tgt": inputs,
            "tgt_mask": None,
            "src_mask": None,
            "tgt_token_mask": mask,  
            "labels": labels,        
            "return_dict": return_dict
        }
    
class RandonMaskShuffleEEGDataCollator:
    def __init__(self, mask_prob=0.15, montage_kind="standard_1020"):
        self.mask_prob = mask_prob
        self.mne_montage = mne.channels.make_standard_montage(montage_kind)

        # 建立 channel_name -> (x, y, z) dict
        self.channel_pos_map = {
            ch_name: torch.tensor(pos, dtype=torch.float32) / torch.norm(torch.tensor(pos, dtype=torch.float32))
            for ch_name, pos in self.mne_montage.get_positions()['ch_pos'].items()
        }
    def __call__(self, features):
        src_list = []
        label_list = []
        pos_list = []
        montage_name_list = []
        lengths = []
        

        max_len = max(f["src"].shape[0] for f in features)  # 最大 channel 數
        feature_dim = features[0]["src"].shape[1]

        for f in features:
            src = f["src"]  # (channel_num, feature_dim)
            montage_names = f["montage"]
            channel_num = src.shape[0]

            # shuffle channel
            perm = torch.randperm(channel_num)
            shuffled_src = src[perm]
            shuffled_names = [montage_names[i] for i in perm.tolist()]

            # get 3D position
            pos_3d = torch.stack([
                self.channel_pos_map.get(name, torch.zeros(3)) for name in shuffled_names
            ])

            # padding
            pad_len = max_len - channel_num
            if pad_len > 0:
                pad_src = torch.zeros((pad_len, feature_dim))
                pad_pos = torch.zeros((pad_len, 3))
                shuffled_src = torch.cat([shuffled_src, pad_src], dim=0)
                pos_3d = torch.cat([pos_3d, pad_pos], dim=0)

            src_list.append(shuffled_src)
            label_list.append(shuffled_src.clone())
            pos_list.append(pos_3d)
            lengths.append(channel_num)
            montage_name_list.append(shuffled_names)

        inputs = torch.stack(src_list)        # (B, max_len, F)
        labels = torch.stack(label_list)      # (B, max_len, F)
        positions = torch.stack(pos_list)     # (B, max_len, 3)
        # montage = torch.stack(montage_name_list)

        batch_size, seq_len, feature_dim = inputs.shape

        # valid mask: 哪些 channel 是有效的
        valid_mask = torch.arange(seq_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)  # (B, C)

        # --- 隨機 Mask 機制 ---
        rand = torch.rand(batch_size, seq_len)
        tgt_token_mask = (rand < self.mask_prob) & valid_mask  # 只 mask 有效 channel

        mask_rand = torch.rand(batch_size, seq_len)
        zero_mask = (mask_rand < 0.8) & tgt_token_mask
        random_mask = (mask_rand >= 0.8) & (mask_rand < 0.9) & tgt_token_mask

        inputs[zero_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = 0.0
        noise = torch.randn_like(inputs)
        inputs[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = noise[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)]

        # --- Attention mask (B, C, C) ---
        attn_mask = (valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2))  # True = masked

        return {
            "src": inputs,
            "tgt": inputs,
            "labels": labels,
            "tgt_token_mask": tgt_token_mask,   # 哪些 channel 是 masked
            "valid_channel_mask": valid_mask,   # 哪些 channel 是有效的
            "src_mask": attn_mask,
            "tgt_mask": attn_mask,
            "montage": montage_name_list,  # (B, C, 3)
            "position": positions,
            "return_dict": True
        }



    # def __call__(self, features):
    #     src_list = []
    #     label_list = []
    #     pos_list = []

    #     for f in features:
    #         src = f["src"]  # (channel_num, 101)
    #         montage_names = f["montage"]  # list of channel names (length = channel_num)

    #         # 隨機打亂 channel 順序
    #         perm = torch.randperm(len(montage_names))
    #         shuffled_src = src[perm]
    #         shuffled_names = [montage_names[i] for i in perm.tolist()]

    #         # 快速查表，未找到的填 0
    #         pos_3d = torch.stack([
    #             self.channel_pos_map.get(name, torch.zeros(3)) for name in shuffled_names
    #         ])  # shape: (channel_num, 3)

    #         for name in shuffled_names:
    #             print(name, self.channel_pos_map.get(name))

    #         src_list.append(shuffled_src)
    #         label_list.append(shuffled_src.clone())
    #         pos_list.append(pos_3d)

    #     inputs = torch.stack(src_list)       # (batch, channel_num, 101)
    #     labels = torch.stack(label_list)     # (batch, channel_num, 101)
    #     positions = torch.stack(pos_list)    # (batch, channel_num, 3)

    #     batch_size, seq_len, feature_dim = inputs.shape

    #     # 產生 mask
    #     mask = torch.rand(batch_size, seq_len) < self.mask_prob

    #     # Mask策略
    #     mask_rand = torch.rand(batch_size, seq_len)
    #     zero_mask = (mask_rand < 0.8) & mask
    #     random_mask = (mask_rand >= 0.8) & (mask_rand < 0.9) & mask

    #     inputs[zero_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = 0.0
    #     noise = torch.randn_like(inputs)
    #     inputs[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)] = noise[random_mask.unsqueeze(-1).expand(-1, -1, feature_dim)]

    #     return {
    #         "src": inputs,
    #         "tgt": inputs,
    #         "labels": labels,
    #         "tgt_token_mask": mask,
    #         "src_mask": None,
    #         "tgt_mask": None,
    #         "montage": positions,  # (batch, channel_num, 3)
    #         "return_dict": True
    #     }
