
import os
import numpy as np
import h5py
import mne
import json
import torch
import time
from torch.utils.data import Dataset


class EEGROIDataset(Dataset):
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
                    # print(roi_data.shape)

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
        for start_idx in range(0, time_len, 2):
            eeg_step = start_idx * 256
            eeg_segment = eeg_data[:, eeg_step:eeg_step+eeg_window_size]

            roi_step = start_idx * 100
            roi_segment = roi_data[:, :, roi_step:roi_step+roi_window_size]
            
            if roi_segment.shape[2] == roi_window_size:
                roi_segment_reshape = roi_segment.reshape(-1, roi_window_size) 
                # print(roi_segment_reshape.shape)
                self.roi_data.append(torch.tensor(roi_segment_reshape, dtype=torch.float16))
                self.eeg_data.append(torch.tensor(eeg_segment, dtype=torch.float16))
            else:
                break

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