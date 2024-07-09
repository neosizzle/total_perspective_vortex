# mne import
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap, Xdawn)
from mne.time_frequency import AverageTFR
from mne.channels import make_standard_montage
from mne.decoding import LinearModel, Vectorizer, get_coef, Scaler, CSP, SPoC, UnsupervisedSpatialFilter
mne.set_log_level('WARNING')

subject = [1] # 1, 4
run_execution = [5, 9, 13]
run_imagery = [6, 10, 14]

raw_files = []

for person_number in subject:
	for j in run_imagery:
		raw_files_imagery = [read_raw_edf(f, preload=True, stim_channel='auto') for f in eegbci.load_data(person_number, j)]
		print(len(raw_files_imagery))
		raw_imagery = concatenate_raws(raw_files_imagery)
		# raw_imagery = raw_files_imagery 

		events, _ = mne.events_from_annotations(raw_imagery, event_id=dict(T0=1,T1=2,T2=3))
		mapping = {1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
		annot_from_events = mne.annotations_from_events(
			events=events, event_desc=mapping, sfreq=raw_imagery.info['sfreq'],
			orig_time=raw_imagery.info['meas_date'])
		raw_imagery.set_annotations(annot_from_events)
        
        
		raw_files.append(raw_imagery)

print(raw_files)