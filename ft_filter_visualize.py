import argparse
import coloredlogs, logging
import traceback
import constants
import matplotlib.pyplot as plt
import pandas as pd
import ft_plotter

import mne
from mne.datasets import eegbci
mne.set_log_level('WARNING')

from autoreject import AutoReject


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-n', '--subjects', help="Speficy number of subjects to load data from", type=int, default=1)
	parser.add_argument('-i', '--init_subject', help="Speficy the first subject number to start loading from", type=int, default=1)
	parser.add_argument('-a', '--action', help="Speficy type of action dataset to load", type=str, default='hands_feet', choices=['hands_feet', 'left_right'])
	parser.add_argument('-d', '--diagram',  help="Generate diagrams for all filtering steps", action='store_true')
	parser.add_argument('-o', '--output',  help="Path to store preprocessed data",  type=str, default="data.csv") 
	return parser.parse_args()

# Run artifact correction with ICA analysis and filtering
# ICA -  capturing features of the data that explain the most variance
# https://neuraldatascience.io/7-eeg/erp_artifacts.html#filter-the-data-for-ica
def get_ica(raw_data, idx, enable):
	res = raw_data.copy().filter(1, 30)
	
	# Break raw data into 1s epochs
	tstep = 1.0
	events_ica = mne.make_fixed_length_events(res, duration=tstep)
	epochs_ica = mne.Epochs(res, events_ica,
							tmin=0.0, tmax=tstep,
							baseline=None,
                        preload=True)
	
	# initialize autoreject agent which rejects signals which are
	# outright obvious artifacts. This is needed for ICA
	ar = AutoReject(n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs_ica.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False
                )
	
	# mark the data which should be rejected
	ar.fit(epochs_ica)
	reject_log = ar.get_reject_log(epochs_ica)

	ft_plotter.plot_for(3, enable, [reject_log, idx])

	# Fit the ICA to the original data (ICA) using marks generated
	# ICA parameters
	random_state = 42   # ensures ICA is reproducible each time it's run
	ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

	# Fit ICA - generate ICA to actually remove unrelated signals
	ica = mne.preprocessing.ICA(n_components=ica_n_components,
								random_state=random_state,
								)
	ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)
	ft_plotter.plot_for(4, enable, [ica, epochs_ica, idx])

	# iteratively adjust z_threshold until we mark at least 2 max_ic components as EOG to be removed 
	ica.exclude = []
	num_excl = 0
	max_ic = 2
	z_thresh = 3.5
	z_step = .05

	while num_excl < max_ic:
		eog_indices, eog_scores = ica.find_bads_eog(epochs_ica,
													ch_name=['Fp1', 'Fp2', 'F7', 'F8'], 
													threshold=z_thresh
													)
		num_excl = len(eog_indices)
		z_thresh -= z_step

	# assign the bad EOG components to the ICA.exclude attribute so they can be removed later
	ica.exclude = eog_indices

	return ica

# filter data from noises. Noises usually occur outside the 0.1 - 30hz frequency channels, so we can remove signals 
# that originate from those sources (artifacts)
# https://neuraldatascience.io/7-eeg/erp_filtering.html
def filter_data(raw_data):
	res = raw_data.copy().filter(0.1, 30)
	return res

# get locations of all sensors to plot them
# since we have 64 channels, we need a montage with 64 sensors
# https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
def get_sensors():
	return mne.channels.make_standard_montage("standard_1005")	

# downloads data and deserializes them based on the arguments provided
# https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py
def get_data(subject_ids, actions, sensors):
	res = []
	for subject_id in subject_ids:
		# downloads data based on subject id and actions required to data_path
		loaded_path = eegbci.load_data(subject_id, actions, constants.DATA_PATH)
		for path_item in loaded_path:
			# reads the downloaded data, enable preload to decrease load time in the expense 
			# of extra memory usage
			raw_rdf = mne.io.read_raw_edf(path_item, preload=True)

			# get events from data (when and what happened) and set the events locally in memory
			# T0 corresponds to rest
			# T1 corresponds to onset of motion (real or imagined) of left fist / both fists based on run number
			# T2 corresponds to onset of motion (real or imagined) of right fist / both feet based on run number
			events, _ = mne.events_from_annotations(raw_rdf, event_id=dict(T0=1,T1=2,T2=3))
			mapping = constants.get_event_mapping(actions)
			annot_from_events = mne.annotations_from_events(
				events=events, event_desc=mapping, sfreq=raw_rdf.info['sfreq'],
				orig_time=raw_rdf.info['meas_date'])
			raw_rdf.set_annotations(annot_from_events)

			# set sensor locations
			eegbci.standardize(raw_rdf)  # set channel names
			raw_rdf.set_montage(sensors)

			res.append(raw_rdf)

	return res

def apply_ica_and_epoch(ica, raw_filt, action, idx, enable):
	events, events_dict = mne.events_from_annotations(raw_filt)
	event_id = {v: k for k, v in constants.get_event_mapping(action).items()}
	ft_plotter.plot_for(5, enable, [events, raw_filt, action, idx])

	# Epoching settings
	tmin =  -.100  # start of each epoch (in sec)
	tmax =  1.000  # end of each epoch (in sec)
	baseline = (None, 0)

	# Create epochs
	epochs = mne.Epochs(raw_filt,
						events, event_id,
						tmin, tmax,
						baseline=baseline, 
						preload=True
					) 
	ft_plotter.plot_for(6, enable, [epochs, action, idx])

	# Apply ICA, which contain exclusion rules for uneanted EOG channels
	epochs_postica = ica.apply(epochs.copy())
	ft_plotter.plot_for(7, enable, [epochs_postica, action, idx])

	# Apply autoreject another time to filter out significant non-ocular noise
	ar = AutoReject(n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs_postica.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False
                )
	epochs_clean, reject_log_clean = ar.fit_transform(epochs_postica, return_log=True)
	ft_plotter.plot_for(8, enable, [epochs, epochs_clean, tmax, action, idx])

	return epochs_clean

def main():
	coloredlogs.install()
	args = get_args()

	enable_diagram = args.diagram
	mne.set_config('MNE_BROWSE_RAW_SIZE', '25,15')
	subject_ids = list(range(args.init_subject, args.subjects + args.init_subject))
	sensors = get_sensors()
	action = constants.RUN_IMAGINE_HANDS_FEET if args.action == "hands_feet" else constants.RUN_IMAGINE_LEFT_RIGHT
	
	im_data = get_data(subject_ids, action, sensors)
	event_mapping = constants.get_event_mapping(action)
	all_data = None
	for idx, edg in enumerate(im_data):
		ft_plotter.plot_for(1, enable_diagram, [edg, idx])

		filtered_edg = filter_data(edg)
		ft_plotter.plot_for(2, enable_diagram, [filtered_edg, idx])

		ica_filter = get_ica(edg, idx, enable_diagram)
		applied_ica_epochs = apply_ica_and_epoch(ica_filter, filtered_edg, action, idx, enable_diagram)
		df = applied_ica_epochs.to_data_frame()
		logging.info(f"EDG data at idx {idx} done")
		if all_data is None :
			all_data = df
		else :
			all_data = pd.concat([all_data, df])
	
	print(f"df {all_data}")
	all_data.to_csv(args.output)

main()