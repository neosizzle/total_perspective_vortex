import constants

import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))

import mne
from mne.datasets import eegbci

from autoreject import AutoReject
import numpy as np

mne.set_log_level('WARNING')

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

# get locations of all sensors to plot them
# since we have 64 channels, we need a montage with 64 sensors
# https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
def get_sensors():
	return mne.channels.make_standard_montage("standard_1005")	

# filter data from noises. Noises usually occur outside the 0.1 - 30hz frequency channels, so we can remove signals 
# that originate from those sources (artifacts)
# https://neuraldatascience.io/7-eeg/erp_filtering.html
def filter_data(raw_data):
	res = raw_data.copy().filter(0.1, 30)
	return res

# Run artifact correction with ICA analysis and filtering
# ICA -  capturing features of the data that explain the most variance
# https://neuraldatascience.io/7-eeg/erp_artifacts.html#filter-the-data-for-ica
def get_ica(raw_data, idx):
	res = raw_data.copy().filter(1, 30)
	
	# Break raw data into 1s epochs
	tstep = 1.0
	events_ica = mne.make_fixed_length_events(res, duration=tstep)
	epochs_ica = mne.Epochs(res, events_ica,
							tmin=0.0, tmax=tstep,
							baseline=None,
                        preload=True)
	# fig = epochs_ica.plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epoch_{idx}.png')
	
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

	# fig, ax = plt.subplots(figsize=[15, 5])
	# reject_log.plot('horizontal', ax=ax, aspect='auto')
	# fig.savefig(f'reject_log_{idx}.png')

	# Fit the ICA to the original data (ICA) using marks generated
	# ICA parameters
	random_state = 42   # ensures ICA is reproducible each time it's run
	ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

	# Fit ICA - generate ICA to actually remove unrelated signals
	ica = mne.preprocessing.ICA(n_components=ica_n_components,
								random_state=random_state,
								)
	ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)
	
	# components_fig = ica.plot_components()
	# if isinstance(components_fig, list):
	# 	for fig_idx, fig in enumerate(components_fig):
	# 		fig.savefig(f'ica_components_{idx}_{fig_idx}.png')
	# else :
	# 	components_fig.savefig(f'ica_components_{idx}.png')

	# properties_fig = ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': 30})
	# for fig_idx, fig in enumerate(properties_fig):
	# 	fig.savefig(f'ica_properties_{idx}_{fig_idx}.png')

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
	# print('Final z threshold = ' + str(round(z_thresh, 2)))

	return ica

def apply_ica_and_epoch(ica, raw_filt, action, idx):
	events, events_dict = mne.events_from_annotations(raw_filt)
	event_id = {v: k for k, v in constants.get_event_mapping(action).items()}
	# fig, ax = plt.subplots(figsize=[15, 5])

	# mne.viz.plot_events(events, raw_filt.info['sfreq'],  
	# 					event_id={v: k for k, v in constants.get_event_mapping(action).items()},                    
	# 					axes=ax)
	
	# fig.savefig(f'events_{idx}.png')

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

	# fig = epochs.plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epoch_{idx}.png')

	# fig = epochs.average().plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epoch_avg_{idx}.png')

	# fig = epochs['imagine/feet', 'imagine/hands'].average().plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epoch_avg_events_{idx}.png')

	# Apply ICA, which contain exclusion rules for uneanted EOG channels
	epochs_postica = ica.apply(epochs.copy())

	# fig = epochs_postica.plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epochs_postica_{idx}.png')

	# fig = epochs_postica.average().plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epochs_postica_avg_{idx}.png')

	# fig = epochs_postica['imagine/feet', 'imagine/hands'].average().plot(scalings=dict(eeg=0.0001))
	# fig.savefig(f'epochs_postica_avg_events_{idx}.png')

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

	# fig, ax = plt.subplots(1, 2, figsize=[12, 3])
	# epochs['imagine/feet', 'imagine/hands'].average().plot(axes=ax[0], show=False) # remember the semicolon prevents a duplicated plot
	# epochs_clean['imagine/feet', 'imagine/hands'].average().plot(axes=ax[1])
	# fig.savefig(f'epochs_compare_filter_{idx}.png')

	# times = np.arange(0, tmax, 0.1)

	# fig = epochs_clean['imagine/feet'].average().plot_topomap(times=times, average=0.050)
	# fig.savefig(f'im_feet_topmap_{idx}.png')

	# fig = epochs_clean['imagine/hands'].average().plot_topomap(times=times, average=0.050)
	# fig.savefig(f'im_hands_topmap_{idx}.png')

	return epochs_clean

def re_ref(epochs, idx):
	conditions = ['imagine/feet', 'imagine/hands']

	evokeds = {c:epochs[c].average() for c in conditions}
	evokeds_avgref = {c:evokeds[c].copy().set_eeg_reference(ref_channels='average') 
                  for c in evokeds.keys()
                  }

	# for c in evokeds.keys():
	# 	fig = evokeds[c].plot_joint(title=c)
	# 	# define the channels we want plots for
	# 	channels = ['Fz', 'Cz', 'Pz', 'Oz']

	# 	# create a figure with 4 subplots
	# 	fig, axes = plt.subplots(2, 2, figsize=(8, 8))
	# 	for chan_idx,chan in enumerate(channels):
	# 		mne.viz.plot_compare_evokeds(evokeds, 
	# 								picks=chan,
	# 								ylim={'eeg':(-200, 200)},
	# 								show_sensors='lower right',
	# 								legend='upper center',
	# 								axes=axes.reshape(-1)[chan_idx],
	# 								show=False
	# 								)
	# 	fig.savefig(f'evoked_{idx}_{c.replace("/", "_")}.png')

	for c in evokeds_avgref.keys():
		fig = evokeds_avgref[c].plot_joint(title=c)
		# define the channels we want plots for
		channels = ['Fz', 'Cz', 'Pz', 'Oz']

		# create a figure with 4 subplots
		fig, axes = plt.subplots(2, 2, figsize=(8, 8))
		for chan_idx,chan in enumerate(channels):
			mne.viz.plot_compare_evokeds(evokeds_avgref, 
									picks=chan,
									ylim={'eeg':(-200, 200)},
									show_sensors='lower right',
									legend='upper center',
									axes=axes.reshape(-1)[chan_idx],
									show=False
									)
		fig.savefig(f'evoked_avgref_{idx}_{c.replace("/", "_")}.png')

	# for c in evokeds.keys():
	# 	fig = evokeds[c].plot_joint(title=c)
	# 	fig.savefig(f'evoked_{idx}_{c.replace("/", "_")}.png')

def main():
	action = constants.RUN_IMAGINE_HANDS_FEET
	sensors = get_sensors()
	im_data = get_data(constants.SUBJECT_ID_TRAIN, action, sensors)

	mne.set_config('MNE_BROWSE_RAW_SIZE', '25,15')
	for idx, edg in enumerate(im_data):
		fig = edg.plot(n_channels=64, duration=120, show_scrollbars=False, show_scalebars=False, scalings=dict(eeg=0.0001))
		# fig.savefig(f'rawedg_{idx}.png')

		filtered_edg = filter_data(edg)
		# fig = filtered_edg.compute_psd(fmax=80).plot(picks="data", exclude="bads", amplitude=False)
		# fig.savefig(f'filtered_psd_{idx}.png')

		ica_filter = get_ica(edg, idx)

		applied_ica_epochs = apply_ica_and_epoch(ica_filter, filtered_edg, action, idx)

		re_ref(applied_ica_epochs, idx)
	# print(f'loaded {len(im_data)} files')
main()