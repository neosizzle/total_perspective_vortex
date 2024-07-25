import logging
import matplotlib.pyplot as plt
import mne
import constants
import numpy as np

def plot_for(id, enable, data=None):
	if not enable:
		return
	match id:
		case 1:
			edg = data[0]
			idx = data[1]
			title = f'diagrams/rawedg_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = edg.plot(n_channels=64, duration=120, show_scrollbars=False, show_scalebars=False, scalings=dict(eeg=0.0001))
			fig.savefig(title)
			plt.close(fig)

		case 2:
			filtered_edg = data[0]
			idx = data[1]
			title = f'diagrams/filtered_psd_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = filtered_edg.compute_psd(fmax=80).plot(picks="data", exclude="bads", amplitude=False)
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/filtered_wavelet_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = filtered_edg.compute_tfr(method="morlet", freqs=list(range(20, 30)), tmin=0.0, tmax=1.0).plot_joint()
			fig.savefig(title)
			plt.close(fig)

		case 3:
			reject_log = data[0]
			idx = data[1]
			title = f'diagrams/ica_reject_log_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig, ax = plt.subplots(figsize=[15, 5])
			reject_log.plot('horizontal', ax=ax, aspect='auto')
			fig.savefig(title)
			plt.close(fig)

		case 4:
			ica = data[0]
			epochs_ica = data[1]
			idx = data[2]

			logging.info(f"Plotting for ica components and properties")
			components_fig = ica.plot_components()
			if isinstance(components_fig, list):
				for fig_idx, fig in enumerate(components_fig):
					fig.savefig(f'diagrams/ica_components_{idx}_{fig_idx}.png')
			else :
				components_fig.savefig(f'diagrams/ica_components_{idx}.png')

			properties_fig = ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': 30})
			for fig_idx, fig in enumerate(properties_fig):
				fig.savefig(f'diagrams/ica_properties_{idx}_{fig_idx}.png')
		case 5:
				events = data[0]
				raw_filt = data[1]
				action = data[2]
				idx = data[3]
				title = f'diagrams/events_{idx}.png'

				fig, ax = plt.subplots(figsize=[15, 5])
				logging.info(f"Plotting for {title}")
				mne.viz.plot_events(events, raw_filt.info['sfreq'],  
									event_id={v: k for k, v in constants.get_event_mapping(action).items()},                    
									axes=ax)
				
				fig.savefig(title)
				plt.close(fig)

		case 6:
				epochs = data[0]
				action = data[1]
				idx = data[2]

				title = f'diagrams/epoch_{idx}.png'
				logging.info(f"Plotting for {title}")
				fig = epochs.plot(scalings=dict(eeg=0.0001))
				fig.savefig(title)
				plt.close(fig)

				title = f'diagrams/epoch_avg_{idx}.png'
				logging.info(f"Plotting for {title}")
				fig = epochs.average().plot(scalings=dict(eeg=0.0001))
				fig.savefig(title)
				plt.close(fig)

				title = f'diagrams/epoch_avg_events_{idx}.png'
				logging.info(f"Plotting for {title}")
				event_mapping = constants.get_event_mapping(action)
				fig = epochs[event_mapping[2], event_mapping[3]].average().plot(scalings=dict(eeg=0.0001))
				fig.savefig(title)
				plt.close(fig)

		case 7:
			epochs_postica = data[0]
			action = data[1]
			idx = data[2]

			title = f'diagrams/epochs_postica_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = epochs_postica.plot(scalings=dict(eeg=0.0001))
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/epochs_postica_avg_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = epochs_postica.average().plot(scalings=dict(eeg=0.0001))
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/epochs_postica_avg_events_{idx}.png'
			logging.info(f"Plotting for {title}")
			event_mapping = constants.get_event_mapping(action)
			fig = epochs_postica[event_mapping[2], event_mapping[3]].average().plot(scalings=dict(eeg=0.0001))
			fig.savefig(title)
			plt.close(fig)

		case 8:
			epochs = data[0]
			epochs_clean = data[1]
			tmax = data[2]
			action = data[3]
			idx = data[4]
			event_mapping = constants.get_event_mapping(action)
			
			title = f'diagrams/epochs_compare_filter_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig, ax = plt.subplots(1, 2, figsize=[12, 3])
			epochs[event_mapping[2], event_mapping[3]].average().plot(axes=ax[0], show=False)
			epochs_clean[event_mapping[2], event_mapping[3]].average().plot(axes=ax[1])
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/{event_mapping[2].replace("/", "_")}_topmap_{idx}.png'
			logging.info(f"Plotting for {title}")
			times = np.arange(0, tmax, 0.1)
			fig = epochs_clean[event_mapping[2]].average().plot_topomap(times=times, average=0.050)
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/{event_mapping[3].replace("/", "_")}_topmap_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = epochs_clean[event_mapping[3]].average().plot_topomap(times=times, average=0.050)
			fig.savefig(title)
			plt.close(fig)

		case 9:
			filtered_edg = data[0]
			idx = data[1]
			title = f'diagrams/final_psd_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = filtered_edg.compute_psd(fmax=80).plot(picks="data", exclude="bads", amplitude=False)
			fig.savefig(title)
			plt.close(fig)

			title = f'diagrams/final_wavelet_{idx}.png'
			logging.info(f"Plotting for {title}")
			fig = filtered_edg.compute_tfr(method="morlet", freqs=list(range(20, 30))).average().plot_joint()
			fig.savefig(title)
			plt.close(fig)

		case _:
			logging.error(f"id {id} not matched in plot_for")