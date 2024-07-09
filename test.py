import mne
from mne.datasets import eegbci

def main():
	edf_raw = mne.io.read_raw_edf("/home/nszl/42cursus/total_perspective_vortex/dataset/physionet.org/files/eegmmidb/1.0.0/S100/S100R01.edf")	
	edf_raw_2 = mne.io.read_raw_edf("/home/nszl/42cursus/total_perspective_vortex/dataset/physionet.org/files/eegmmidb/1.0.0/S102/S102R01.edf")	
	print(f"{edf_raw}")
	print(f"{edf_raw_2}")
	# fig = edf_raw.compute_psd(fmax=70).plot()
	fig = edf_raw_2.plot(n_channels=2, duration=50)
	# fig = edf_raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
	fig.savefig("test.svg")
main()