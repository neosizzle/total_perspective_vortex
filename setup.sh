# venv
python3 -m venv env
source env/bin/asctivate

# download dataset 
mkdir dataset
cd dataset
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

# pip install stuff
pip install matplotlib mne