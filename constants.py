# Path where mne library will download the data
DATA_PATH = "datasets"

# Id of test subjects from the MNE dataset to be used for training
SUBJECT_ID_TRAIN = list(range(1, 1 + 1))

# Run numbers that correspond to imagining the action of imagining closing hands or feet
RUN_IMAGINE_HANDS_FEET = [6, 10, 14]

# Run numbers that correspond to imagining the action of executing closing hands or feet
RUN_EXECUTE_HANDS_FEET = [5, 9, 13] 

# TODO: more actions

# get event mapping for specified actions
def get_event_mapping(actions):
	if actions == RUN_EXECUTE_HANDS_FEET:
		return {1:'rest', 2: 'execute/feet', 3: 'execute/hands'}
	if actions == RUN_IMAGINE_HANDS_FEET:
		return {1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
	raise ValueError("No actions to match for get_event_mapping")