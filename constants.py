# Path where mne library will download the data
DATA_PATH = "datasets"

# Run numbers that correspond to imagining the action of imagining closing hands or feet
RUN_IMAGINE_HANDS_FEET = [6, 10, 14]

# Run numbers that correspond to imagining the action of executing closing hands or feet
RUN_EXECUTE_HANDS_FEET = [5, 9, 13] 

# Run numbers that correspond to imagining the action of imagining closing lefr or right hand
RUN_IMAGINE_LEFT_RIGHT = [3, 7, 11]

# Run numbers that correspond to imagining the action of executing closing lefr or right hand
RUN_EXECUTE_LEFT_RIGHT = [4, 8, 12] 

# get event mapping for specified actions
def get_event_mapping(actions):
	if actions == RUN_EXECUTE_HANDS_FEET:
		return {1:'rest', 2: 'execute/feet', 3: 'execute/hands'}
	if actions == RUN_IMAGINE_HANDS_FEET:
		return {1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
	if actions == RUN_EXECUTE_LEFT_RIGHT:
		return {1:'rest', 2: 'execute/left', 3: 'execute/right'}
	if actions == RUN_IMAGINE_LEFT_RIGHT:
		return {1:'rest', 2: 'imagine/left', 3: 'imagine/right'}
	raise ValueError("No actions to match for get_event_mapping")