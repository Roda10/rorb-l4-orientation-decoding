from src.data_access import get_boc, get_eligible_experiments, load_session_data
from src.preprocessing import compute_trial_responses, get_stimulus_trials
from config import SESSION_OF_INTEREST


boc = get_boc()

experiments = get_eligible_experiments(boc=boc)

# session_id = experiments.iloc[0]["id"]
session_id = SESSION_OF_INTEREST
print(f"Testing session: {session_id}")

data_set = load_session_data(boc, session_id)

timestamps, dff = data_set.get_dff_traces()
stim_table = data_set.get_stimulus_table("drifting_gratings")

activity, labels = compute_trial_responses(dff, stim_table)
activity_stim, labels_stim = get_stimulus_trials(activity, labels)

print("DFF shape:", dff.shape)
# print(dff)
print("Stim table shape:", stim_table.shape)
# print(stim_table)
print("Activity shape:", activity.shape)
print("Activity without blanks:", activity_stim.shape)
# print(activity_stim)
print("Unique labels:", sorted(set(labels_stim)))