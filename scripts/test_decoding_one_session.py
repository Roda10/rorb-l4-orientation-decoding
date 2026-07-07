from src.data_access import get_boc, get_eligible_experiments, load_session_data
from src.preprocessing import compute_trial_responses, get_stimulus_trials
from src.decoding import decode_orientation, get_confusion_matrix


boc = get_boc()

experiments = get_eligible_experiments(
    boc=boc,
    cre_lines=["Rorb-IRES2-Cre"],
    targeted_structures=["VISp", "VISal", "VISpm"],
    stimuli=["drifting_gratings"],
)

session_id = experiments.iloc[0]["id"]
print(f"Testing decoding on session: {session_id}")

data_set = load_session_data(boc, session_id)

timestamps, dff = data_set.get_dff_traces()
stim_table = data_set.get_stimulus_table("drifting_gratings")

activity, labels = compute_trial_responses(dff, stim_table)
activity_stim, labels_stim = get_stimulus_trials(activity, labels)

mean_acc, chance, fold_acc = decode_orientation(
    activity_stim,
    labels_stim,
    decoder_type="logistic_regression",
    n_splits=5,
    random_state=42,
)

cm, cm_labels = get_confusion_matrix(
    activity_stim,
    labels_stim,
    decoder_type="logistic_regression",
    n_splits=5,
    random_state=42,
)

print("Activity shape:", activity_stim.shape)
print("Labels:", sorted(set(labels_stim)))
print("Chance level:", chance)
print("Fold accuracies:", fold_acc)
print("Mean CV accuracy:", mean_acc)
print("Confusion matrix shape:", cm.shape)
print("Confusion matrix labels:", cm_labels)