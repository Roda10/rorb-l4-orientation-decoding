"""
data_access.py

Handles communication with the Allen Brain Observatory Visual Coding 2P dataset
through AllenSDK's BrainObservatoryCache.

This module focuses on data access and metadata filtering only.
It does not perform decoding or statistical analysis.
"""

import os
import platform
import sys
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def get_cache_dir():
    """
    Return the local path to the Allen Brain Observatory Visual Coding 2P data.

    The function tries to support:
    - AWS / Colab environments
    - macOS with an external TReND2026 drive
    - Windows with an external drive mounted as E:/
    - Linux using an environment variable or a default mounted path
    """
    platstring = platform.platform()

    if ("amzn" in platstring) or ("google.colab" in sys.modules):
        return "/data/allen-brain-observatory/visual-coding-2p"

    if ("Darwin" in platstring) or ("macOS" in platstring):
        data_root = "/Volumes/TReND2026/"
    elif "Windows" in platstring:
        data_root = "D:/"
    else:
        username = os.environ.get("USER", "USERNAME")
        data_root = os.environ.get(
            "BRAIN_OBSERVATORY_DATA_ROOT",
            f"/media/{username}/TReND2026/"
        )

    return os.path.join(
        data_root,
        "allen-brain-observatory",
        "visual-coding-2p"
    )


def get_boc(manifest_path=None):
    """
    Create the BrainObservatoryCache object.

    Parameters
    ----------
    manifest_path : str or None
        Path to the AllenSDK manifest.json file.
        If None, the path is inferred automatically.

    Returns
    -------
    BrainObservatoryCache
    """
    if manifest_path is None:
        manifest_path = os.path.join(get_cache_dir(), "manifest.json")

    print(f"Using manifest file: {manifest_path}")
    return BrainObservatoryCache(manifest_file=manifest_path)


def get_eligible_containers(boc, cre_lines, targeted_structures):
    """
    Find experiment containers matching Cre line and cortical areas.

    An experiment container groups multiple imaging sessions from the same
    field of view.

    Parameters
    ----------
    boc : BrainObservatoryCache
    cre_lines : list of str
    targeted_structures : list of str

    Returns
    -------
    pandas.DataFrame
        One row per eligible experiment container.
    """
    containers = boc.get_experiment_containers(
        cre_lines=cre_lines,
        targeted_structures=targeted_structures,
    )

    return pd.DataFrame(containers)


def get_eligible_experiments(boc, cre_lines, targeted_structures, stimuli):
    """
    Find ophys experiment sessions matching the project filters.

    Parameters
    ----------
    boc : BrainObservatoryCache
    cre_lines : list of str
    targeted_structures : list of str
    stimuli : list of str

    Returns
    -------
    pandas.DataFrame
        One row per eligible ophys experiment session.
    """
    experiments = boc.get_ophys_experiments(
        cre_lines=cre_lines,
        targeted_structures=targeted_structures,
        stimuli=stimuli,
    )

    return pd.DataFrame(experiments)


def load_session_data(boc, session_id):
    """
    Load one ophys experiment session.

    Parameters
    ----------
    boc : BrainObservatoryCache
    session_id : int
        Ophys experiment/session ID.

    Returns
    -------
    data_set
        AllenSDK data object giving access to dF/F traces,
        stimulus tables, cell specimen IDs, running speed, etc.
    """
    return boc.get_ophys_experiment_data(session_id)


def summarize_eligible_experiments(experiments):
    """
    Print a simple summary of eligible sessions.
    Useful for checking whether filtering worked.
    """
    print(f"\nFound {len(experiments)} eligible sessions\n")

    if experiments.empty:
        print("No eligible experiments found.")
        return

    print("Available columns:")
    print(experiments.columns.tolist())

    print("\nFirst rows:")
    print(experiments.head())

    if "targeted_structure" in experiments.columns:
        print("\nSessions per cortical region:")
        print(experiments["targeted_structure"].value_counts())

    if "imaging_depth" in experiments.columns:
        print("\nImaging depths:")
        print(experiments["imaging_depth"].value_counts().sort_index())

    if "ophys_container_id" in experiments.columns:
        print("\nNumber of unique containers:")
        print(experiments["ophys_container_id"].nunique())


if __name__ == "__main__":
    # Quick manual test for the project research question.

    boc = get_boc()

    experiments = get_eligible_experiments(
        boc=boc,
        cre_lines=["Rorb-IRES2-Cre"],
        targeted_structures=["VISp", "VISal", "VISpm"],
        stimuli=["drifting_gratings"],
    )

    summarize_eligible_experiments(experiments)