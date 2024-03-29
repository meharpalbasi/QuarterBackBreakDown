import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np

df_2023 = nfl.import_pbp_data([2023]) 


pbp_clean = df_2023[df_2023['pass'] == 1 & (df_2023['play_type'] != 'no_play')]
pbp = pbp_clean[
    (pbp_clean['qb_kneel'] == 0) &
    (pbp_clean['qb_spike'] == 0)
]

total_plays_per_qb = pbp.groupby('passer')['epa'].count().reset_index(name='total_plays')

qbs_with_min_plays = total_plays_per_qb[total_plays_per_qb['total_plays'] >= 10]['passer']

filtered_passing_plays = pbp[pbp['passer'].isin(qbs_with_min_plays)]


# Define a function for safe sorting that handles None and mixed types
def safe_sort(values):
    # Exclude np.nan using pd.notna for numerical columns if necessary
    return sorted([str(x) for x in values if pd.notna(x)])

def dynamic_filter_and_aggregate(data, selections):
    for key, values in selections.items():
        if values:  # If the list is not empty, apply filter
            if 'Not Specified' in values:
                if len(values) > 1:
                    specified_values = [x for x in values if x != 'Not Specified']
                    # Convert to appropriate type based on the key
                    if key in ['defenders_in_box', 'number_of_pass_rushers']:
                        specified_values = [int(x) for x in specified_values if x.isdigit()]  # Safeguard with isdigit() and convert
                    data = data[(data[key].isnull()) | (data[key].isin(specified_values))]
                else:
                    # Filtering for null values only in the column
                    data = data[data[key].isnull()]
            else:
                # Standard filtering, with type conversion for numeric fields
                if key in ['defenders_in_box', 'number_of_pass_rushers']:
                    specified_values = [int(x) for x in values if x.isdigit()]  # Convert only if digit, safe for numeric fields
                    data = data[data[key].isin(specified_values)]
                else:
                    # Apply filtering without conversion for non-numeric fields
                    data = data[data[key].isin(values)]



# Preprocess to include only non-scramble attempts in completion calculations
    data['valid_pass_attempt'] = data.apply(lambda row: row['complete_pass'] if row['qb_scramble'] == 0 else np.nan, axis=1)


    # If no selections are made (i.e., all are empty), treat it as 'any' for all
    groupby_cols = [k for k, v in selections.items() if v]
    if not groupby_cols:
        groupby_cols = ['passer', 'defense_coverage_type', 'defense_man_zone_type', 'defenders_in_box', 'number_of_pass_rushers','was_pressure']
    
    # Aggregate data
    aggregated_data = data.groupby(groupby_cols).agg(
        avg_epa=('epa', 'mean'),
        play_count=('epa', 'size'),
        completion_percentage=('valid_pass_attempt', lambda x: np.nan if len(x) == 0 else (x.sum() / x.count()) * 100),
        avg_wpa=('wpa', 'mean'),
        avg_air_yards=('air_yards', 'mean'),
        avg_time_to_throw=('time_to_throw','mean')
    ).reset_index()
    
    return aggregated_data, groupby_cols + ['avg_epa', 'play_count', 'completion_percentage', 'avg_wpa', 'avg_air_yards','avg_time_to_throw'],

# Streamlit UI for user selections
st.title('NFL Pass Play Analysis')

def safe_sort_with_null_option(values):
    """Sorts values and includes an option for nulls."""
    sorted_values = sorted([x for x in values if pd.notnull(x)])
    return ['Not Specified'] + [str(int(x)) for x in sorted_values]

# Multi-selects without 'any' as an explicit option
selected_passer = st.multiselect(
    'Passer',
    options=safe_sort(filtered_passing_plays['passer'].dropna().unique())
)
selected_coverage_type = st.multiselect(
    'Defense Coverage Type',
    options=safe_sort(filtered_passing_plays['defense_coverage_type'].dropna().unique())
)
selected_man_zone_type = st.multiselect(
    'Man/Zone Type',
    options=safe_sort(filtered_passing_plays['defense_man_zone_type'].dropna().unique())
)
selected_defenders_in_box = st.multiselect(
    'Defenders in Box',
    options=safe_sort_with_null_option(filtered_passing_plays['defenders_in_box'].unique()),
    format_func=lambda x: x  # Assuming numeric values and "Not Specified" for nulls
)
selected_number_of_rushers = st.multiselect(
    'Number of Pass Rushers',
    options=safe_sort_with_null_option(filtered_passing_plays['number_of_pass_rushers'].unique()),
    format_func=lambda x: x  # Assuming numeric values and "Not Specified" for nulls
)
selected_was_pressure = st.multiselect(
    'Was Pressure?',
    options=['Yes', 'No'],  # User-friendly options
    default=['Yes', 'No']  # Default to showing both options
)

if 'Yes' in selected_was_pressure and 'No' in selected_was_pressure:
    selected_was_pressure = [True, False]  # Interpret as 'any' if both are selected
elif 'Yes' in selected_was_pressure:
    selected_was_pressure = [True]
elif 'No' in selected_was_pressure:
    selected_was_pressure = [False]
else:
    selected_was_pressure = [] 
# Prepare selections for aggregation
selections = {
    'passer': selected_passer,
    'defense_coverage_type': selected_coverage_type,
    'defense_man_zone_type': selected_man_zone_type,
    'defenders_in_box': selected_defenders_in_box,
    'number_of_pass_rushers': selected_number_of_rushers,
    'was_pressure': selected_was_pressure
}

aggregated_data, display_columns = dynamic_filter_and_aggregate(filtered_passing_plays, selections)

# Display the results
if not aggregated_data.empty:
    st.write(aggregated_data)
else:
    st.write("Please make a selection to view data.")