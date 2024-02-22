def count_raters(data):
    
    real_data_filtered = data[data['RaterType'] != 'self']

    # Group by 'ESI_Key' and 'RaterType', then count the occurrences
    rater_counts = real_data_filtered.groupby(['ESI_Key', 'RaterType']).size().unstack(fill_value=0)

    # Calculate the median number of each rater type across all 'self' raters
    median_rater_counts = rater_counts.median()
    return(median_rater_counts)