import pandas as pd

weekdays = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}
periods = {'days':7, 'hours': 168, 'minutes': 10080}
week_indices = {'days': [f'{weekdays[x]}' for x in range(7)],
                'hours': [f'{weekdays[x//24]}-{str(x%24).zfill(2)}:00' for x in range(168)],
                'minutes': [f'{weekdays[x//1440]}-{str(x//60).zfill(2)}:{str(x%60).zfill(2)}' for x in range(10080)]}


def timeseries_to_week_lists(timeseries: pd.Series, resampled_at: str = 'hours'):
    parsed_periods = periods[resampled_at]
    if not isinstance(timeseries, pd.Series):
        raise ValueError('Only functions on a pandas Series object.')
    if resampled_at not in periods.keys():
        raise ValueError('Does not support that ')

    weeks = []
    # Auto-add the first-item to our current week...
    # so that we avoid any issues with a week starting exactly at sunday-midnight
    current_week = [timeseries[0]]
    # Iterate through the remaining periods and add them into our nested lists as appropriate
    for index_date, signal in timeseries[1:].iteritems():
        if index_date.weekday() == 0 and index_date.hour == 0 and index_date.minute == 0:
            # Found the end of one week make sure it was a complete week
            while len(current_week) < parsed_periods:
                current_week.insert(0, None)
            # add that completed list to our master list-of-lists
            weeks.append(current_week)
            # start the new week off
            current_week = [signal]
        else:
            current_week.append(signal)
    # Make sure we capture anything left of the final week...
    while len(current_week) < parsed_periods:
        current_week.append(None)
    # add that final week into the master
    weeks.append(current_week)
    return weeks


def split_weeks(timeseries: pd.Series, resampled_at: str = 'hours') -> pd.DataFrame:
    week_lists = timeseries_to_week_lists(timeseries, resampled_at)
    final_index = week_indices[resampled_at]
    return pd.DataFrame({f'week_{x}': row_data for x, row_data in enumerate(week_lists)}, index=final_index)


def split_overlapping_weeks(timeseries: pd.Series, additional_periods=12, resampled_at: str = 'hours') -> pd.DataFrame:
    weeks = timeseries_to_week_lists(timeseries, resampled_at)
    final_index = week_indices[resampled_at]
    final_index = [str('-') + x for x in final_index[-additional_periods:]] + final_index + \
                  [str('+') + x for x in final_index[:additional_periods]]
    # empty dataframe for the final result
    result_df = pd.DataFrame(index=final_index)
    # Pull the overlapping Data from the other weeks
    for week_idx, week_data in enumerate(weeks):
        # Add the end of the prior to the start of this one...
        if week_idx > 0:
            week_data = weeks[week_idx - 1][-additional_periods:] + week_data
        else:
            week_data = [None for _ in range(additional_periods)] + week_data
        # Add the beginning of the subsequent to the end of this one...
        if week_idx + 1 < len(weeks):
            week_data = week_data + weeks[week_idx + 1][:additional_periods]
        else:
            week_data = week_data + [None for _ in range(additional_periods)]
        # Add the week to the dataframe
        result_df[f'week_{week_idx}'] = week_data
    return result_df
