import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
import scipy.stats


# Read the file
file = pd.read_csv('pumf_cchs.csv')
selected_columns = ['DHHGAGE', 'DHH_SEX', 'SMK_045', 'GEN_020', 'WTS_M']
data = file[selected_columns].values.tolist()


def group(age, gender, higher_lower):
    """Function to get rows by age, gender, and stress

    Keyword arguments:
    age -- age group of the person
    gender -- gender of the person
    higher_lower -- stress level of the person

    Returns:
    A list containing rows that match the criteria

    """

    result_list = [];

    for row in data:
        if row[0] == age and row[1] == gender:
            if higher_lower == 'high_stress' and row[3] >= 3:
                result_list.append(row)
            elif higher_lower == 'low_stress' and row[3] <= 2:
                result_list.append(row)
    return result_list


def cigs_per_day(group_data):
    """Calculates the weighted average of cigarettes smoked per day

    Keyword arguments:
    group_data: The data for a group based on age, gender and stress

    Returns:
    The weighted average number of cigarettes smoked per day

    """

    cigs_list = []; weight_list = []; weighted_data = []

    for row in group_data:
        cigs_per_day = float(row[2])
        weight = float(row[4])
        if cigs_per_day < 996:
            cigs_list.append(cigs_per_day)
            weight_list.append(weight)
            weighted_data_value = cigs_per_day * weight
            weighted_data.append(weighted_data_value)

    weighted_data_total = sum(weighted_data)
    weight_total = sum(weight_list)

    return weighted_data_total / weight_total


def compute_standard_error(group_data, confi=0.80):
    """Computes the standard error

    Keyword Arguments:
    group_data: The data for a group based on age, gender and stress
    confi: Confidence level

    Returns:
    The standard error

    """

    data_weighted = []; total_weight = 0;

    # Loop through rows and compute weighted data and total weight
    for row in group_data:
        cigs_per_day = row[2]
        weight = row[4]
        if cigs_per_day < 996:
            data_weighted.append(cigs_per_day * weight)
            total_weight += weight


    # Calculate Standard Error
    ts = scipy.stats.t.ppf((1 + confi) / 2., len(data_weighted) - 1)
    return ts * (scipy.stats.sem(data_weighted, ddof=1) / total_weight) * len(data_weighted)


# Age groups and lists to hold data
ages = [1.0, 2.0, 3.0, 4.0, 5.0];
ml, mh, fl, fh, ml_se, mh_se, fl_se, fh_se = [], [], [], [], [], [], [], []


"""Compute the weighted average cigarettes smoked per day
   and standard error for each group"""
for age in ages:
    ml_rows = group(age, 1.0, 'low_stress')
    ml.append(cigs_per_day(ml_rows))
    ml_se.append(compute_standard_error(ml_rows));

    mh_rows = group(age, 1.0, 'high_stress')
    mh.append(cigs_per_day(mh_rows))
    mh_se.append(compute_standard_error(mh_rows));

    fl_rows = group(age, 2.0, 'low_stress')
    fl.append(cigs_per_day(fl_rows))
    fl_se.append(compute_standard_error(fl_rows));

    fh_rows = group(age, 2.0, 'high_stress')
    fh.append(cigs_per_day(fh_rows))
    fh_se.append(compute_standard_error(fh_rows))


# Plot the graph
age_ranges = ['12 to 17', '18 to 34', '35 to 49', '50 to 64', '65 and older'];
x = np.arange(len(age_ranges));
width = 0.16;
plt.figure(figsize=(8, 6))

plt.bar(x-1.6*width, ml, width, label='Low Stress Male', color='#60f0e4', edgecolor='black', linewidth=0.5);

plt.bar(x-0.4*width, fl, width, label='Low Stress Female', color='pink', edgecolor='black', linewidth=0.5)

plt.bar(x+0.8*width, mh, width, label='Hi Stress Male', color='#60f0e4',
edgecolor='black', linewidth=0.5, hatch='O',);

plt.bar(x+2.0*width, fh, width, label='Hi Stress Female', color='pink',
edgecolor='black', linewidth=0.5, hatch='O')

plt.errorbar(x-1.6*width, ml, yerr=ml_se, fmt='none', c='k', capsize=3)
plt.errorbar(x-0.4*width, fl, yerr=fl_se, fmt='none', c='k', capsize=3)
plt.errorbar(x+0.8*width, mh, yerr=mh_se, fmt='none', c='k', capsize=3)
plt.errorbar(x+2.0*width, fh, yerr=fh_se, fmt='none', c='k', capsize=3)

plt.xlabel('Age of smoker', fontsize=14);
plt.ylabel('Number of Cigarettes per Day', fontsize=14)
plt.title('Daily Smokers, Age + Gender + Stress vs.\n Cigarettes Smoked / Day, 80% Confidence Intervals', fontsize=16)

plt.xticks(x, age_ranges, fontsize=12)
plt.tick_params(axis='x', bottom=False)
plt.gca().xaxis.set_tick_params(pad=0)

plt.ylim(0, 25)
plt.yticks(fontsize=12);

plt.legend()
plt.show()
