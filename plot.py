import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(values, f_name, use_seaborn=False):
    plt.clf()
    if use_seaborn:
        ax = sns.distplot(values)
    else:
        plt.hist(values)
    if f_name:
        plt.savefig(f_name)
    else:
        plt.show()

def figure_3(outlets_poi_stats, f_name):
    outlets = list(outlets_poi_stats.keys())
    # sorted in order as in the paper
    outlets = ['Reuters', 'Guardian', 'Washington Post', 'New York Times', 'Fox News', 'Vox', 'CNN', 'Buzzfeed News', 'Breitbart', 'NPR', 'New York Post', 'Atlantic', 'Talking Points Memo', 'Business Insider', 'National Review']
    x = np.arange(len(outlets))
    bins_colors = {
        'omitted_1': '#f6a8a2',
        'omitted_2': '#ef675d',
        'omitted_3': '#e82517',
        'corroborated_1': '#a2adf6',
        'corroborated_2': '#5d70ef',
        'corroborated_3': '#1733e8',
    }

    data = [(np.array([outlets_poi_stats[o][k] for o in outlets]), v) for k, v in bins_colors.items()]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    # plt.clf()
    bottom = np.zeros((len(outlets,)))

    for d in data:
        values, color = d
        ax1.bar(x, values, color=color, bottom=bottom)
        bottom += values
    # totals is the sum of all for the same outlet
    totals = bottom
    bottom = np.zeros((len(outlets,)))
    # now can plot percentages
    for d in data:
        values, color = d
        perc = values / totals
        ax2.bar(x, perc, color=color, bottom=bottom)
        bottom += perc
    plt.xticks(x, outlets, rotation=40, horizontalalignment='right')
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('Percentage')
    fig.tight_layout()
    if f_name:
        plt.savefig(f_name)
    else:
        plt.show()
