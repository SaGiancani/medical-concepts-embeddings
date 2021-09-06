import dash
import dash_table

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go
import seaborn as sns


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot", H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""
    plt.rcParams["figure.figsize"] = (30,10)
    #matplotlib.style.use('fivethirtyeight') 
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    #x_coords = []
    #heights = []
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df: loop over the dfs aka the two columns
        for j, pa in enumerate(h[i:i+n_col]): # loop over the subcolumns
            x_ = 0
            for z, rect in enumerate(pa.patches): # for each index: aka loop over the x elements
                #height = rect.get_height()
                x_ = rect.get_x() + 1 / float(n_df + 1) * i / float(n_col)
                rect.set_x(x_)
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))
                #x_coords.append(x_+(1 / float(n_df + 1))/2)
                #heights.append(height)
                #print(height)
    
    # Code for text labeling the bars
    #x_coords = np.transpose(np.array(x_coords).reshape(-1, n_ind))
    #heights = np.transpose(np.array(heights).reshape(-1, n_ind))
    #tmp_y = []
    #tmp_x = []
    #for i in range(n_ind):
    #    for j in range(round(len(heights[i])/n_col)):
    #        print(j)
    #        tmp_y.append(np.sum(heights[i][j*n_col:j*n_col+n_col]))
    #        tmp_x.append(x_coords[i][j*n_col+n_col-1])
    
    #count = 0
    #for x,y in zip(tmp_x, tmp_y):
    #    if (count%2 == 0):
    #        plt.text(x, y, text[0], ha='center', va='top', color='w')    
    #    else:
    #        plt.text(x, y, text[1], ha='center', va='top', color='w')                 
    #    count +=1
                
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 45)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="white", hatch=H * i))
    
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe


def plot_heatmaps(dict_parameters, seed, name_seed, red_sign = True):

    fig, big_axes = plt.subplots( figsize=(15.0, 25.0) , nrows=len(dict_parameters['models']), ncols=1, sharey=True) 

    a = []
    count_title = 0

    for i, k in enumerate(dict_parameters['models']):
        if "ones" in k['results'][name_seed]:
            #for inserting it in the columns count of the subplot**
            ones_value = 3 
            for o, j in enumerate([k['results'][name_seed]['pos'],
                                   k['results'][name_seed]['neg'],
                                   k['results'][name_seed]['ones']]):
                a.append(j)
        else:
            ones_value = 2
            for o, j in enumerate([k['results'][name_seed]['pos'],
                                   k['results'][name_seed]['neg']]):
                a.append(j)

    #This for-loop is due to the building of a superior subplot for the partial titles for each two heatmaps  
    for row, big_ax in enumerate(big_axes, start=1):
        if count_title == 0:
            big_ax.set_title("%s\n\n%s \n" % (name_seed,dict_parameters['models'][row-1]['name']), fontsize=16)
            count_title +=1
        
        else:
            big_ax.set_title("%s \n" % dict_parameters['models'][row-1]['name'], fontsize=16)
    
        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    #This for-loop is due to each heatmap
    for i in range(1,(len(dict_parameters['models'])*ones_value+1)):
        ax = fig.add_subplot(len(dict_parameters['models']),ones_value,i)
        sns.heatmap(a[i-1], cmap="viridis", ax=ax)
        ax.set_xlabel('Most similar words', fontsize=10)
        ax.set_ylabel('Seeds', fontsize='medium')
        
        #It doesnt compute the list comprehensions every cicle: they just change every 2 
        if (i%ones_value!=0):
            c = [item.get_text() for item in ax.get_yticklabels(which='both')]
            seeds_ = [seed[int(p)] for p in c]
            hightlights = [seed.index(l) for l in dict_parameters['models'][math.floor((i-1)/ones_value)]['results'][name_seed]['oov']]
            
        ax.set_yticklabels(seeds_, rotation=40, ha="right", fontsize=8)
        #print(hightlights)
        if red_sign:
            for index in hightlights:
                ax.add_patch(Rectangle((0, index), dict_parameters['K-Most_Similar'], 1, fill=False, edgecolor='red', lw=0.5))
        
    fig.set_facecolor('w')
    plt.tight_layout()
    fig.savefig('%s.png' % datetime.datetime.now().strftime("%d_%m_%Y%H%M%S"))
    
    
def tabular_result(k_results, k = None):
#print('{:s}: pos_dcg: {:.2f}, neg_dcg: {:.2f}, perc_dcg: {:.4f}, oov: {:d}, #seed: {:d}\n'.format(seed[0], measures.pos_dcg(d), measures.neg_dcg(d), measures.percentage_dcg(d), measures.oov(d), len(d)))
# big_g = {k: {name: {seed: [pos, neg, perc, oov]}}}
# big_g[k][name][seed[0]] = [measures.pos_dcg(d), measures.neg_dcg(d), measures.percentage_dcg(d), measures.oov(d), len(seed[1])]


    dash_table.DataTable(
        columns=[
            {"name": "Name", "id": "name"},
            {"name": "Seed", "id": "seed"},
            {"name": "posDCG", "id": "pos"},
            {"name": "negDCG", "id": "neg"},
            {"name": "percDCG", "id": "perc"},
            {"name": "OOV", "id": "oov"},
            {"name": "#Seed", "id": "#"},
        ],
        data=[
            {
                "name": name,
                "seed": seed_name,
                "pos": values[0],
                "neg": values[1],
                "perc": values[2],
                "oov": values[3],
                "#": values[4],
            }
            for name, seeds in zip(list(k_results.keys()), list(k_results.values())) 
            for seed_name, values in zip(list(seeds.keys()), list(seeds.values()))
        ],
        merge_duplicate_headers=True,
        style_as_list_view=True,
        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'},
    )

    
