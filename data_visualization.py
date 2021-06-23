import dash
import dash_table
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly
import plotly.graph_objects as go
import seaborn as sns


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

    
