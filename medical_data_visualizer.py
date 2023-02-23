import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("./medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where(df.weight/(df.height/100)**2 > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.cholesterol = np.where(df.cholesterol == 1, 0, 1)
df.gluc = np.where(df.gluc == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df,
                value_vars= ['active', 'alco', 
                             'cholesterol', 'gluc',
                             'overweight', 'smoke'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.melt(df,
                id_vars=['cardio'],
                value_vars= ['active', 'alco', 
                             'cholesterol', 'gluc',
                             'overweight', 'smoke'])

    # Draw the catplot with 'sns.catplot()'
    catPlot = sns.catplot(
                x="variable",
                col="cardio",
                hue="value",
                data=df_cat,
                kind="count"
                ).set(ylabel="total")


    # Get the figure for the output
    fig = catPlot.figure

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) & 
    (df['height'] >= df.height.quantile(0.025)) & 
    (df['height'] <= df.height.quantile(0.975)) & 
    (df['weight'] <= df.weight.quantile(0.975)) &
    (df['weight'] >= df.weight.quantile(0.025))
    ]


    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = []

    for i in range(len(corr.columns)):
      mask_i = []
      for j in range(len(corr.columns)):
        if i <= j:
          mask_i.append(True)
        else: 
          mask_i.append(False)
      mask.append(mask_i)

    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    labels = corr.columns.to_list()

    xticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    color = ["Black", "#F4A460", "#D2691E", "#191970", "#B22222", "Maroon", "#4169E1"]
    
    ax.set_facecolor("white")
    fig.set_size_inches(9, 13)
    cbar_ax = fig.add_axes([.91, .3, .02, .35])
    ax.set_xticks(xticks, labels=labels, minor=True, rotation=90, ha='right')

    # Draw the heatmap with 'sns.heatmap()'
    heat = sns.heatmap(corr,
            cmap=sns.set_palette(color),
            annot=True,
            fmt='.1f',
            ax=ax,
            mask=np.array(mask),
            cbar_ax = cbar_ax
           )

    heat.set_yticklabels(heat.get_yticklabels(), rotation=0, horizontalalignment='right')


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
