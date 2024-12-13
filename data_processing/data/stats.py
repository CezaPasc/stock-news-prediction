from matplotlib import pyplot as plt
import seaborn as sns
import os
 
def plot_dist(data, name, show=False, path="."):
    plt.figure()
    plt.hist(data, bins=100)
    plt.savefig(os.path.join(path, "res", name + ".png"))
    if show:
        plt.show()
    plt.close()


def clean_url(url):
    import re
    return re.sub(r'.+\/\/|www\.|\..+', '', url)

def plot_publisher_freq(data, name, top=25, show=False, path="."):
    data["publisher"] = data["Publisher"].apply(clean_url)

    # Group by the 'publisher' column and count the observations
    publisher_counts = data['publisher'].value_counts().reset_index()
    publisher_counts.columns = ['publisher', 'count']
    
    # Sort the DataFrame by 'count' in descending order and select the top 25
    df_sorted = publisher_counts.sort_values(by='count', ascending=False).head(top)
    # Plot using matplotlib
    plt.figure(figsize=(14, 10))
    sns.barplot(y='publisher', x='count', data=df_sorted, palette='viridis')
    plt.xlabel('Count')
    plt.ylabel('Publisher')
    plt.title('Top 25 Publishers by Count')
    plt.savefig(os.path.join(path, "res", name + ".png"))
    if show:
        plt.show()
    plt.close()

