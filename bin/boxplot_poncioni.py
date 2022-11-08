import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    '''
    import results, read df, plot results
    '''
    data = pd.read_csv('bin/mia-result/2022-11-06-16-12-39/results.csv', sep = ';')
    data['1SPCFTY'] = 1 - data['SPCFTY']
    # plt.figure(figsize=(10, 10))
    # sns.boxplot(x='LABEL', y='DICE', data=data, palette="Set3", linewidth=1)
    # plt.title('Dice coefficients')
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(x='DICE', y='HDRFDST', data=data, hue='LABEL', palette="Set3", linewidth=1)
    # plt.title('Correlation between Dice and Hausdorff95 distance')
    # plt.show()


    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='1SPCFTY', y='SNSVTY', data=data, hue='LABEL', palette="Set3", linewidth=1)
    plt.title('Correlation between Specificity and Sensitivity')
    plt.show()

if __name__ == '__main__':
    main()
