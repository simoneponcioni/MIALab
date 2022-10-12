import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    '''
    import results, read df, plot results
    '''
    data = pd.read_csv('bin/mia-result/2022-10-12-12-03-29/results.csv', sep = ';')
    plt.figure(figsize=(10, 10))
    sns.boxplot(x='LABEL', y='DICE', data=data, palette="Set3", linewidth=1)
    plt.title('Dice coefficients')
    plt.show()


if __name__ == '__main__':
    main()
