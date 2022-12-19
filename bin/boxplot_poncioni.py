import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# font stix
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = 16


def plot_violin(data, title, savepath):
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="LABEL", y="SBD_MEAN", data=data, palette="Set3", linewidth=1)
    plt.title(title, weight="bold", fontsize=20)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_box(data, title, savepath):
    plt.figure(figsize=(10, 10))
    sns.boxplot(x="LABEL", y="SBD_MEAN", data=data, palette="Set3", linewidth=1)
    plt.title(title, weight="bold", fontsize=20)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def main():
    """
    import results, read df, plot results
    """
    data = pd.read_csv("bin/mia-result/2022-12-13-11-56-12/results.csv", sep=";")
    data_sbd = pd.read_csv(
        "/home/simoneponcioni/Documents/03_LECTURES/MIALab/bin/mia-result/results_bsd.csv",
        sep=",",
    )
    title_s = "Symmetric Boundary Dice Comparison"
    basepath = "bin/mia-result/2022-12-13-11-56-12"
    plot_violin(data_sbd, title_s, f"{basepath}/sbd_violin.png")
    plot_box(data_sbd, title_s, f"{basepath}/sbd_box.png")


if __name__ == "__main__":
    main()
