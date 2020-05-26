import seaborn as sns
import matplotlib.pyplot as plt

#  sns.set(style="darkgrid")
sns.set_style("white")
iris = sns.load_dataset("iris")

# Subset the iris dataset by species
setosa = iris.query("species == 'setosa'")
virginica = iris.query("species == 'virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")


# Draw the two density plots
ax = sns.kdeplot(
    setosa.sepal_width,
    setosa.sepal_length,
    cmap="Reds",
    shade=True,
    shade_lowest=False,
    legend=False,
)

ax = sns.kdeplot(
    virginica.sepal_width,
    virginica.sepal_length,
    cmap="Blues",
    shade=True,
    shade_lowest=False,
)

ax.set(xticklabels=[])
ax.set(xlabel=None)
ax.set(yticklabels=[])
ax.set(ylabel=None)
sns.despine(bottom=True, left=True)

plt.savefig("weak_cert_iso.png", dpi=300)

# Add labels to the plot
#  red = sns.color_palette("Reds")[-2]
#  blue = sns.color_palette("Blues")[-2]
#  ax.text(2.5, 8.2, "virginica", size=16, color=blue)
#  ax.text(3.8, 4.5, "setosa", size=16, color=red)
