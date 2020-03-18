1 + 1

print("Hellow world")


def hello(name):
    print("Hello", name)


## matlab like cell mode
print("Execute this")

#
print("but no this!")

##

import altair as alt
import altair_viewer
import pandas as pd

alt.renderers.enable("altair_viewer", inline=False)
df = pd.DataFrame({"x": range(0, 20), "y": range(1, 21)})

altair_viewer.display(alt.Chart(df).mark_bar().encode(x="x", y="y").interactive())

##

source = pd.DataFrame(
    {
        "a": ["A", "X", "C", "D", "E", "F", "G", "H", "I"],
        "b": [29, 5, 44, 91, 81, 53, 19, 807, 53],
    }
)

altair_viewer.display(alt.Chart(source).mark_bar().encode(x="a", y="b").interactive())
