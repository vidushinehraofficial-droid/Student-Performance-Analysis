import pandas as pd

df = pd.read_csv("StudentData.csv")

top_students = df.sort_values("G3", ascending=False)

print(top_students[["age", "studytime", "G3"]].head(10))
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("StudentData.csv")

plt.scatter(df["absences"], df["G3"])

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")

plt.show()