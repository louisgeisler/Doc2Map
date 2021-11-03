from Doc2Map import Doc2Map

from sklearn.datasets import fetch_20newsgroups

#Doc2Map.test_20newsgroups("test-learn", "all")

d2m = Doc2Map(speed="learn", ramification=22)
dataset = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

for i, (data, target) in enumerate(zip(dataset.data, dataset.target)):
    d2m.add_text(data, str(i), target=target)

d2m.build()
d2m.display_tree()
d2m.display_simplified_tree()
d2m.plotly_interactive_map()
d2m.scatter()
d2m.interactive_map()

print("Fin")