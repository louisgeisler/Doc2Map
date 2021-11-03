import json
from numba.cuda.args import InOut
import numpy as np
import os, sys
from Doc2Map import Doc2Map



if not os.path.exists(os.path.dirname(sys.argv[0])+"/simplewiki.json"):
    print("The simplewiki.json isn't here. Please download it from that web page: https://www.kaggle.com/louisgeisler/simple-wiki?select=simplewiki.json")
    os.system(r"""start https://www.kaggle.com/louisgeisler/simple-wiki?select=simplewiki.json""")
    input()
    exit()

with open(os.path.dirname(sys.argv[0])+"/simplewiki.json", "r", encoding='utf-8') as f:
    lData = json.load(f)

p1 = np.percentile([len(data["content"]) for data in lData], 80)
p2 = np.percentile([len(data["content"]) for data in lData], 90)
lData = [data for data in lData if p1<len(data["content"])<p2]
lData = np.random.choice(lData, 80, replace=False)
n = len(lData)
percent = 5*n/100
d2m = Doc2Map(speed="deep-learn", ramification=5, min_count = 3)
for i, info in enumerate(lData):
    if (i%percent==0):
        print((100*i)//percent)
    d2m.add_text(info["content"], info["title"], url = info["url"].replace("http://s.wikipedia.org/","https://simple.wikipedia.org/"))
        
d2m.build()
d2m.display_tree()
d2m.display_simplified_tree()
d2m.scatter()
d2m.plotly_interactive_map()
d2m.interactive_map()