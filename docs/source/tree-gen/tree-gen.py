import os
import json
import re

# exclude_dirs = ["examples", "tests"]


scqubits_root = "../../../scqubits/"

colors = ["#0B5FA5", "#043C6B", "#3F8FD2",  # blue colors
          "#00AE68", "#007143", "#36D695",  # green colors
          "#FF4500", "#692102", "#BF5730"
          ]

#          "#FF9400", "#A66000", "#FFAE40"
#          "#FF6F00", "#A64800", "#BF6E30"

module_cmap = {"core": 0,  # core
               "utils": 1,  # utils
               }

hidden_modules = ['testing', 'dimensions', 'logging_utils', 'matplotlib_utilities']
exclude_patterns = [r'.*_td', r'.*_es', r'.*_mc', r'.*_ode', r'_.*']

module_list = []

num_items = 0

for root, dirs, files in os.walk(scqubits_root):
    # print(root, dirs, files)

    for f in files:
        if f[-3:] == ".py" and f[0] != "_" and f != "setup.py":
            module = f[:-3]
            print(module)
            if module not in hidden_modules:
                idx = module_cmap[module] if module in module_cmap else -1
                color = colors[idx] if idx >= 0 else "black"

                symbol_list = []

                cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (scqubits_root, f)
                for name in os.popen(cmd).readlines():
                    print(name)
                    if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                        symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
                        num_items += 1
                module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})
    for d in dirs:
        print(d)
        for root, dr, files in os.walk(scqubits_root + '/' + d):
            for f in files:
                if f[-3:] == ".py" and f[0] != "_" and f != "setup.py":
                    module = f[:-3]
                    idx = module_cmap[module] if module in module_cmap else -1
                    color = colors[idx] if idx >= 0 else "black"

                    symbol_list = []

                    cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (
                        scqubits_root + '/' + d, f)
                    for name in os.popen(cmd).readlines():
                        if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                            symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
                            num_items += 1
                    module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})

module_list_sorted = sorted(module_list, key=lambda x: x["idx"])
scqubits_struct = {"name": "scqubits", "children": module_list_sorted, "size": 2000}

with open('./scqubits.json', 'w') as outfile:
    json.dump(scqubits_struct, outfile, sort_keys=True, indent=4)

print(num_items)