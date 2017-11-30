# -*- coding: utf-8 -*-
import argparse
import json
import tqdm
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument("--log_files", required=True, nargs='+')
parser.add_argument("--x_key", required=True)
parser.add_argument("--y_key", required=True)
args = parser.parse_args()

figure = pyplot.figure()
plot = figure.add_subplot(111)  # type: pyplot.Figure
last_ys = []
AVG_NUM = 5

for log_file in args.log_files:
    with open(log_file) as f:
        log_data = json.load(f)
    xs, ys = [], []
    for i, data in enumerate(tqdm.tqdm(log_data, desc=log_file)):
        if args.y_key in data and args.x_key in data:
            xs.append(float(data[args.x_key]))
            y = float(data[args.y_key])
            if len(last_ys) == 0:
                for _ in range(AVG_NUM):
                    last_ys.append(y)
            else:
                last_ys.pop(0)
                last_ys.append(y)
            ys.append(sum(last_ys)/AVG_NUM)
            if int(data["iteration"]) > 50000:
                break
    plot.plot(xs, ys, label=log_file)

plot.set_xlabel(args.x_key)
plot.set_ylabel(args.y_key)
# plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
legend = plot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=1, mode="expand", borderaxespad=0., fancybox=True)
# pyplot.tight_layout()
legend.draggable()
pyplot.show(block=True)
# pyplot.savefig(bbox_extra_artists=(legend,))
