import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

with open('DLinkNet_Train.json', 'r') as JSON:
    dlink_train_dict = json.load(JSON)
with open('DLinkNet_Val.json', 'r') as JSON:
    dlink_val_dict = json.load(JSON)
with open('NLLinkNet34_Train.json', 'r') as JSON:
    nllink_train_dict = json.load(JSON)
with open('NLLinkNet34_Val.json', 'r') as JSON:
    nllink_val_dict = json.load(JSON)

x = list(range(5, 255, 5))

model_1_train = []
model_1_val = []

model_2_train = []
model_2_val = []


for i in x:
    model_1_train.append(100 * dlink_train_dict[str(i)])
    model_1_val.append(100 * dlink_val_dict[str(i)])
    model_2_train.append(100 * nllink_train_dict[str(i)])
    model_2_val.append(100 * nllink_val_dict[str(i)])


line1, = ax.plot(x, model_1_train, '--', label='D-LinkNet (train)', color='royalblue')
line2, = ax.plot(x, model_1_val, label='D-LinkNet (val)', color='royalblue')
line3, = ax.plot(x, model_2_train, '--', label='NL34-LinkNet (train)', color='crimson')
line4, = ax.plot(x, model_2_val, label='NL34-LinkNet (val)', color='crimson')

formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
plt.gca().xaxis.set_major_formatter(formatter)

ax.legend()
ax.set_aspect(aspect=7.)
ax.set_xlabel('Epochs')
ax.set_ylabel('mIOU (%)')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
