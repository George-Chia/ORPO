import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
fig, ax = plt.subplots(figsize=(6, 6))
# ax.axis('off')
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
square = patches.Rectangle((-3, -3), 6, 6, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(square)
x_line = np.linspace(-3, 3, 100)
y_line1 = -x_line + 0.25
y_line2 = -x_line - 0.25
ax.plot(x_line, y_line1, color='darkblue',linewidth=1.5)
ax.plot(x_line, y_line2, color='darkblue', linewidth=1.5)

# ax.fill_between(x_line, y_line1, y_line2, color='lightblue', alpha=1)


# circle = patches.Circle((3, 3), 1, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.6)
# ax.add_patch(circle)


with open("./random_dataset_0.25width_10000.pkl", "rb") as file:
     dataset = pickle.load(file)
points = dataset['observations'] #+ dataset['next_observations']



x = [point[0] for point in points]
y = [point[1] for point in points]
ax.scatter(x, y, s=50,alpha=0.6,marker='x')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.savefig('./datasets_fig2-a.png')


ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.show()