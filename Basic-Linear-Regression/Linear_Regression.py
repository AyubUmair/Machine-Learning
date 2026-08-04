"""
Interactive Linear Regression Visualization

This program demonstrates:
- Basic Linear Regression
- Prediction using y = wx + b
- Cost Function
- Interactive Weight Slider
- Cost Curve Visualization

Author: A. Umair
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
x_margin = (x_train.max() - x_train.min()) * 0.1
y_margin = (y_train.max() - y_train.min()) * 0.1

print(f"x_train.shape: {x_train.shape}")
# m denotes the number of training examples which is six in this case.
m = x_train.shape[0]      # OR
m = len(x_train)

print(f"Number of training examples: {m}")

# Plotting

plt.scatter(x_train, y_train, marker='o', c='b') # Plot the data points
plt.title("Data Points") # Set title
plt.xlabel("Size(1000 sqft)")   # Set X label
plt.ylabel("Price (1000 LKR)")  #Set Y label
plt.show()

# Linear Regression model is Y = aX+b Where a = gradient and b. So let a = 200, b=100
w = 209.359
b = 2.4325

# Below function computes the Prediction of Our model
# We have to choose a & b insuch a way that it fits with our trining data
def compute_model_output(x,w,b):
    return w*x+ b

temp_model = compute_model_output(x_train,w,b)

plt.plot(x_train,temp_model,c='b',label="Actual Values")   #Plot the prediction
plt.scatter(x_train, y_train, marker='x', c='y') # Plot the data points
plt.title("Regression model for housing Prices") # Set title
plt.xlabel("Size(1000 sqft)")   # Set X label
plt.ylabel("Price (1000 LKR)")  #Set Y label
plt.show()



# Calculating the Cost
def compute_cost(x,y,w_range,b):
    m = x.shape[0]
    cost_sum = []
    for w in w_range:
        f_wb = w*x+b
        total_cost = np.sum((f_wb-y)**2) /(2*m)
        cost_sum.append(total_cost)
    return np.array(cost_sum)

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,5),label="Actual Values")
plt.subplots_adjust(bottom = 0.25)

ax1.scatter(x_train, y_train, marker='x', c='red')
line_pred, = ax1.plot(x_train, temp_model, c='blue',linewidth=4,linestyle='--',label="Our Prediction Curve")
ax1.set_title("Housing Prices")
ax1.set_xlabel("Size(1000 sqft)")
ax1.set_ylabel("Price (1000 LKR)")
ax1.set_xlim(x_train.min() - x_margin, x_train.max() + x_margin)
ax1.set_ylim(y_train.min() - y_margin, y_train.max() + y_margin)
x_line = np.array([x_train.min() - x_margin, x_train.max() + x_margin])
ax1.legend()

w_range = np.linspace(-20,400,100)
cost_range = compute_cost(x_train,y_train,w_range,b)
w_init = 200
#Cost curve & Current Cost Point

ax2.plot(w_range,cost_range, c ='blue', linewidth = '4')
f_wb_init = w_init * x_train + b
init_cost = np.sum(f_wb_init-y_train)**2 / (2*m)
dot_cost, = ax2.plot(w_init,init_cost,c='darkred',marker='o',markersize='12')
ax2.set_title('cost vs W (b = 100)')
ax2.set_xlabel('W')
ax2.set_ylabel('Cost')


fig.suptitle(f'minimize Cost: Current Cost={init_cost:.0f}')

slider_ax = plt.axes([0.25,0.1,0.5,0.03])
w_slider = Slider(
    ax = slider_ax,
    label = 'W',
    valmin = -20.0 ,
    valmax = 400 ,
    valinit= 200,
    valfmt = '%0.0f',
    color ='blue'
)
reset_ax = plt.axes([0.9,0.1,0.04,0.03])
reset_button = Button(reset_ax,label = 'Reset',color='grey',hovercolor='0.975')

def update_slider(val):
    w_current = w_slider.val
    line_pred.set_ydata(w_current*x_train + b)
    f_wb_current = w_current * x_train + b
    current_cost = np.sum((f_wb_current-y_train)**2) / (2*m)

    dot_cost.set_data([w_current],[current_cost])
    ax2.legend([dot_cost],[f'cost at w = {w_current:.0f}'])

    fig.suptitle(f'minimize Cost: Current Cost={current_cost:.0f}')
    fig.canvas.draw_idle()
def reset(event):
    w_slider.reset()

w_slider.on_changed(update_slider)
reset_button.on_clicked(reset)
plt.show()
