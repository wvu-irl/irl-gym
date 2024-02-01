# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

# Lengths for bicep and forearm
len_bicep = .5
len_forearm = .5
# Base Y coordinates for the arms
base_y_coords = [-.05, .05]







top_right = Arm(base_y_coords[0],len_bicep,len_forearm,True, 'TR',[.1,-.5,2] )
top_left  = Arm(base_y_coords[1],len_bicep,len_forearm,False,'TL',[.1, .5,2] )
middle_right = Arm(base_y_coords[0],len_bicep,len_forearm,True, 'MR',[.1,-.5,1] )
middle_left  = Arm(base_y_coords[1],len_bicep,len_forearm,False,'ML',[.1, .5,1] )
bottom_right = Arm(base_y_coords[0],len_bicep,len_forearm,True, 'BR',[.1,-.5,0.1] )
bottom_left  = Arm(base_y_coords[1],len_bicep,len_forearm,False,'BL',[.1, .5,0.1] )

initial_flowers = 60
FLOWERS = generate_flower_points(initial_flowers, (0.25, .75), (-.3, .3), (0.25, 2))
ORIGINAL_FLOWERS = FLOWERS


sim_time = 2000
save_video = False
headless = True

TIME_2_POLL_FLOWERS = []
NUMBER_ARMS = []

ARMS = [top_right,top_left, middle_right, middle_left, bottom_right, bottom_left]
number_of_flowers_pollenated = []
NUMBER_ARMS.append(6)

# Calling the function with the defined parameters
fig, ax = plot_arms(ARMS)
plot_flowers(fig, ax, FLOWERS)
ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax), repeat=False, interval=1)
#ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)
plt.show()


#========================== 6 arm experiment =================================================
FLOWERS = ORIGINAL_FLOWERS
ARMS = [top_right,top_left, middle_right, middle_left, bottom_right, bottom_left]
number_of_flowers_pollenated = []
NUMBER_ARMS.append(6)


arms6_number_of_flowers_pollenated = experiment(ARMS, FLOWERS,sim_time)#number_of_flowers_pollenated
plt.show()


print("first experiment done")

# ========================= 4 arm experiment ===================================================
FLOWERS = ORIGINAL_FLOWERS
ARMS = [top_right,top_left, bottom_right, bottom_left]
number_of_flowers_pollenated = []
NUMBER_ARMS.append(4)

# Calling the function with the defined parameters
# fig, ax = plot_arms(ARMS)
# plot_flowers(fig, ax, FLOWERS)
# ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax),repeat=False,interval=1)
#ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)
arms4_number_of_flowers_pollenated = experiment(ARMS, FLOWERS,sim_time)#number_of_flowers_pollenated
plt.show()
plt.close()

print("experiment done")
# ========================= 2 arm experiment ===================================================
FLOWERS = ORIGINAL_FLOWERS
ARMS = [top_right,top_left]
number_of_flowers_pollenated = []
NUMBER_ARMS.append(2)

# Calling the function with the defined parameters
# fig, ax = plot_arms(ARMS)
# plot_flowers(fig, ax, FLOWERS)
# ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax),repeat=False,interval=1)
#ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)
arms2_number_of_flowers_pollenated = experiment(ARMS, FLOWERS,sim_time)#number_of_flowers_pollenated
plt.show()
plt.close()

print("experiment done")
# ========================= 1 arm experiment ===================================================
FLOWERS = ORIGINAL_FLOWERS
ARMS = [top_right]
number_of_flowers_pollenated = []
NUMBER_ARMS.append(1)

# Calling the function with the defined parameters
# fig, ax = plot_arms(ARMS)
# plot_flowers(fig, ax, FLOWERS)
# ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax),repeat=False,interval=1)
#ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)
arms1_number_of_flowers_pollenated = experiment(ARMS, FLOWERS,sim_time)#number_of_flowers_pollenated
plt.show()
plt.close()

print("experiment done")

################################ RESULTS ########################################################

arms6_sum = np.cumsum(arms6_number_of_flowers_pollenated)
arms4_sum = np.cumsum(arms4_number_of_flowers_pollenated)
arms2_sum = np.cumsum(arms2_number_of_flowers_pollenated)
arms1_sum = np.cumsum(arms1_number_of_flowers_pollenated)

plt.figure(figsize=(10,6))
plt.plot(arms6_sum, label="6 Arms")
plt.plot(arms4_sum, label="4 Arms")
plt.plot(arms2_sum, label="2 Arms")
plt.plot(arms1_sum, label="1 Arm")
plt.title("Cumulative Number of Flowers Pollenated Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Total Flowers Pollenated")
plt.legend()
plt.grid(True)
plt.show()

def time_to_pollinate_all(cumulative_sum):
    global initial_flowers
    # Find the index where the cumulative sum first equals 60
    indices = np.where(cumulative_sum == initial_flowers)
    if len(indices[0]) > 0:
        return indices[0][0]
    else:
        return np.nan  # Return NaN if all flowers were not pollinated

# Calculating the time taken to pollinate all flowers for each set of arms
time_arms6 = time_to_pollinate_all(arms6_sum)
time_arms4 = time_to_pollinate_all(arms4_sum)
time_arms2 = time_to_pollinate_all(arms2_sum)
time_arms1 = time_to_pollinate_all(arms1_sum)

# Data for plotting
num_arms = [1, 2, 4, 6]
pollination_times = [time_arms1, time_arms2, time_arms4, time_arms6]

# Plotting
plt.figure(figsize=(10,6))
plt.plot(num_arms, pollination_times, 'o-', markerfacecolor='red')
plt.title("Time to Pollinate All Flowers vs Number of Arms")
plt.xlabel("Number of Arms")
plt.ylabel("Time to Pollinate All Flowers")
plt.grid(True)
plt.show()


def compute_pollination_rate(arms_number_of_flowers_pollenated, arms_sum):
    # Create a time vector for non-zero indices
    time_vector = np.nonzero(arms_number_of_flowers_pollenated)[0]
    time_vector = np.insert(time_vector, 0, 0)

    # Remove duplicates and create a smooth sum
    smooth_sum = np.unique(arms_sum, return_index=True)[1]
    smooth_sum = [arms_sum[index] for index in sorted(smooth_sum)]
    
    # Compute the differences for the derivative
    diff_smooth_sum = np.diff(smooth_sum)
    diff_time = np.diff(time_vector)

    # Avoid division by zero
    diff_time[diff_time == 0] = 1e-10

    # Compute the pollination rate
    pollination_rate = diff_smooth_sum / diff_time
    
    return pollination_rate, time_vector

def compute_bulk_slope(cumulative_sum, window_size=200):
    # Calculate the difference in the cumulative sum over the window
    diff_cumulative_sum = [cumulative_sum[i + window_size] - cumulative_sum[i] for i in range(0, len(cumulative_sum) - window_size, window_size)]
    
    # Calculate the slope (rate) by dividing the difference by the window size
    bulk_slope = [diff / window_size for diff in diff_cumulative_sum]
    
    # Calculate the adjusted time vector as the mid-point of each window
    time_adjusted = [(i + window_size // 2) for i in range(0, len(cumulative_sum) - window_size, window_size)]
    
    return bulk_slope, time_adjusted

# Calculate the bulk slope and adjusted time vectors for each set of arms
bulk_slope6, time6_adjusted = compute_bulk_slope(arms6_sum)
bulk_slope4, time4_adjusted = compute_bulk_slope(arms4_sum)
bulk_slope2, time2_adjusted = compute_bulk_slope(arms2_sum)
bulk_slope1, time1_adjusted = compute_bulk_slope(arms1_sum)

# Plot
plt.figure(figsize=(10,6))
plt.plot(time6_adjusted, bulk_slope6, label='6 arms')
plt.plot(time4_adjusted, bulk_slope4, label='4 arms')
plt.plot(time2_adjusted, bulk_slope2, label='2 arms')
plt.plot(time1_adjusted, bulk_slope1, label='1 arm')
plt.legend()
plt.title("Pollination Rate Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Rate of Pollination [flowers/time-step]")
plt.grid(True)
plt.show()

print('done')