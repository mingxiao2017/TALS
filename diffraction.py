import numpy as np
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# Calculate the average value of specified range of images in the folder
def avg_images(folder_path, start_num, end_num):
    end_num=end_num+1
    file_paths = [os.path.join(folder_path, f"{i}.tif") for i in range(start_num, end_num)]

    sample_image = cv2.imread(file_paths[0], cv2.IMREAD_UNCHANGED)
    image_dtype = sample_image.dtype
    if image_dtype == np.uint8:
        #print("The image format is 8-bit.")
        images = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in file_paths]
        avg_image = np.mean(images, axis=0).astype(np.float32)
        avg_image = np.round(avg_image).astype(np.uint8)

    elif image_dtype == np.uint16:
        #print("The image format is 16-bit.")
        images = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in file_paths]
        avg_image = np.mean(images, axis=0).astype(np.float64)
        avg_image = np.round(avg_image).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image format: {image_dtype}")
        
    return avg_image

# Subtract background
def subtract_background(image, image_back):
    image = image.astype(np.float32)
    image_back = image_back.astype(np.float32)
    image = np.maximum(image - image_back, 0)
    return image.astype(np.uint16)


#polt and save heatmap
def plot_heatmap(channel, folder_path, threshold=0):
  
    log_channel = np.log1p(channel)
    threshold = threshold # Set a threshold as needed, and pixels below this value will be set to 0 (black).
    masked_log_channel = np.ma.masked_where(log_channel < threshold, log_channel)
    cmap = plt.get_cmap('jet')
    new_cmap = ListedColormap(cmap(np.linspace(0, 1, cmap.N)))
    new_cmap.set_under(color='black')


    height, width = log_channel.shape
    fig_width = width / 100
    fig_height = height / 100
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, 2, width_ratios=[20, 1], figure=fig) 
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])

    sns.heatmap(masked_log_channel, 
                ax=ax,
                cmap=new_cmap, 
                cbar=True,
                cbar_ax=cbar_ax,
                vmin=threshold, 
                vmax=8.3,
                square=True,
                cbar_kws={'label': 'Intensity (log(I))'}
    ) 
    ax.axis('off')
    
    output_path = fr'{folder_path}\heatmap.tif'
    plt.savefig(output_path, format='jpeg', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()


# Detect the scattering range and center
def detect_scattering_range(channel, image):
    # Determine the range of scattered signals
    blurred = cv2.GaussianBlur(channel, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 100, 1000, cv2.THRESH_BINARY)  # (src, thresh, maxval, type)
    thresh = np.uint8(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # Determine the scattering center
    blurred = cv2.GaussianBlur(channel, (51, 51), 0)  # 使用较大的核来确保平滑局部区域
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    center_x, center_y = max_loc
    # Mark the scattering range and center
    output = image.copy()
    cv2.circle(output, (center_x, center_y), 10, (255, 255, 255), -1)  # 在局部区域中心点上画一个红色圆点
    cv2.drawContours(output, [largest_contour], -1, (255, 255, 255), 2)
    # Display
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"Scattering Center:( {center_x}, {center_y})")
    plt.show()
    return largest_contour, center_x, center_y

# Calculate the average value and variance of the intensity in the scattering range
def calculate_average_and_variance(blue_channel, center_x, center_y, largest_contour, intensity_threshold=0):
    #calculate max_radius
    contour_x = largest_contour[:, 0, 0]
    max_radius = max(abs(contour_x - center_x))
    # image size
    height, width = blue_channel.shape
    y = np.arange(height).reshape(-1, 1)
    x = np.arange(width)

    # Calculate the distance between each pixel and the center point
    distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    rounded_distance = np.around(distances, decimals=2)  
    flatten_distance = rounded_distance.flatten()  
    flatten_blue_channel = blue_channel.flatten() 

    # Create a mask for the largest contour area
    contour_mask = np.zeros_like(blue_channel, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    flatten_contour_mask = contour_mask.flatten()

    df = pd.DataFrame({
        'R': flatten_distance,
        'intensity': flatten_blue_channel,
        'mask': flatten_contour_mask
    })
    
    spectrum_filtered = df[(df['R'] <= max_radius) & (df['mask'] == 255) & (df['intensity'] > intensity_threshold)]
    spectrum_sorted = spectrum_filtered.sort_values(by='R')

    result = spectrum_sorted.groupby('R').agg(
        average_intensity=('intensity', 'mean'),
        intensity_variance=('intensity', 'std')
    ).reset_index()
    return result


# Function to calculate the q value
def calculate_q(theta, n_eff, alpha, lambda_):
    term1 = np.sin(np.radians(2*theta - alpha))
    term2 = np.sin(np.radians(alpha))
    inverse_sin_term1 = np.degrees(np.arcsin(term1 / n_eff))
    inverse_sin_term2 = np.degrees(np.arcsin(term2 / n_eff))
    sin_half_sum = np.sin(np.radians(0.5 * (inverse_sin_term1 + inverse_sin_term2)))
    q = (4 * np.pi * n_eff / lambda_) * sin_half_sum
    return q

