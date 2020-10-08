import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter import *
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import tensorflow as tf
import random
import math
from scipy.ndimage.filters import gaussian_filter
import inception5h 
from PIL import ImageTk
from PIL import Image
 
model = inception5h.Inception5h()
len(model.layer_tensors)
model
session = tf.InteractiveSession(graph=model.graph)
    

def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)    
    # Convert to bytes.
    image = image.astype(np.uint8)    
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
        
def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.    
    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image/255.0, 0.0, 1.0)       
        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)      
        # Convert pixels to bytes.
        image = image.astype(np.uint8)
        # Convert to a PIL-image and display it.
        display(PIL.Image.fromarray(image))

def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()
    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)  
    return x_norm     

def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)    
    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()   

def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor       
        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]   
    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))
    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)   
    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)   
    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)   
    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)   
    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)
    return img_resized    
    
def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """
    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))   
    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)   
    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)   
    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)
   # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape
    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4
    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4
    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)
    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size       
        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)
        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size
            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                            y_start_lim:y_end_lim, :]
            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)
            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)
            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)
            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g           
            # Advance the start-position for the y-axis.
            y_start = y_end
        # Advance the start-position for the x-axis.
        x_start = x_end
    return grad

def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.
    
    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """
    # Copy the image so we don't overwrite the original image.
    img = image.copy()    
    print("Image before:")
    plot_image(img)
    print("Processing image: ", end="")
    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)    
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img,
                              tile_size=tile_size)        
        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        # Update the image by following the gradient.
        img += grad * step_size_scaled
        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")
    print()
    print("Image after:")
    plot_image(img)    
    return img

def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """
    # Do a recursive step?
    if num_repeats>0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)          
        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)    
        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)
        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled
    print("Recursive level:", num_repeats)
    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)  
    return img_result





def wavy(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[1]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    



def Crosswaves(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[2]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Cross Waves",image)
    cv2.waitkey(0)        
    
    
    

def cellwall(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[3]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Cell Wall",image)
    cv2.waitkey(0)        
    
    
    


def monkeyeyes(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[4]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Monkey Eyes",image)
    cv2.waitkey(0)        
    
    
    


def animalconversion(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[5]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Animal Conversion",image)
    cv2.waitkey(0)        
    
    
    


def animalhorror(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[6]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Animal Horror",image)
    cv2.waitkey(0)        
    
    
    
    

def wolf(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[7]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wolf",image)
    cv2.waitkey(0)        
    
    
    

def feature8(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[8]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    
    
    


def feature9(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[9]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    
    
    


def feature10(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[10]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    
    
    


def feature11(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[11]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    
    
    


def feature12(): 
   
    image = load_image(filename='image.jpg')
    plot_image(image)
    layer_tensor = model.layer_tensors[12]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,          # only maximizing 6th layer feauters
                    num_iterations=10, step_size=3.0, rescale_factor=0.7,
                    num_repeats=4, blend=0.2)
    save_image(img_result, filename='transformed.png')
    image = cv2.imread("transformed.png",1)
    cv2.imshow("Wavy",image)
    cv2.waitkey(0)        
    
    
    


def deepdream():
    # Making object root form class Tk
    root = Tk()
    root.geometry("1280x720")
        
    # Creating button 
    b1 = Button(root, text = "Wavy",  width=50, fg = "black", command = wavy, padx=10, pady=10)
    b1.grid(row =0 ,column =2)
    
     # Creating button 
    b1 = Button(root, text = "Cross Waves",  width=50, fg = "black", command = Crosswaves , padx=10, pady=10)
    b1.grid(row =0 ,column =4)
    
     # Creating button 
    b1 = Button(root, text = "Cell Wall",  width=50, fg = "black", command = cellwall, padx=10, pady=10)
    b1.grid(row =0 ,column =6)
    
     # Creating button 
    b1 = Button(root, text = "Monkey Eyes",  width=50, fg = "black", command = monkeyeyes, padx=10, pady=10)
    b1.grid(row =2 ,column =2)
    
     # Creating button 
    b1 = Button(root, text = "Animal Conversion",  width=50, fg = "black", command = animalconversion, padx=10, pady=10)
    b1.grid(row =2 ,column =4)
    
     # Creating button 
    b1 = Button(root, text = "Animal Horror",  width=50, fg = "black", command = animalhorror, padx=10, pady=10)
    b1.grid(row =2 ,column =6)
    
     # Creating button 
    b1 = Button(root, text = "Wolf",  width=50, fg = "black", command = wolf, padx=10, pady=10)
    b1.grid(row =4 ,column =2)
    
     # Creating button 
    b1 = Button(root, text = "Extreme",  width=50, fg = "black", command = feature8, padx=10, pady=10)
    b1.grid(row =4 ,column =4)
    
     # Creating button 
    b1 = Button(root, text = "F9",  width=50, fg = "black", command = feature9, padx=10, pady=10)
    b1.grid(row =4 ,column =6)
    
     # Creating button 
    b1 = Button(root, text = "F10",  width=50, fg = "black", command = feature10, padx=10, pady=10)
    b1.grid(row =6 ,column =2)
    
     # Creating button 
    b1 = Button(root, text = "F11",  width=50, fg = "black", command = feature11, padx=10, pady=10)
    b1.grid(row =6 ,column =4)
    
     # Creating button 
    b1 = Button(root, text = "F12",  width=50, fg = "black", command = feature12, padx=10, pady=10)
    b1.grid(row =6 ,column =6)
    
     
        
    first_app = root.mainloop()

    
    return

class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source
 
         # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
 
         # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
         self.canvas.pack()
 
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
         self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
         
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Enter to Deep Dream", width=50, command=deepdream)
         self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 15
         self.update()
 
         self.window.mainloop()
 
     def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             cv2.imwrite("image.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
     def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
         self.window.after(self.delay, self.update)
 
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "TDeep Dream")

