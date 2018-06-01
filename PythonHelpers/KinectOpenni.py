from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import time


Depth_ResX = 512
Depth_ResY = 424
Depth_fps = 30
RGB_ResX = 512
RGB_ResY = 424
RGB_fps = 30

class Kinect(object):
    def __init__(self, debug=False):
        self.debug = debug
        self._scale_depth = True
        self._scale_rgb_colors = False

        openni2.initialize("/usr/lib")

        dev = openni2.Device.open_any()
        if debug:
            print(dev.get_device_info())

        dev.set_image_registration_mode(True)
        dev.set_depth_color_sync_enabled(True)

        # create depth stream
        self.depth_stream = dev.create_depth_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                    resolutionX=Depth_ResX,
                                                    resolutionY=Depth_ResY,
                                                    fps=Depth_fps,
                                                    ))
        depth_sensor_info = self.depth_stream.get_sensor_info()

        self.max_depth = self.depth_stream.get_max_pixel_value()
        self.min_depth = 0

        if self.debug:
            for itm in depth_sensor_info.videoModes:
                print(itm)
            print("Min depth value: {}".format(self.min_depth))
            print("Max depth value: {}".format(self.max_depth))
        
        self.rgb_stream = dev.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                    resolutionX=RGB_ResX,
                                                    resolutionY=RGB_ResY,
                                                    fps=RGB_fps,
                                                    ))

        if self.debug:
            rgb_sensor_info = self.rgb_stream.get_sensor_info()
            for itm in rgb_sensor_info.videoModes:
                print(itm)
    
    def start(self):
        self.depth_stream.start()
        self.rgb_stream.start()

    def stop(self):
        self.depth_stream.stop()
        self.rgb_stream.stop()
    
    def scale_depth(self, use_scale):
        self._scale_depth = use_scale
    
    def scale_rgb(self, use_rgb_scale):
        self._scale_rgb_colors = use_rgb_scale
    
    def get_pointcloud(self):
        before = time.time()

        depth_frame = self.depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        rgb_frame = self.rgb_stream.read_frame()
        rgb_data = rgb_frame.get_buffer_as_triplet()

        depth_array = np.frombuffer(depth_data, dtype=np.uint16)
        if self.debug:
            print(depth_array.shape)
        
        depth_array = np.frombuffer(depth_data, dtype=np.uint16)
        depth_image = depth_array.reshape(Depth_ResY, Depth_ResX)

        depth_image[depth_image==0x0000] = 0 #max_depth
        depth_image[depth_image==0x7ff8] = 0
        depth_image[depth_image==0xfff8] = 0
        max_depth = np.max(depth_image)
        depth_image = (depth_image - self.min_depth) / (self.max_depth - self.min_depth)
        if self.debug:
            print("Depth min: {}".format(np.min(depth_image)))
            print("Depth max: {}".format(max_depth))
        
        color_array = np.frombuffer(rgb_data, dtype=np.uint8)
        color_image = color_array.reshape(Depth_ResY, Depth_ResX, -1)
        
        index_x = np.tile(np.arange(0, Depth_ResX), (Depth_ResY, 1))
        index_y = np.tile(np.arange(0, Depth_ResY), (Depth_ResX, 1)).T

        valid_points = np.count_nonzero(depth_image != 0.0)
        valid_rgb = color_image[depth_image != 0.0]
        valid_depth = depth_image[depth_image != 0.0]
        valid_x = index_x[depth_image != 0.0]
        valid_y = index_y[depth_image != 0.0]

        point_cloud = np.zeros((valid_points, 6))
        point_cloud[:,0] = valid_x
        point_cloud[:,1] = -valid_y
        point_cloud[:,2] = valid_depth
        point_cloud[:,3:] = valid_rgb

        if self._scale_depth:
            depth_scale = ((np.abs(np.mean(point_cloud[:,0])) + np.abs(np.mean(point_cloud[:,1]))) / 2) / np.mean(point_cloud[:,2])
            point_cloud[:,2] *= -depth_scale
            if self.debug:
                print("Depth scale: {}".format(depth_scale))

        if self._scale_rgb_colors:
            point_cloud[:,3:] /= 255.0

        after = time.time()
        if self.debug:
            print("Pointcloud calculation time: {} secs".format(after-before))

        return point_cloud

