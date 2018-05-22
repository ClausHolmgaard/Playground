from openni import openni2
from openni import _openni2 as c_api
import numpy as np

Depth_ResX = 512
Depth_ResY = 424
Depth_fps = 30
RGB_ResX = 512
RGB_ResY = 424
RGB_fps = 30

class Kinect(object):
    def __init__(self, debug=False):
        self.debug = debug

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
    
    def get_pointcloud(self):
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
            print("Depth max: {}".format(np.max(depth_image)))
        
        color_array = np.frombuffer(rgb_data, dtype=np.uint8)
        color_image = color_array.reshape(Depth_ResY, Depth_ResX, -1) / 255.0
        
        num_points = Depth_ResX * Depth_ResY

        point_cloud = np.zeros((num_points, 6))

        counter = 0
        for x in range(Depth_ResX):
            for y in range(Depth_ResY):
                # Skip invalid points
                if depth_image[y, x] != 0.0:
                    point_cloud[counter, 0] = x
                    point_cloud[counter, 1] = -y
                    point_cloud[counter, 2] = -depth_image[y, x] * 255.0 * 10
                    if depth_image[y, x] != 0:

                        point_cloud[counter, 3] = color_image[y, x, 0]
                        point_cloud[counter, 4] = color_image[y, x, 1]
                        point_cloud[counter, 5] = color_image[y, x, 2]
                    else:
                        point_cloud[counter, 3] = 0
                        point_cloud[counter, 4] = 0
                        point_cloud[counter, 5] = 0
                    counter += 1
                    
        return point_cloud

