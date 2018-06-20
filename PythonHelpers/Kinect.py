import numpy as np
import threading
import cv2
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
import time

class Kinect(threading.Thread):
    def __init__(self, color=False,
                       depth=False,
                       ir=False,
                       registered=False,
                       undistorted=False,
                       pointcloud=True,
                       debug=False):
        threading.Thread.__init__(self)
        self._debug = debug
        self._isrunning = True

        self._save_color = color
        self._save_depth = depth
        self._save_ir = ir
        self._save_registered = registered
        self._save_undistorted = undistorted
        self._save_pointcloud = pointcloud

        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None

        self._color_image = None
        self._ir_image = None
        self._depth_image = None
        self._registered_image = None
        self._undistorted = None
        self._point_cloud = None

        self._color_image_lock = threading.Lock()
        self._ir_image_lock = threading.Lock()
        self._depth_image_lock = threading.Lock()
        self._registered_image_lock = threading.Lock()
        self._undistorted_lock = threading.Lock()
        self._point_cloud_lock = threading.Lock()

        self._index_x = None
        self._index_y = None

        self._scale_rgb = False
        self._size_scale = 1.0
    
    def scale_rgb(self, do_scale):
        self._scale_rgb = do_scale
    
    def scale_size(self, scale):
        self._size_scale = scale

    def _get_index_x_y(self, res_x, res_y):
        if self._index_x is None:
            self._index_x = np.tile(np.arange(0, res_y), (res_x, 1))
        if self._index_y is None:
            self._index_y = np.tile(np.arange(0, res_x), (res_y, 1)).T
        
        return self._index_x, self._index_y
    
    def get_color_image(self):
        with self._color_image_lock:
            return self._color_image
    
    def _set_color_image(self, im):
        with self._color_image_lock:
            self._color_image = im

    def get_ir_image(self):
        with self._ir_image_lock:
            return self._ir_image
    
    def _set_ir_image(self, im):
        with self._ir_image_lock:
            self._ir_image = im
    
    def get_depth_image(self):
        with self._depth_image_lock:
            return self._depth_image
    
    def _set_depth_image(self, im):
        with self._depth_image_lock:
            self._depth_image = im
    
    def get_registered_image(self):
        with self._registered_image_lock:
            return self._registered_image
    
    def _set_registered_image(self, im):
        with self._registered_image_lock:
            self._registered_image = im
    
    def get_undistorted_image(self):
        with self._undistorted_lock:
            return self._undistorted
    
    def _set_undistorted_image(self, im):
        with self._undistorted_lock:
            self._undistorted = im
    
    def get_pointcloud(self):
        with self._point_cloud_lock:
            return self._point_cloud
    
    def _set_pointcloud(self, im):
        with self._point_cloud_lock:
            self._point_cloud = im

    def stop(self):
        self._isrunning = False

    def run(self):
        self._isrunning = True

        try:
            from pylibfreenect2 import OpenGLPacketPipeline
            pipeline = OpenGLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenCLPacketPipeline
                pipeline = OpenCLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                pipeline = CpuPacketPipeline()
        if self._debug:
            print("Packet pipeline:", type(pipeline).__name__)

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if self._debug:
            print("Number of devices: {}".format(num_devices))
        
        serial = self.fn.getDeviceSerialNumber(0)
        device = self.fn.openDevice(serial, pipeline=pipeline)
        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
        
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        
        if self._debug:
            print("Init done")
        
        device.start()

        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

        params = device.getIrCameraParams()
        self.cx = params.cx
        self.cy = params.cy
        self.fx = params.fx
        self.fy = params.fy

        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)

        while self._isrunning:
            frames = listener.waitForNewFrame()

            color = frames["color"]
            ir = frames["ir"]
            depth = frames["depth"]

            registration.apply(color, depth, undistorted, registered, bigdepth=None, color_depth_map=None)

            if self._save_color:
                self._set_color_image(cv2.cvtColor(color.asarray(), cv2.COLOR_BGR2RGB))
            if self._save_depth or self._save_pointcloud:
                depth_arr = depth.asarray()
                if self._save_depth:
                    self._set_depth_image(depth_arr)
            if self._save_ir:
                self._set_ir_image(ir.asarray())
            if self._save_registered or self._save_pointcloud:
                reg = cv2.cvtColor(registered.asarray(np.uint8), cv2.COLOR_BGR2RGB)
                if self._save_registered:
                    self._set_registered_image(reg)
            if self._save_undistorted:
                und = undistorted.asarray(np.uint8)
                self._set_undistorted_image(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
            if self._save_pointcloud:
                self.compute_pointcloud(reg, depth_arr)
            
            listener.release(frames)

        if self._debug:
            print("Stopping device")
        device.stop()
        if self._debug:
            print("Closing device")
        device.close()
        if self._debug:
            print("Device stopped and closed")
    
    def wait_for_init(self):
        WAIT_TIME = 0.1
        if self._save_color:
            while self.get_color_image() is None:
                time.sleep(WAIT_TIME)
        
        if self._save_depth:
            while self.get_depth_image() is None:
                time.sleep(WAIT_TIME)
        
        if self._save_ir:
            while self.get_ir_image() is None:
                time.sleep(WAIT_TIME)
        
        if self._save_registered:
            while self.get_registered_image() is None:
                time.sleep(WAIT_TIME)
        
        if self._save_undistorted:
            while self.get_undistorted_image() is None:
                time.sleep(WAIT_TIME)
        
        if self._save_pointcloud:
            while self.get_pointcloud() is None:
                time.sleep(WAIT_TIME)

    def compute_pointcloud(self, color, depth):

        index_x, index_y = self._get_index_x_y(*depth.shape)

        valid_points = np.count_nonzero(depth != 0.0)
        valid_rgb = color[depth != 0.0]

        valid_depth = depth[depth!= 0.0] / 255.0
        valid_x = valid_depth * (index_x[depth != 0.0] - self.cx) / self.fy
        valid_y = valid_depth * (index_y[depth != 0.0] - self.cy) / self.fx

        point_cloud = np.zeros((valid_points, 6))
        point_cloud[:,0] = -valid_x * self._size_scale
        point_cloud[:,1] = valid_y * self._size_scale
        point_cloud[:,2] = valid_depth * self._size_scale
        if self._scale_rgb:
            point_cloud[:,3:] = valid_rgb / 256.0
        else:
            point_cloud[:,3:] = valid_rgb

        self._set_pointcloud(point_cloud)

if __name__ == "__main__":
    k = Kinect(debug=True)
    k.start()

    
    k.wait_for_init()
    p = k.get_pointcloud()
    print(p.shape)
    time.sleep(2)
    k.stop()
    k.join()
    print("Done!")