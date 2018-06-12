import numpy as np
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

class Kinect(object):
    def __init__(self, debug=False):
        self.debug = debug
        
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
        if self.debug:
            print("Packet pipeline:", type(pipeline).__name__)
        
        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        if self.debug:
            print("Number of devices: {}".format(num_devices))
        
        serial = fn.getDeviceSerialNumber(0)
        self.device = fn.openDevice(serial, pipeline=pipeline)
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
        
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        
        if debug:
            print("Init done")
        
    def start(self):
        if self.debug:
            print("Starting device")
        self.device.start()
        #if self.debug:
        #    print("Device startet")
            
        #self.registration = Registration(self.device.getIrCameraParams(),
        #                                 self.device.getColorCameraParams())
        
        
        # Optinal parameters for registration
        # Including if needed in later implementation
        """
        need_bigdepth = False
        need_color_depth_map = False
        bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
        color_depth_map = np.zeros((424, 512),  np.int32).ravel() if need_color_depth_map else None
        """

    def stop(self):
        if self.debug:
            print("Stopping device")
        """
        self.device.stop()
        if self.debug:
            print("Closing device")
        self.device.close()
        if debug:
            print("Device stopped and closed")
        """
    
    def get_pointcloud(self):
        frames = self.listener.waitForNewFrame()
        color = frames["color"]
        #ir = frames["ir"]
        depth = frames["depth"]
        
        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)
        
        self.registration.apply(color, depth, undistorted, registered)