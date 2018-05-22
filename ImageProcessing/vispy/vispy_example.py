import numpy as np
import vispy.scene
from vispy.scene import visuals

from Kinect import Kinect


class VispyExample(object):
    def __init__(self, callback):        
        keys = { "q": vispy.app.quit,
                 "esc": vispy.app.quit,
                 "Space": self.key_space
        }
        
        self.callback = callback

        self.canvas = vispy.scene.SceneCanvas(keys=keys, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        self.do_update = True

    def run(self):
        timer = vispy.app.Timer()
        timer.connect(self.draw_from_callback)
        timer.start(1, -1) #interval, iterations -1: don't stop
        vispy.app.run()
        self.stop()
    
    def stop(self):
        self.kinect.stop()

    def key_space(self):
        if self.do_update:
            print("Disabling update")
            self.do_update = False
        else:
            print("Enabling update")
            self.do_update = True
    
    def draw_from_callback(self, ev):
        if not self.do_update:
            return

        p = self.callback()
        if p is None:
            return

        pos = p[:,:3]
        colors = p[:,3:]

        self.scatter.set_data(pos, edge_color=colors, size=1)
            
        self.view.camera = 'turntable'
