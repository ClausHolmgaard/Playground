import numpy as np
import vispy.scene
from vispy.scene import visuals

class Vispy(object):
    def __init__(self, callback, interval=0.3, point_size=0.001, edge_width=0.003, symbol='o'):        
        keys = { "q": vispy.app.quit,
                 "esc": vispy.app.quit,
                 "Space": self.key_space
        }
        
        self.callback = callback
        self.interval = interval

        self.canvas = vispy.scene.SceneCanvas(keys=keys, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        self.do_update = True

        self._point_size = point_size
        self._edge_width = edge_width
        self._symbol = symbol

    def run(self):
        timer = vispy.app.Timer()
        timer.connect(self.draw_from_callback)
        timer.start(self.interval, -1) #interval, iterations -1: don't stop
        vispy.app.run()

    def key_space(self):

        if self.do_update:
            self.do_update = False
            print("Getting Frame.")
            p = self.callback()

            if p is None:
                return

            pos = p[:,:3]
            colors = p[:,3:]

            self.scatter.set_data(pos, edge_color=colors, size=1)
                
            self.view.camera = 'turntable'
        else:
            self.do_update = True
    
    def draw_from_callback(self, ev):
        if not self.do_update:
            return

        p = self.callback()

        if p is None:
            return

        pos = p[:,:3]
        colors = p[:,3:] / 255.0

        self.scatter.set_data(pos,
                              face_color=colors,
                              edge_color=colors,
                              size=self._point_size,
                              edge_width=self._edge_width,
                              symbol=self._symbol,
                              scaling=True)

        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        #self.view.camera.distance = 1

if __name__ == "__main__":
    from Kinect import Kinect

    

    k = Kinect()
    k.start()
    k.wait_for_init()

    v = Vispy(k.get_pointcloud)
    #pc = k.get_pointcloud()

    v.run()

    k.stop()

