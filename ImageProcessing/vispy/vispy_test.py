import numpy as np
import vispy.scene
from vispy.scene import visuals

class VispyTest(object):
    def __init__(self):
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()