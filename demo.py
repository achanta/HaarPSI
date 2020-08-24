'''
The following code show how to use the HaarPSI metric implelentation in the file haarPSI.py
'''

import numpy as np
from PIL import Image
from timeit import default_timer as timer
from haarPSI import compute_HaarPSI_similarity


def test():

    ref = Image.open("./bee.png")
    ref = np.asarray(ref)
    img = Image.open("./bee180.png")
    img = np.asarray(img)

    print(ref.shape,img.shape)

    start = timer()
    val = compute_HaarPSI_similarity(ref,img)
    end = timer()

    print(val,"(in",end-start, " seconds)")



test()
