import sys
import glob
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":


    fig = plt.figure()

    ims = []

    flist = list(sys.argv[1:])
    flist.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    for f in flist:
        image = misc.imread(f)
        if image is None: continue
        ims.append((plt.imshow(image),))

    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=100, blit=True)

    ani.save('foo.gif', writer='imagemagick')
