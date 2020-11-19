
import sys, os, glob
from time import time, sleep
from numpy import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from shutil import rmtree

import h5py

from vars2plot import get_var_funcs


# Parse command line arguments.

parser = argparse.ArgumentParser()
# parser.add_argument("-slc_ax", type=int,
#             help="choose axis perpendicular to 2d slice: x=0, y=1, z=2")
# parser.add_argument("-slc_pos", type=int,
#             help="choose slice cell index along the chosen axis")
parser.add_argument("-mn1", type=float,
            help="minimum value for plot 1")
parser.add_argument("-mx1", type=float,
            help="maximum value for plot 1")
parser.add_argument("-mn2", type=float,
            help="minimum value for plot 2")
parser.add_argument("-mx2", type=float,
            help="maximum value for plot 2")
parser.add_argument("-cmap", help="colormap name")
parser.add_argument("--mov", action="store_true", help="save a movie")
parser.add_argument("-frames", type=int,
            help="total number of frames for video")
parser.add_argument("-fps", type=int, help="frame rate of the movie")
parser.add_argument("-d", help="relative path of output directory")
parser.add_argument("-n", help="output file name: {name}_{number}.hdf5")

args = parser.parse_args()

vmin1 = args.mn1 if args.mn1 != None else None
vmax1 = args.mx1 if args.mx1 != None else None
vmin2 = args.mn2 if args.mn2 != None else None
vmax2 = args.mx2 if args.mx2 != None else None
cmap = args.cmap if args.cmap != None else None
frames = args.frames if args.frames != None else 300
fps = args.fps if args.fps != None else 15
out_path = args.d if args.d != None else 'out'
out_name = args.n if args.n != None else 'slc'


# Obtain functions calculating variables to plot.

var_funcs = get_var_funcs()
Nvar = len(var_funcs)


# Read simulation parameters from the first output file.

os.chdir(out_path)

f0 = h5py.File(out_name+'_000.hdf5', 'r')

L = f0.attrs['size']
N = shape(var_funcs[0](f0))[::-1]
dx = L[0]/N[0]

# Set max/min limits for plots if not set by user.

A1 = var_funcs[0](f0)
if vmin1 == None:
    vmin1 = A1.min()
if vmax1 == None:
    vmax1 = A1.max()
if Nvar==2:
    A2 = var_funcs[1](f0)
    if vmin2 == None:
        vmin2 = A2.min()
    if vmax2 == None:
        vmax2 = A2.max()

f0.close()

# if N[1]>1 and N[2]>1:
#     dim='3d'
# el
if N[1]>1:
    dim='2d'
else:
    dim='1d'

# dim='1d'

# if dim=='3d':
#
#     slc_ax = args.slc_ax if args.slc_ax != None else 2
#     slc_pos = args.slc_pos if args.slc_pos != None else 0
#     if slc_pos < 0 or slc_pos > N[slc_ax]:
#         print 'slice position is outside the box limits!'
#         sys.exit()
#
# elif dim=='2d':
#     slice_ax=2
#     slice_pos=0


# Set up the figure.

if Nvar==1:
    fig = plt.figure(figsize=(7.5,6.5))
    ax1= fig.add_subplot(111)
else:
    fig = plt.figure(figsize=(15,6.5))
    ax1= fig.add_subplot(121)
    ax2= fig.add_subplot(122)


if dim=='1d':

    plot1, = ax1.plot([],[])
    ax1.set_xlim(xmin=0,xmax=L[0])
    ax1.set_ylim(ymin=vmin1,ymax=vmax1)
    if Nvar==2:
        plot2, = ax2.plot([],[])
        ax2.set_xlim(xmin=0,xmax=L[0])
        ax2.set_ylim(ymin=vmin2,ymax=vmax2)

else:

    extent = [0, L[0], 0, L[1]]

    plot1 = ax1.imshow(zeros((N[1],N[0])), vmin=vmin1, vmax=vmax1,
                    extent=extent, interpolation="nearest",
                    aspect=1, cmap=cmap)#,cmap='YlOrRd')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(plot1, cax=cax)

    if Nvar==2:
        plot2 = ax2.imshow(zeros((N[1],N[0])), vmin=vmin2, vmax=vmax2,
                        extent=extent, interpolation="nearest",
                        aspect=1, cmap=cmap)#,cmap='YlOrRd')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(plot2, cax=cax)

plt.subplots_adjust(left=0.105, wspace=0.25, right=0.86)

    # arrws = 48
    # X,Y = mgrid[0:Lx:1./arrws, 0:Ly:1./arrws]
    # b_plot = ax1.quiver(X,Y, U[::N/arrws,::N/arrws, 7],
    #                          U[::N/arrws,::N/arrws, 8],
    #                    color='w', headwidth=2, scale=30.)#/sqrt(0.5*beta))

if dim=="1d":
    title_pos = 0.05*L[0], vmin1+0.93*(vmax1-vmin1)
else:
    title_pos = 0.05*L[0], 0.93*L[1]

props = dict(boxstyle='square,pad=0.11', facecolor='w',
            edgecolor='grey',alpha=0.4)
title1 = ax1.text(0.025, 0.9, '', transform=ax1.transAxes,
                 fontsize=18, bbox=props)
# if Nvar==2:
#     title2 = ax2.text(0.76*L[0], vmin2+0.47*(vmax2-vmin2),
#                     '', fontsize=24)

n=0

#-------------------------------------------------------------------------------

def init():

    if dim=='1d':
        plot1.set_data((arange(N[0])+0.5)*dx, zeros(N[0]))
        if Nvar==2:
            plot2.set_data((arange(N[0])+0.5)*dx, zeros(N[0]))
    else:
        plot1.set_data(zeros((N[0],N[1])))
        if Nvar==2:
            plot2.set_data(zeros((N[0],N[1])))

    # U = asarray(sim.U[3:-3,3:-3,0,:])
    # b_plot.set_UVC(U[::N/arrws,::N/arrws,7],
    #                U[::N/arrws,::N/arrws,8])
    if Nvar==1:
        return  title1, plot1,
    else:
        return title1, plot1, plot2,


def animate(k):

    global n

    if os.path.isfile(out_name+'_{:03d}.hdf5'.format(n+1)):

        print 'frame: '+out_name+'_{:03d}.hdf5'.format(n)

        with h5py.File(out_name+'_{:03d}.hdf5'.format(n), 'r') as f:

            t = f.attrs['t']
            A1 = var_funcs[0](f)
            if Nvar==2:
                A2 = var_funcs[1](f)

        if dim=='1d':
            plot1.set_data((arange(N[0])+0.5)*dx, A1[0,:])
            if Nvar==2:
                plot2.set_data((arange(N[0])+0.5)*dx, A2[0,:])
        else:
            plot1.set_array(A1[::-1])
            if Nvar==2:
                plot2.set_array(A2[::-1])

        title1.set_text(r'$t={}\tau_A$'.format(round(t,2)))

        # b_plot.set_UVC(bx[::N/arrws,::N/arrws], by[::N/arrws,::N/arrws])

        # sleep(0.05)

        n+=1

    if Nvar==1:
        return plot1, title1,
    else:
        return plot1, plot2, title1,

#----------------------------------------------------------------

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               interval=50, blit=True, frames = frames)
if args.mov:

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)

    if os.path.isdir('out_frames'):
        rmtree('out_frames')

    anim.save('out.mp4', writer=writer)

if not args.mov:
    plt.show()
