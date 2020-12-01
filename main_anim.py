
import sys, os, glob
from time import time, sleep
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import animation

import simulation

def init():

    global fig
    sc_plot.set_data(zeros((Nx,Ny)))
    # oned_plot.set_data((arange(Nx)+0.5)*dx, zeros(Nx))

    # U = asarray(sim.U[3:-3,3:-3,0,:])
    # b_plot.set_UVC(U[::N/arrws,::N/arrws,7],
    #                U[::N/arrws,::N/arrws,8])
    return  sc_plot, #oned_plot, #b_plot,

def animate(k):

    global fig,title, sim, evry,dx, step, stepmax,traj
    # print 'step =', k*evry

    # if step>1: sys.exit()

    t = time()
    sim.integrate(evry)
    # print 't = ', sim.time
    # print 'dt =', sim.dt
    step += evry

    U = asarray(sim.U[3:-3,3:-3,0,:])


    # coup = asarray(sim.Coup[:,:,0,:])
    # save('coup.npy', coup)
    #
    # if k==0:
    #     prts = sim.particles
    #     save('prts.npy', prts)

    # traj[:,k+1,:] = prts[:]
    # if k==5:
    #     save('traj.npy', traj)
    # save('U_'+str(step), U)

    Ek = 0.5*(U[...,MX]**2 + U[...,MY]**2 + U[...,MZ]**2)/U[...,RHO]
    Em = 0.5*(U[...,BX]**2 + U[...,BY]**2 + U[...,BZ]**2)
    # Bm = sqrt(2*Em)

    p = (gam-1) * (U[...,EN] - Em - Ek) #- 0.5 * beta
    # ppd = U[...,IPD]*sqrt(2*Em)
    # ppl = 3*p - 2*ppd
    # ppl = U[...,IPL]/(2*Em)*U[...,RHO]**2
    # p = 0.333 * (2 * ppd + ppl)# - 0.5 * beta

    # Ipd = U[...,IPD].mean()
    # Ipl = U[...,IPL].mean()
    # ppl0 = ppl.mean()
    # ppd0 = ppd.mean()

#    S = p/U[...,RHO]**(5./3)
    # delta = (U[3:-3,5,6]*Bm[3:-3,5] - 0.5*p[3:-3,5])/p[3:-3,5]

    # Bm = sqrt(2*Em)
    # bx = U[...,BX]/Bm
    # by = U[...,BY]/Bm
    # bz = U[...,BZ]/Bm

    # Emm = Em.mean()
    # Ekm = Ek.mean()
    # Etm = U[...,EN].mean() - Emm - Ekm

    # print 'Em =', Emm
    # print 'dB2 =', 0.5*(U[...,BZ]**2).mean()
    # print 'Ek =', Ekm
    # print 'Et =', Etm
    # print 'Etot =', Ekm+Emm+Etm

    sc_plot.set_array(p.T[::-1])
    # oned_plot.set_data((arange(Nx)+0.5)*dx, (ppd-ppl)/(2*Em))#ppd-0.5*beta)#)##
    title.set_text(r'$t={}\tau_A$'.format(round(sim.time,2)))

    # b_plot.set_UVC(bx[::N/arrws,::N/arrws], by[::N/arrws,::N/arrws])

    # sleep(0.1)

    print 't_comp = ', time()-t, '\n'
    # print Ipd, Ipl, ppd0,ppl0

    return sc_plot,title, #oned_plot, #b_plot,

#------------------------------------------------------------------------------


sim = Simulation()
sim.init('./params.cfg')

dt = sim.dt

Lx,Ly,Lz = sim.dim_phys
Nx,Ny,Nz = sim.dim_cells
dx = Lx/Nx
N = int(1./dx)

params = sim.params
gam = params['gam']
beta = params['beta']

RHO, MX,MY,MZ, EN,         PSC, BX,BY,BZ = range(9)
# RHO, MX,MY,MZ, EN, SE,     PSC, BX,BY,BZ = range(10)
#
# RHO, MX,MY,MZ, EN, IPD,    PSC, BX,BY,BZ = range(10)
# RHO, MX,MY,MZ, IPL,IPD,    PSC, BX,BY,BZ = range(10)
#
# RHO, MX,MY,MZ, EN, IPD,SE, PSC, BX,BY,BZ = range(11)
# RHO, MX,MY,MZ, IPL,IPD,SE, PSC, BX,BY,BZ = range(11)

U = asarray(sim.U[:,:,0,:])
# prts = sim.particles
# save('U_0.npy', U)
# save('prts_0.npy', prts)


if not os.path.exists('out'):
    os.mkdir('out')
    os.chdir('out')
else:
    for f in glob.glob('out/*'): os.remove(f)
    os.chdir('out')


step=0

evry=20
stepmax=10

# vmin = -1.1
# vmax = 0.1

vmin = 0.03
vmax = 3.2

# traj=zeros((shape(prts)[0],stepmax+1,shape(prts)[1]))
# traj[:,0,:]=prts[:]

# vmin=0.8
# vmax=2.2

#----------------------------------------------------------------


fig = plt.figure(figsize=(8,8))
ax1= fig.add_subplot(111, xlim=[0, Lx],
                          ylim=[0, Ly])
title = ax1.text(0.76,-0.47,'', fontsize=24)

extent = [0, Lx, 0, Ly]
sc_plot = ax1.imshow(zeros((Nx,Ny)).T[::-1], vmin=vmin, vmax=vmax,
                extent=extent, interpolation="nearest",
                aspect=1, cmap='jet')#,cmap='YlOrRd')
plt.colorbar(sc_plot)

# oned_plot, = ax1.plot((arange(Nx)+0.5)*dx, U[3:-3,0,RHO])
# ax1.set_ylim(ymin=vmin,ymax=vmax)


# arrws = 48
# X,Y = mgrid[0:Lx:1./arrws, 0:Ly:1./arrws]
# b_plot = ax1.quiver(X,Y, U[::N/arrws,::N/arrws, 7],
#                          U[::N/arrws,::N/arrws, 8],
#                    color='w', headwidth=2, scale=30.)#/sqrt(0.5*beta))

anim = animation.FuncAnimation(fig, animate, frames=stepmax, init_func=init,
                               interval=10, blit=True)
# anim.save('alfven_disruption.mp4', fps=30)
plt.show()
