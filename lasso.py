#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from numba import jit, double, float_, int_, float64
from mpi4py import MPI
import scipy.io as sio
import numpy as np
from numpy import dot
from numpy import linalg as LA
from scipy.linalg import cholesky, cho_solve, solve
import sys


#@jit
def soft_threshold(v, sigma):
    """ shrinkage function """
    mask = np.abs(v) <= sigma;
    v[mask] = 0;
    mask = np.abs(v) > sigma;
    v[mask] = v[mask] - np.sign(v[mask])*sigma;
    return v;

#@jit
def objective(A, b, lmbd, z):
    """ 
    calculate objective function (dual form)
    minimize \lambda * ||x||_1 + 0.5 * ||Ax - b||_2^2 
    """ 
    return 0.5*LA.norm(dot(A,z)-b, 2)**2 + lmbd*LA.norm(z, 1);

#@profile
def main(argv):
    MAX_ITER = 50 
    RELTOL = 1e-2
    ABSTOL = 1e-4

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # determine current running process
    size = comm.Get_size() # total number of processes
    N = float(size)
    #rank = 0
    
    dataCenterDir = "."
    if len(argv) == 2:
        big_dir = argv[1]
    else:
        big_dir = "data"

    comm.Barrier()

    #%% Read in local data
    # 
	# Subsystem n will look for files called An.dat and bn.dat
	# in the current directory; these are its local data and 
	# do not need to be  visible to any other processes. Note that
	# m and n here refer to the dimensions of the *local* coefficient matrix.
	# 
    try:
        # Read A
        s = "%s/%s/A%d.dat" % (dataCenterDir, big_dir, rank + 1)
        print "[%d] reading %s" % (rank, s)
        A = sio.mmread(s)

        # Read b
        s = "%s/%s/b%d.dat" % (dataCenterDir, big_dir, rank+1)
        print "[%d] reading %s" % (rank, s)
        b = sio.mmread(s)
        b.shape = b.shape[0]

        # Read xs
        #s = "%s/%s/xs%d.dat" % (dataCenterDir, big_dir, rank + 1)
        #print "[%d] reading %s" % (rank, s);
        #xs = sio.mmread(s)
        #xs.shape = xs.shape[0]

        (m, n) = A.shape
        skinny = (m > n)

        rho = 1.0;

        nxstack  = 0;
        nystack  = 0;
        prires   = 0;
        dualres  = 0;
        eps_pri  = 0;
        eps_dual = 0;

        Atb = dot(A.T, b);

        lmbd = 0.5;
        if rank == 0:
            print "using lambda: %.4f" % (lmbd,);

        # precalculate (alpha + mu/N) I + beta AAt
        if skinny :
            L = dot(A.T, A) + rho*np.eye(n);
            L = cholesky(L, lower=True)

        else:
            L = dot(A, A.T)/rho  + np.eye(m);
            L = cholesky(L, lower=True)

        # Main ADMM solver loop 
        startAllTime = MPI.Wtime()

        iter = 0;
        if rank == 0:
            print "%3s %10s %10s %10s %10s %10s" % ("#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");		
            

        x  = np.zeros(n);
        u  = np.zeros(n);
        z  = np.zeros(n);
        r  = np.zeros(n);
        send = np.zeros(3);
        recv = np.zeros(3);
        while iter < MAX_ITER:
            startTime = MPI.Wtime()

            # u-update: u = u + x - z */
            u = u + x-z;

            # x-update: x = (A^T A + rho I) \ (A^T b + rho z - y) */
            q = Atb + rho*(z-u);

            if skinny:
                # x = U \ (L \ q) */
                x = cho_solve((L, True), q);
            else:
                # x = q/rho - 1/rho^2 * A^T * (U \ (L \ (A*q))) */
                p = cho_solve((L, True), dot(A, q));
                x = q/rho - dot(A.T, p)/(rho**2);

            #
            # Message-passing: compute the global sum over all processors of the
            # contents of w and t. Also, update z.
            #

            w = x + u;

            send[0] = dot(r, r);
            send[1] = dot(x, x);
            send[2] = dot(u, u)/(rho**2);

            zprev = np.copy(z);

            # could be reduced to a single Allreduce call by concatenating send to w
            comm.Allreduce(w, z, op=MPI.SUM); 
            comm.Allreduce(send, recv, op=MPI.SUM); 

            prires  = np.sqrt(recv[0]);  #/* sqrt(sum ||r_i||_2^2) */
            nxstack = np.sqrt(recv[1]);  #/* sqrt(sum ||x_i||_2^2) */
            nystack = np.sqrt(recv[2]);  #/* sqrt(sum ||y_i||_2^2) */

            z = z/N;
            z = soft_threshold(z, lmbd/(N*rho));

            # Termination checks */

            # dual residual */
            dualres = np.sqrt(N) * rho * LA.norm(z-zprev,2); #/* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

            # compute primal and dual feasibility tolerances */
            eps_pri  = np.sqrt(n*N)*ABSTOL + RELTOL * np.fmax(nxstack,
                                                           np.sqrt(N)*LA.norm(z,2));
            eps_dual = np.sqrt(n*N)*ABSTOL + RELTOL * nystack;

            if rank == 0:
                print "%3d %10.4f %10.4f %10.4f %10.4f %10.4f" % (iter, 
                        prires, eps_pri, dualres, eps_dual, objective(A, b, lmbd, z));

            if prires <= eps_pri and dualres <= eps_dual:
                break;

            # Compute residual: r = x - z */
            r = x - z;

            iter+=1;
            # End while loop ========================================

        # Have the master write out the results to disk 
        if rank == 0:
            endAllTime = MPI.Wtime()
            print "Elapsed time is: %lf " % (endAllTime - startAllTime,);

            f = open("data/pysolution.dat", "w");
            f.write("x = \n");
            f.write(np.array_str(x));
            f.write("\n");
            f.close();

    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    #%%

#%% Entry
if __name__ == "__main__":
    main(sys.argv)
    #main(['', '../Data/Gaussian/4'])


