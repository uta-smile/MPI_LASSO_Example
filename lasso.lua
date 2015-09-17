#!/usr/bin/env th

th = require('torch')
require 'mpiT'
require 'mm'

printf = function(s,...)
    return io.write(s:format(...) .. '\n')
end -- function

function soft_threshold(v, sigma)
   -- shrinkage function
   mask = th.abs(v):le(sigma)
   v[mask] = 0
   mask = th.abs(v):gt(sigma)
   v[mask] = v[mask] - th.sign(v[mask])*sigma
   return v
end

function objective(A, b, lmdb, z)
    -- calculate objective function (dual form)
    -- minimize \lambda * ||x||_1 + 0.5 * ||Ax - b||_2^2 
    return 0.5 * th.norm(A * z - b, 2)^2 + lmbd*th.norm(z, 1)
end

function cholesky(A)
    L = th.potrf(A)
    L = L:t()
    return L
end

function cho_solve(L, b)
    y2 = torch.cat(b, b, 2) -- Ugly way to make it trtrs function work
    t = torch.trtrs(y2, L, 'L')
    xh2 = torch.trtrs(t, L:t())
    return xh2[{{}, 1}]
end

function main(argv)
    MAX_ITER = 50 
    RELTOL = 1e-2
    ABSTOL = 1e-4

    mpiT.Init()
    _rank = th.IntStorage(1)
    comm = mpiT.COMM_WORLD
    mpiT.Comm_rank(comm,_rank)
    rank = _rank[1]
    _size = th.IntStorage(1)
    mpiT.Comm_size(comm,_size)
    size = _size[1]
    N = size

    print(rank)

    dataCenterDir = "."
    if #argv == 2 then
        big_dir = argv[1]
    else
        big_dir = "data"
    end

    mpiT.Barrier(comm)
    -- Read in local data
    -- 
	-- Subsystem n will look for files called An.dat and bn.dat
	-- in the current directory; these are its local data and 
	-- do not need to be  visible to any other processes. Note that
	-- m and n here refer to the dimensions of the *local* coefficient matrix.
	-- 
    if true then
        -- Read A 
        s = string.format("%s/%s/A%d.dat", dataCenterDir, big_dir, rank + 1)
        printf("[%d] reading %s", rank, s)
        A = mm.import(s)
        --print(A)

        -- Read b
        s = string.format("%s/%s/b%d.dat", dataCenterDir, big_dir, rank + 1)
        printf("[%d] reading %s", rank, s)
        b = mm.import(s)

        m = A:size(1)
        n = A:size(2)
        skinny = (m > n)

        rho = 1.0

        nxstack  = 0;
        nystack  = 0;
        prires   = 0;
        dualres  = 0;
        eps_pri  = 0;
        eps_dual = 0;
        
        Atb = A:t() * b

        lmbd = 0.5;
        if rank == 0 then
            printf("using lambda: %.4f", lmbd);
        end

        -- precalculate (alpha + mu/N) I + beta AAt
        if skinny then
            L = A:t()*A + rho*th.eye(n)
            L = cholesky(L)
        else
            L = A*A:t()/rho + th.eye(m)
            L = cholesky(L)
        end

        -- Main ADMM solver loop 
        startAllTime = mpiT.Wtime()

        iter = 0;
        if rank == 0 then
            printf("%3s %10s %10s %10s %10s %10s", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");		
        end

        x  = th.zeros(n);
        u  = th.zeros(n);
        z  = th.zeros(n);
        r  = th.zeros(n);
        send = th.zeros(3);
        recv = th.zeros(3);
        while iter < MAX_ITER do
            startTime = mpiT.Wtime()

            -- u-update: u = u + x - z */
            u = u + x-z;

            -- x-update: x = (A^T A + rho I) \ (A^T b + rho z - y) */
            q = Atb + (z-u)*rho;

            if skinny then
                -- x = U \ (L \ q) */
                x = cho_solve(L, q);
            else
                -- x = q/rho - 1/rho^2 * A^T * (U \ (L \ (A*q))) */
                p = cho_solve(L, A*q);
                x = q/rho - A:t()*p/(rho^2);
            end

            --
            -- Message-passing: compute the global sum over all processors of the
            -- contents of w and t. Also, update z.
            --

            w = x + u;

            send[1] = r:dot(r);
            send[2] = x:dot(x);
            send[3] = u:dot(u)/(rho^2);

            zprev = z:clone();

            -- could be reduced to a single Allreduce call by concatenating send to w
            --comm.Allreduce(w, z, op=MPI.SUM); 
            --comm.Allreduce(send, recv, op=MPI.SUM); 
            mpiT.Allreduce(w:storage(), z:storage(), w:numel(),
                           mpiT.DOUBLE, mpiT.SUM, mpiT.COMM_WORLD)
            mpiT.Allreduce(send:storage(), recv:storage(), send:numel(),
                           mpiT.DOUBLE, mpiT.SUM, mpiT.COMM_WORLD)

            prires  = math.sqrt(recv[1]);  --/* sqrt(sum ||r_i||_2^2) */
            nxstack = math.sqrt(recv[2]);  --/* sqrt(sum ||x_i||_2^2) */
            nystack = math.sqrt(recv[3]);  --/* sqrt(sum ||y_i||_2^2) */

            z = z/N;
            z = soft_threshold(z, lmbd/(N*rho));

            -- Termination checks */

            -- dual residual */
            dualres = math.sqrt(N) * rho * th.norm(z-zprev,2); --/* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

            -- compute primal and dual feasibility tolerances */
            eps_pri  = math.sqrt(n*N)*ABSTOL + RELTOL * math.max(nxstack,
                                                           math.sqrt(N)*th.norm(z,2));
            eps_dual = math.sqrt(n*N)*ABSTOL + RELTOL * nystack;

            if rank == 0 then
                printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f", iter, 
                    prires, eps_pri, dualres, eps_dual, objective(A, b, lmbd, z));
            end

            if prires <= eps_pri and dualres <= eps_dual then
                break;
            end

            -- Compute residual: r = x - z */
            r = x - z;

            iter= iter + 1;
        end -- End while loop ========================================

        -- Have the master write out the results to disk 
        if rank == 0 then
            endAllTime = mpiT.Wtime()
            printf("Elapsed time is: %f ", endAllTime - startAllTime);

            local f = io.open("data/thsolution.dat", "w+")
            f:write("x = \n");
            f:write(tostring(x));
            f:write("\n");
            f:close();
        end
    end

    mpiT.Finalize()
end

main(arg)
