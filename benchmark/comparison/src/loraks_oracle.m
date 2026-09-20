function loraks_oracle(in_file, out_file, loraks_dir)
% LORAKS_ORACLE  Run the authors' LORAKS reconstruction on data written by Julia.
%
% Driver for the reference implementation distributed with
%
%   [1] T. H. Kim, J. P. Haldar. LORAKS Software Version 2.0: Faster Implementation and
%       Enhanced Capabilities. University of Southern California, Los Angeles, CA,
%       Technical Report USC-SIPI-443, May 2018.
%   [2] J. P. Haldar. Low-Rank Modeling of Local k-Space Neighborhoods (LORAKS) for
%       Constrained MRI. IEEE Transactions on Medical Imaging 33:668-681, 2014.
%   [3] J. P. Haldar, J. Zhuo. P-LORAKS: Low-Rank Modeling of Local k-Space Neighborhoods
%       with Parallel Imaging Data. Magnetic Resonance in Medicine 75:1499-1514, 2016.
%
% The LORAKS package itself is NOT redistributed with MriReconstructionToolbox: its licence
% allows educational / research / non-profit use only, which is incompatible with this
% repository's MIT licence. Download it from http://mr.usc.edu/download/LORAKS2/ and unpack it
% into benchmark/comparison/original_implementations/LORAKS2 (gitignored). This file is the
% only part of the bridge that lives here, and it calls the package without copying from it.
%
% `in_file` must contain:
%   kData    N1 x N2 x Nc complex, zero-filled at the unsampled positions
%   kMask    N1 x N2 real, 1 where sampled
%   rank     scalar rank for the non-convex penalty of Eq. (2) of [1]
%   radius   k-space neighbourhood radius R
%   ltype    'C', 'S' or 'W'
%   max_iter scalar iteration cap
%
% `out_file` receives `recon` (N1 x N2 x Nc reconstructed k-space) and `elapsed` (seconds).

addpath(loraks_dir);
S = load(in_file);

% lambda = 0 selects the data-consistency-constrained formulation of Eq. (6) in [1], which is
% the formulation MRT's calibrationless `StructuredLowRank` corresponds to: the acquired
% samples are held fixed and only the missing ones are filled in.
lambda = 0;
alg = 2;   % multiplicative half-quadratic, no FFT approximation -- the accurate reference

t0 = tic;
recon = P_LORAKS(S.kData, S.kMask, double(S.rank), double(S.radius), S.ltype, ...
                 lambda, alg, 1e-4, double(S.max_iter));
elapsed = toc(t0); %#ok<NASGU>

save(out_file, 'recon', 'elapsed', '-v7.3');
end
