# LiouvilleDMD
This repository contains MATLAB code for dynamic mode decomposition in continuous time using Liouville operators and occupation kernels. The code implements the methods described in

 - https://link.springer.com/article/10.1007/s00332-021-09746-w (Liouville DMD - `lib/LiouvilleDMD.m`),
 - https://arxiv.org/abs/2101.02646 (Higher order Liouville DMD - `lib/SecondOrderLiouvilleDMD.m`),
 - https://epubs.siam.org/doi/10.1137/22M1475892 (Provably Convergent Liouville DMD - `lib/ConvergentLiouvilleEigenDMD.m`),
 - https://arxiv.org/abs/2101.02620 (Control Liouville DMD - `lib/ControlLiouvilleDMD.m`),
 - https://arxiv.org/abs/2309.09817 (Control Koopman DMD - `lib/ControlKoopmanDMD.m`)
 - https://arxiv.org/abs/2106.00106 (Kernel Perspective on Koopman DMD - `lib/KoopmanDMD.m`), and
 - https://arxiv.org/abs/1411.2260 (Kernel DMD by Williams et al. - `lib/WilliamsKDMD.m`).
 
Several examples that illustrate the use of the methods are also available in the Examples folder.

Compatibility: MATLAB R2020b or newer.
