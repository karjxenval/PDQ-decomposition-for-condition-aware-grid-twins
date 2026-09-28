# Scientific plan

## R1 — Real structural sparse-to-dense recovery
Use Ponte Moesa distributed fibre data as dense measured truth. Hide 80–95% of spatial channels. Compare POD/SVD, recovery-aware subspace design, QR, leverage, random and deliberately poor sensor sets. Stratify by damage state.

## R2 — Real electrical measurement recovery
Use KIOS/GridEye synchronized PMU channels. Use chronological 60/20/20 train/validation/test splits. Hide channels at deployment and reconstruct them from the retained channels. Report error, exact amplification, projection error, rank and smallest singular value.

## R3 — Large distribution external stress
Use selected LBNL micro-PMU sites first; freeze exact raw file names and time windows before results. Then scale to the full archive if needed.

## R4 — Uganda realism audit
Use ERA official purchases/sales/losses, maximum demand, network length, customer mix and capacity/generation as constraints on the Uganda topology experiment. It remains “synthetic response on real topology” until measured feeder/node data are obtained.

## Punchline criterion
A real-data experiment should answer not only “what is the mean error?” but also “why did it fail?” Separate representation-limited, sensor-identifiability-limited, noise-limited and regularization-limited cases using the paper’s diagnostics.
