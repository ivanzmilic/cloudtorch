## Instructions for developing the code further. 

- We are starting from the main idea of cloudtorch, summarized (by you) in cloudtorch_summary.pdf
- We are planning the approach first. 
- DO NOT run any tests before we actually start writing code, plan is mostly fixed, by me. 
- Ask whatever is unclear. 
- Suggest if my approach has an obvious mistake. 

## Goal: Fit x,y,lambda,t cube with a model that has background + two clouds 

Background - fit via PCA decomposition of the the basis given by the last several snapshots. 
Cloud model - same as now. 

## Steps: 

After each step check with me

All code written in pytorch

1) Write the code for PCA decomposition from the last N timesteps of the cube. 
2) Construct a model of background with ~ 10 free parameters that uses the basis to formulate background
3) Construct a composite model of background + two clouds 
4) OPTIONAL, I need your feedback: formulate the unknown model as a neural field (PINN)
5) Fit the whole x,y,lambda,t to the data using the PINN.