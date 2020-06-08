# User-User Collaborative Filtering Using Singular Value Decomposition (SVD)
This directory contains code for a small framework to tune, train and evaluate recommendation engine using SVD. 
Code will find optimized parameters for SVD and run evaluation against both SVD and TunedSVD against found parameters.

*`recsys` script is modified at the moment to run against data with mongodb objectIDs.*

## Directory Info
`datasets` directory contains dataset csv files.
- `products.csv` contains scrapped product names from foodpanda
- `ratings.csv` contains scrapped ratings from food panda
- `final_products.csv` contains data from `products.csv` but is formatted to use against mongodb documents
- `final_ratings.csv` contains data from `ratings.csv` is formatted to use agains mongodb documents

`models` directory will save trained models for both SVD and Tuned SVD if `saveModel` flag is set to True in `RecSys` code.

`results` contains screenshot of evaluation results for both SVD and SVDTuned 

## Dependencies
surpriseLib
numpy
pandas
pickle