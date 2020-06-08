# User-User Collaborative Filtering Using Singular Value Decomposition (SVD)
This directory contains the deployable as well as training code for User-User Collaborative Filtering Using Singular Value Decomposition (SVD).
This directory is directly deployable to heroku

# API Reference
The app only expose a single `POST` API to get recommendations:

`<app-host>/api/getRecs`

Parameters:
(required) `userId`: mongodb document id for user
(optional) `limit`: number, if passed only specified number of recommendations are returned, if not all recommendations are returned 

## Dependencies
Flask
Gunicorn
Pickle
Surprise