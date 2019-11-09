#AUTHOR: Mohanad ALBUGHDADI
#TERRANIS
#DATE 6th AUGUST 2019


FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
#RUN add-apt-repository ppa:fkrull/deadsnakes
RUN apt-get update

RUN apt-get install -y build-essential python3.5 python3.5-dev python3-pip python3.5-venv
RUN apt-get install -y git
RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get update
RUN apt-get install libgtk2.0-dev





# update pip
RUN python3.5 -m pip install pip --upgrade
RUN python3.5 -m pip install wheel





RUN python3.5 -m pip install -U numpy
RUN python3.5 -m pip install -U scipy==1.2.0
RUN python3.5 -m pip install -U scikit-learn==0.20.0
RUN python3.5 -m pip install -U scikit-image==0.15.0
RUN python3.5 -m pip install -U matplotlib==3.0.3
RUN python3.5 -m pip install -U pandas==0.24.0
RUN python3.5 -m pip install -U tifffile==0.15.1
RUN python3.5 -m pip install -U xgboost
RUN python3.5 -m pip install opencv-contrib-python
RUN python3.5 -m pip install six
RUN python3.5 -m pip install geopandas==0.4.0
RUN python3.5 -m pip install rasterstats==0.13.0
RUN python3.5 -m pip install plotly==3.6.1



RUN python3.5 -m pip install setuptools==41.0.1
RUN python3.5 -m pip install tensorflow
RUN python3.5 -m pip install keras





RUN python3.5 -m pip install ipykernel
RUN python3.5 -m pip install jupyter


RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN apt-get install -y python3-gdal
RUN apt-get install -y python3-pyproj
RUN apt-get install -y libxt6
RUN apt-get install -y libxpm4
RUN apt-get install -y libxmu6
RUN apt-get install -y curl
