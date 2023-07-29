# Unified Decision-Making and Control Framework for Urban Autonomous Driving
> This autonomous driving framework is based on optimization methods, so it is light-weighted, interpretable and adaptable to various driving scenarios.
> Live demo [_here_](https://youtu.be/Jn2BrnhnCoU). <!-- If you have the project hosted somewhere, include the link here. -->

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#driving-demo-in-the-first-person-view)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- This project contains the necessary code for our presented UDMC. We tested it on varieties of scenarios, such as multi-lane adaptive cruise control, roundabout driving, intersection crossing, and so on.
- The purpose of this project is to build a versatile and light-weighted autonomous driving framework from V2X perception to motion control, settling the problems of the hierarchical rule-based and optimization-based framework (complex pipeline design, low computational efficiency), and the problems of the end-to-end framework (not interpretable, time-consuming training process, low safety guarantee). And this driving framework settles the traffic rule compliance problem using the artificial potential field, rather than if-else commands in most current literature.
- This project improves the adaptability and computational efficiency of autonomous driving to various urban driving conditions under the guarantee of safety with traffic rule compliance.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Ubuntu 20.04
- Python 3.8 with `requirements.txt`
- CasADi 3.6.3
- CARLA 0.9.13


## Features
List the ready features here:
- light-weighted
- interpretable
- various scenarios adaptable
- traffic rules compliance


## Driving Demo in the First-Person View


https://github.com/henryhcliu/udmc_carla/assets/44318132/e23f58be-ef6e-4e17-93ef-3ef4fec1c03a


## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment
```Shell
git clone https://github.com/henryhcliu/udmc_carla.git
cd udmc_carla
conda create -n udmc_carla python=3.8
conda activate udmc_carla
pip3 install -r requirements.txt
```

Download and setup CARLA 0.9.13
```Shell
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
sudo apt-get update # Update the Debian package index
sudo apt-get install carla-simulator=0.9.13 # Install the 0.9.13 CARLA version
cd /opt/carla-simulator # Open the folder where CARLA is installed
./CarlaUE4.sh
```

## Usage
Before running the following codes, CARLA Simulator should be active.
```Shell
cd /opt/carla-simulator # Open the folder where CARLA is installed
./CarlaUE4.sh
```
In another terminal:
```Shell
cd udmc_carla # Open the folder where this repository exists
```
### Run the UDMC with different driving scenario settings and surrounding vehicle spawn options
```Shell 
# crossroad driving with randomly generated surrounding vehicles
python udmc_main.py crossroad True

# multilane ACC driving with randomly generated surrounding vehicles
python udmc_main.py multilaneACC True

# roundabout driving with randomly generated surrounding vehicles
python udmc_main.py roundabout True
```
If you want to spawn surrounding vehicles with certain spawnpoints (to test the performance of different methods), please change the last argument to `False`.
### Run the Parameter Identification of the vehicle dynamics model
```Shell
python param_est_using_slsqp.py
```
This program prints the estimated parameters to the terminal.
### Run the Interpolation-based Gaussian Process Regression training process and store the GPR model to file
```Shell
python IGPR_predict_sv_wps.py
```

## File structure
If you want to modify this driving system to adapt to specific applications, please refer to the structure of this repository.
```bash
-data # contains the Potential Functions' value and the control input during autonomous driving
-images # contains the pictures captured by a camera mounted on the Ego Vehicle with T_s time step
-official # some official examples provided by CARLA (with our modification)
-scripts # core implementation of the UDMC
    - env.py # interact with CARLA, it includes spawn vehicles, initial visualization, and CARLA environment, etc.
    - others_agent.py # Be in charge of the behavior of other vehicles, like following the lane and changing lane
    - vehicle_obs.py # the IDAC core with traffic rules (lane keeping, not running to solid lane markings, not running a red light, etc)
    - x_v2x_agent.py # implement the main function, such as acc, overtaking, and parking
-spawnpoints # contains the spawn points for the surrounding vehicles when not using `random_spawn` mode
-utils # some third-party or commonly-used function
-fms_main.py # run this code to execute the same simulation with udmc_main.py, but the Ego Vehicle uses Finite State Machine to control its motion
-IGPR_predict_sv_wps.py # use this code to train an IGPR model for surrounding vehicles' motion from 15 pieces of history state record to 10 pieces of future state prediction.
-udmc_main.py # main entrance of this paper, run it from the command line with an augment (crossroad, multilaneACC, roundabout,...), before that you need to launch CarlaUE4 following the instruction above.
```

## Project Status
Project is: _complete_ 


## Acknowledgements
- Thanks to the CasADi [tutorial](https://web.casadi.org/docs/#nonlinear-programming)


## Contact
Created by [@henryhcliu](https://www.linkedin.com/in/haichaoliu) - feel free to contact me!


<!-- Optional -->
## License -->
This project is open source and available under the [MIT License](https://opensource.org/license/mit/).

<!-- You don't have to include all sections - just the one's relevant to your project -->
