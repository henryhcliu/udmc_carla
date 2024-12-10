# A Unified Decision-Making and Control Framework for Urban Autonomous Driving with Traffic Rule Compliance
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/henryhcliu/udmc_carla/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0-brightgreen)](https://github.com/henryhcliu/udmc_carla/releases/tag/v1.0)

> This autonomous driving framework (named UDMC) is based on optimization methods, so it is light-weighted, interpretable, and adaptable to various driving scenarios.

> Live demo [https://www.youtube.com/watch?v=jftTsf1jXjU](https://www.youtube.com/watch?v=jftTsf1jXjU).

## Table of Contents
* [General Info](#general-information)
* [Usage](#usage)
* [Project Status](#project-status)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
* [License](#license)


## General Information
<!-- Insert the udmc_structure.png figure using 0.5\width width -->

<div style="text-align:center;">
  <img src="img/udmc_structure.png" alt="The Structure of the UDMC" width="80%" />
</div>

- This project contains the necessary code for our presented Unified Decision-Making and Control (UDMC) scheme for urban driving. 
- The purpose of this project is to build a uniformed, versatile and light-weighted autonomous driving framework for both decision-making and motion control.
- This driving framework settles the complex traffic rule compliance problem using the artificial potential field (APF), rather than if-else commands in most current rule-based methods.
- We tested it on varieties of scenarios (car following, overtaking, roundabout, intersection, T-junction, etc.) and `CARLA Town05 Benchmark`.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


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
### Run the UDMC with `CARLA Town05` Benchmark
**Change to the branch of `town05short`**, and then run the following command:
```Shell
cd leaderboard/scripts
# Run the CARLA Town05 Benchmark automatically
./local_evaluation.sh 
```
Wait until the program is finished, and the evaluation result on the Town05 Short Benchmark can be inspected in the `results` folder.

## Project Status
Project is: _complete_ 


## Acknowledgements
- Thanks to the [CasADi](https://web.casadi.org/docs/#nonlinear-programming).
- Thanks to the [CARLA](https://carla.readthedocs.io/en/0.9.14/).
- Thanks to the authors of [VAD](https://arxiv.org/abs/2303.12077)

## Contact
Created by [@henryhcliu](https://henryhcliu.github.io) - feel free to contact me!

## License
This project is open source and available under the [MIT License](https://opensource.org/license/mit/).