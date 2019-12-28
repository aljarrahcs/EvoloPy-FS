### EvoloPy-FS: An Open-Source Nature-Inspired Optimization Framework in Python for Feature Selection

### Beta Version ###

EvoloPy-FS is a python open-source optimization framework that includes several well-regarded swarm intelligence (SI) algorithms. It is geared toward feature selection optimization problems. It is an easy to use, reusable, and adaptable framework. The objective of developing EvoloPy-FS is providing a feature selection engine to help researchers even those with less knowledge in SI in solving their problems and visualizing rapidresults with a less programming effort. That is why the orientation of this work wasto build an open-source, white-box framework, where algorithms and data structures are being explicit, transparent, and publicly available. EvoloPy-FS comes to continueour path for building an integrated optimization environment, which was started bythe original EvoloPy for global optimization problems, then EvoloPy-NN for training multilayer perception neural network, and finally the new EvoloPy-FS for features election optimization. EvoloPy-FS is freely hosted on (www.evo-ml.com) with ahelpful documentation. 


The full list of implemented optimizers is available here https://github.com/7ossam81/EvoloPy/wiki/List-of-optimizers


## Features
- Six nature-inspired metaheuristic optimizers are implemented.
- The implimentation uses the fast array manipulation using [`NumPy`] (http://www.numpy.org/).
- Matrix support using [`SciPy`'s] (https://www.scipy.org/) package.
- More optimizers are comming soon.
 

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy` and `SciPy` for
you.

- If you are installing EvoloPy-FS onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.
- If you are installing onto Ubuntu or Debian and using Python 3 then
  this will pull in all the dependencies from the repositories:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/aljarrahcs/EvoloPy-FS.git


## Quick User Guide
EvoloPy-FS Framework contains six datasets (All of them are obtainied from UCI repository). 
The main file is the main.py, which considered the interface of the framewok. In the main.py you 
can setup your experiment by selecting the optmizers, the datasets, number of runs, number of iterations, number of neurons
and population size. The following is a sample example to use the EvoloPy-NN framework.
To choose PSO optimizer for your experiment, change the PSO flag to true and others to false.

Select optimizers:    
PSO= True  
MVO= False  
GWO = False  
MFO= False  
.....


After that, Select datasets:

datasets=["BreastCancer", "iris"]

The folder datasets in the repositoriy contains 3 binary datasets (All of them are obtained from UCI repository).

To add new dataset:
- Put your dataset in a csv format (No header is required)
- Normalize/Scale you dataset ([0,1] scaling is prefered) #(Optional)
- Place the new datset files in the datasets folder.
- Add the dataset to the datasets list in the main.py (Line 18).
  
  For example, if the dastaset name is Seed, the new line  will be like this:
        
        datasets=["BreastCancer", "iris", "Seed"]


Change NumOfRuns, PopulationSize, and Iterations variables as you want:
    
    For Example: 

    NumOfRuns=10  
    PopulationSize = 50  
    Iterations= 1000

Now your experiment is ready to go. Enjoy!  

The results will be automaticly generated in excel file called Experiment which is concatnated with the date and time of the experiment.
The results file contains the following measures:


    Optimizer	Dataset	objfname	Experiment	startTime	EndTime	ExecutionTime	trainAcc	testAcc
    Optimizer: The name of the used optimizer
    Dataset: The name of the dataset.
    objfname: The objective function/ Fitness function
    Experiment: Experiment ID/ Run ID.
    startTime: Experiment's starting time
    EndTime: Experiment's ending time
    ExecutionTime : Experiment's executionTime (in seconds)
    trainAcc: Trainig Accuracy
    testAcc: Trainig Accuracy
    Iter1	Iter2 Iter3... : Convergence values (The bjective function values after every iteration).	
    Iter1	Iter2 Iter3... : Convergence values (The number of features after every iteration).	
    

## Contribute
- Issue Tracker: https://github.com/aljarrahcs/EvoloPy-FS/issues  
- Source Code: https://github.com/aljarrahcs/EvoloPy-FS

## Support

Use the [issue tracker](https://github.com/aljarrahcs/EvoloPy-FS/issues). 

## Citation Request:

Please include these citations if you plan to use this Framework:

- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An
open-source nature-inspired optimization framework in python for feature selection. In Evolutionary
Machine Learning Techniques, pages 131–173. Springer, 2020

- Hossam Faris, Ibrahim Aljarah, Sayedali Mirjalili, Pedro Castillo, and J.J Merelo. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python". In Proceedings of the 8th International Joint Conference on Computational Intelligence - Volume 3: ECTA,ISBN 978-989-758-201-1, pages 171-177.

- Hossam Faris, Ali Asghar Heidari, Al-Zoubi Ala’M, Majdi Mafarja, Ibrahim Aljarah, Mohammed
Eshtay, and Seyedali Mirjalili. Time-varying hierarchical chains of salps with random weight networks
for feature selection. Expert Systems with Applications, page 112898, 2019.

- Majdi Mafarja,Ibrahim Aljarah, Ali Asghar Heidari, Hossam Faris, Philippe Fournier-Viger,Xiaodong Li, and Seyedali Mirjalili.  Binary dragonfly optimization for feature selection using time-varying transfer functions. Knowledge-Based Systems, 161:185–204, 2018.

- Ibrahim Aljarah, Majdi Mafarja, Ali Asghar Heidari, Hossam Faris, Yong Zhang, and Seyedali Mirjalili. Asynchronous accelerating multi-leader salp chains for feature selection. Applied Soft Computing, 71:964–979, 2018.

- Hossam Faris, Majdi M Mafarja, Ali Asghar Heidari,Ibrahim Aljarah, Al-Zoubi Ala’M, Seyedali Mirjalili, and  Hamido Fujita.   An  efficient binary salp swarm algorithm  with crossover  scheme for feature selection problems. Knowledge-Based Systems, 154:43–67, 2018.


