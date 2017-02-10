# To Do List
## Agent modelling

### New models to implement

* Basci's agent (Basci, 1999)

* Kindler's agent (Kindler, 2016). Similar to Basci's one.

* Backward RL model.

* Super Forward RL agent with forward propagation up to 5 rounds.

* Total Gogol Imitator (c).

### On existing models

* For FRL (Forward RL): reduce number of free parameters (merging parameters for initial q values)

## Economic structure

* Make heterogeneous economies (different type of agents)


## Back up

* Keep track of agent choice, agent good, partner choice, partner type, partner good (same as exp)


## Plans

* Forward RL model: correlations for between gamma and speculation frequency

* Make play best model with best parameters for each agent against KW agents and check 
if level of speculation is the same.

## Preliminary results

### Fitting

* Duffy is the worst model (bug or not bug?). But Marimon should have very bad scores also
 (as Marimon's agent can not recognize their own consumption good)

* Forward RL model has the best maximum likelihood for every single subject.

* For FRL ('fwel'),  alpha is often weak but gamma strong, although gamma is varying across subjects.
 
* In FRL and SRL (Strategic RL), strong 'priors' (fuck Bayes, we prefer RL). 


### Artificial economies

* Only pure KW can achieve teh speculative equilibrium

## Experimental 

* Code Germain's task in Python

* Recode Aurelien's one for implementing KW instead of Iwai

*  Design the experiment