# Deep Reinforcement Learning 

Repository implementing deep reinforcement learning and conducting experiments on the OpenAI environments

## Models

* Deep Q-Network (vanilla) [`dqn_plain.py`](models/dqn_plain.py)
* Deep Q-Network (with target network) [`dqn.py`](models/dqn.py)
* Double Deep Q-Network [`ddqn.py`](models/ddqn.py)

## Setup

```bash
# Clone repo
git clone git@github.com:eugeneyan/deep_rl.git && cd deep_rl

# Create conda environment
conda env create -f=environment.yml

# Activate environment
source activate deep_rl
```

### box2D errors

If a box2D error is encountered (e.g., AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'), please follow the steps below:  

```bash
pip uninstall Box2D box2d-py
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python setup.py clean
python setup.py build
python setup.py install
```

More details here: <https://github.com/openai/gym/issues/100>

## Running experiments

Running experiments is a simple as 

```bash
# To run experiment on cartpole environment
python cartpole_runner.py

# To run experiment on lunar lander environment 
python lunarlander_runner.py
```

### Usage

```bash
python lunarlander_runner.py --help

usage: lunarlander_runner.py [-h] [--model MODEL] [--render-env RENDER_ENV]
                             [--render-freq RENDER_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         DeepRL model to use (options: dqn_plain, dqn, ddqn;
                        default: ddqn)
  --render-env RENDER_ENV
                        Whether to render the environment (default: y)
  --render-freq RENDER_FREQ
                        How frequently to render the env (default: 500)
                        --render-env must be set to "y" to render environment
```

## Industrial applications of Deep RL

* J. Gao, R. Evans. “DeepMind AI Reduces Google Data Centre Cooling Bill by 40%”, https://deepmind.com/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-40/, 18 Mar 2018

* J. Gao, “Machine Learning Applications for Data Center Optimization”, http://static.googleusercontent.com/media/www.google.com/en//about/datacenters/efficiency/internal/assets/machine-learning-applicationsfor-datacenter-optimization-finalv2.pdf, 18 Mar 2018

* J. Zhao, “Deep Reinforcement Learning for Sponsored Search Real-time Bidding”, arXiv:1803.00259, 2018

## References

* C. Watkins “Learning from delayed rewards”, PhD Thesis, University of Cambridge, England, 1989

* V. Mnih, et al. “Human-level control through deep reinforcement learning.” Nature 518(7540):529-533, 2015.

* V. Minh, et al. “Playing atari with deep reinforcement learning.” CoRR abs/1312.5602, 2013.

* H. van Hasselt, A. Guez, and D. Silver. “Deep reinforcement learning with double Q-Learning.” CoRR abs/1509.06461, 2015

