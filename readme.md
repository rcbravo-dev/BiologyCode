# Draft of README.md by ChatGPT


# Self-Replicating Programs Simulation

This project explores the emergence of self-replicating programs in a computational environment, inspired by the research presented in "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction" by Blaise Agüera y Arcas et al.

## Introduction

This repository contains code that simulates the evolution of self-replicating programs in a genetic pool. The simulation is based on principles from the study of artificial life (ALife) and investigates how simple computational entities can develop complex behaviors, such as self-replication, without any explicit fitness function or predefined goals.

### Key Features:
- A genetic server that manages a pool of "tapes" or programs.
- Self-replication and mutation mechanisms to explore evolutionary dynamics.
- Configurable parameters for the number of epochs, mutation rate, pool size, and more.
- Ability to track and analyze the evolution of programs over time.

## Installation and Setup

To run the simulation, you need Python 3.x and the following libraries:

- `numpy`
- `asyncio`
- `argparse`
- `aiohttp`

Install the required libraries using the example below:

```bash
pip install numpy
```

### Running the Simulation

1. **Server Setup:**

Run the server with the following command:

```bash
python3 server -hs localhost -p 8080 -e 100 -ps 512 -t 64 -m 0.05 -en 'experiment_name' --track_gen
```

- **Arguments:**
  - `-hs` or `--host`: Host IP or "localhost".
  - `-p` or `--port`: Server port (default: 8080).
  - `-e` or `--epoch`: Number of epochs to run (default: 10).
  - `-ps` or `--pool_size`: Size of the genetic pool (default: 100).
  - `-t` or `--tape_length`: Length of each tape (default: 64).
  - `-m` or `--mutation_rate`: Mutation rate (default: 0.05).
  - `-en` or `--experiment_name`: Name for the experiment run.
  - `--track_gen`: Enable tracking generations (optional).

2. **Client Interaction:**

Use `client.py` to interact with the server an programatically combine tapes from the genetic pool managed by the server.

```bash
python3 -m client --work -hs 'localhost' -p 8080 -c alice
```

- **Arguments:**
  - `-hs` or `--host`: Host IP or "localhost".
  - `-p` or `--port`: Server port (default: 8080).
  - `-c` : Client name.

3. **Normal Operations:**
Once the server has started and the client is being served tapes, stdout is set to display the dominant gene in the genetic pool at the completion of each epoch. The string representation of the tape is displayed followed by a fitness score. A 1.0 fitness score indicates 100% replication, whereas a 0.5 score represents no significant replication. 

```bash
Dominant Gene: w[www[.w{<w{<.........]]...]......w.....]...........r.........{{, fitness: 1.0, epochs: 6
Dominant Gene: w[www[.w{<.{<.........]]...]............]...........r........>{., fitness: 0.767578125, epochs: 5
Dominant Gene: w[www[<w{<.{<.........]]...]............].....................{<, fitness: 0.572265625, epochs: 4
Dominant Gene: w[www[<w{<.{<.........]]]..]............].....................{], fitness: 0.556640625, epochs: 3
Dominant Gene: w[www[<{{<.{<<........]]...]......w.....].....................{<, fitness: 0.513671875, epochs: 2
Dominant Gene: w[www[<{{<.{<<........]]...]............].....................{<, fitness: 0.501953125, epochs: 1
Dominant Gene: w[www[<w{<.{<<........]]]..]............].....................{], fitness: 0.55078125, epochs: 0
```


## Attribution

This project is inspired by the research paper:

**Agüera y Arcas, B., Alakuijala, J., Evans, J., Laurie, B., Mordvintsev, A., Niklasson, E., Randazzo, E., & Versari, L. (2024). "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction".**  
Available from [Google Scholar link or publisher].

## Further Development

- Experiment with different programming languages or substrates for the genetic programs.
- Modify mutation rates, pool sizes, or selection mechanisms to explore different evolutionary outcomes.
- Analyze the dynamics of self-replication using novel complexity metrics.

## Contributing

We welcome contributions! Please fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [GPLv3 License](LICENSE).

---

Would you like to modify or expand any specific sections?