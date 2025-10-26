from simulation import Simulation


simulation = Simulation.from_track("data/軌道座標點.txt")
simulation.train("data/train6dAll.txt", learning_rate=0.01)
simulation.run()
