import tkinter as tk
from simulation import Simulation


def run(datafile):
	simulation = Simulation.from_track("data/軌道座標點.txt")
	simulation.train("data/" + datafile, learning_rate=0.01)
	simulation.run()
	simulation.save_data()
	simulation.plot()


root = tk.Tk()
tk.Button(root, text="Simulation by 4D data", command=lambda: run("train4dAll.txt")).pack(pady=10)
tk.Button(root, text="Simulation by 6D data", command=lambda: run("train6dAll.txt")).pack(pady=10)
root.mainloop()
