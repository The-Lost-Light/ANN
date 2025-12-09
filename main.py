import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt

import lib
from hopfield import HopfieldNetwork


class HopfieldGUI:
	def __init__(self, root):
		self.root = root
		self.root.title("Hopfield Network")
		self.root.geometry("1280x720")

		self.dataset_var = tk.StringVar(value="Basic")
		self.current_test_data = []
		self.current_train_data = []
		self.current_pattern_index = 0
		self.p_width = 0
		self.lines_height = 0

		control_frame = ttk.LabelFrame(root, text="Configuration")
		control_frame.pack(fill="x", padx=10, pady=5)

		ttk.Label(control_frame, text="Dataset:").pack(side="left", padx=5)
		dataset_combo = ttk.Combobox(control_frame, textvariable=self.dataset_var, state="readonly", width=10)
		dataset_combo["values"] = ("Basic", "Bonus")
		dataset_combo.pack(side="left", padx=5)
		dataset_combo.bind("<<ComboboxSelected>>", self.reset_index)

		ttk.Button(control_frame, text="Run", command=self.run_prediction).pack(side="left", padx=20)
		display_frame = tk.Frame(root)
		display_frame.pack(expand=True, fill="both", padx=10, pady=5)

		for i in range(2):
			display_frame.rowconfigure(i, weight=1)
		for i in range(3):
			display_frame.columnconfigure(i, weight=1)

		self.canvases = {}
		self.create_panel(display_frame, 0, 0, "1. Training Pattern", "train", bg="#f0f0f0")
		self.create_panel(display_frame, 0, 1, "3. Sync (Theta=0)", "sync_no_theta")
		self.create_panel(display_frame, 0, 2, "4. Sync", "sync_theta")
		self.create_panel(display_frame, 1, 0, "2. Testing Pattern", "test")
		self.create_panel(display_frame, 1, 1, "5. Async (Theta=0)", "async_no_theta")
		self.create_panel(display_frame, 1, 2, "6. Async", "async_theta")

		nav_frame = tk.Frame(root)
		nav_frame.pack(fill="x", pady=5)
		ttk.Button(nav_frame, text="< Prev Pattern", command=self.prev_pattern).pack(side="left", padx=20)
		self.lbl_index = ttk.Label(nav_frame, text="Pattern: 0")
		self.lbl_index.pack(side="left", padx=10)
		ttk.Button(nav_frame, text="Next Pattern >", command=self.next_pattern).pack(side="left", padx=20)

	def create_panel(self, parent, row, col, title, key, bg="white"):
		frame = ttk.LabelFrame(parent, text=title)
		frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
		canvas = tk.Canvas(frame, bg=bg)
		canvas.pack(expand=True, fill="both", padx=3, pady=3)
		self.canvases[key] = canvas

	def reset_index(self, event=None):
		self.current_pattern_index = 0
		self.lbl_index.config(text=f"Pattern: {self.current_pattern_index}")

	def prev_pattern(self):
		if self.current_pattern_index > 0:
			self.current_pattern_index -= 1
			self.run_prediction()

	def next_pattern(self):
		self.current_pattern_index += 1
		self.run_prediction()

	def get_dataset_config(self):
		dataset_name = self.dataset_var.get()
		if dataset_name == "Basic":
			lines_height = 12
		else:
			lines_height = 10

		train_data, test_data, p_width = lib.load_patterns(dataset_name, lines_height)

		max_idx = len(test_data) - 1
		if self.current_pattern_index > max_idx:
			self.current_pattern_index = max_idx
		if self.current_pattern_index < 0:
			self.current_pattern_index = 0

		return dataset_name, lines_height, p_width, train_data, test_data

	def draw_pattern(self, canvas, pattern, width_chars, height_chars, is_target=False):
		canvas.delete("all")
		if pattern is None:
			return

		c_width = canvas.winfo_width()
		c_height = canvas.winfo_height()

		if width_chars == 0 or height_chars == 0:
			return
		cell_w = c_width / width_chars
		cell_h = c_height / height_chars
		cell_size = min(cell_w, cell_h) * 0.9

		offset_x = (c_width - cell_size * width_chars) / 2
		offset_y = (c_height - cell_size * height_chars) / 2

		for r in range(height_chars):
			for c in range(width_chars):
				idx = r * width_chars + c
				if idx < len(pattern):
					val = pattern[idx]
					x0 = offset_x + c * cell_size
					y0 = offset_y + r * cell_size
					x1 = x0 + cell_size
					y1 = y0 + cell_size

					color = "black" if val == 1 else "white"
					outline = "blue" if is_target else "#ccc"
					canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=outline)

	def run_prediction(self):
		try:
			dataset_name, lines_height, p_width, train_data, test_data = self.get_dataset_config()
			self.p_width = p_width
			self.lines_height = lines_height
			self.lbl_index.config(text=f"Pattern: {self.current_pattern_index + 1} / {len(test_data)}")
			input_pattern = test_data[self.current_pattern_index]
			target_pattern = (
				train_data[self.current_pattern_index] if self.current_pattern_index < len(train_data) else None
			)

			self.draw_pattern(self.canvases["train"], target_pattern, p_width, lines_height, is_target=True)
			self.draw_pattern(self.canvases["test"], input_pattern, p_width, lines_height)

			scenarios = [
				{"name": "Sync (No Theta)", "key": "sync_no_theta", "theta": False, "async": False},
				{"name": "Sync (With Theta)", "key": "sync_theta", "theta": True, "async": False},
				{"name": "Async (No Theta)", "key": "async_no_theta", "theta": False, "async": True},
				{"name": "Async (With Theta)", "key": "async_theta", "theta": True, "async": True},
			]

			results = []
			for sc in scenarios:
				model = HopfieldNetwork(lines_height * p_width)
				model.train(train_data, threadhold=sc["theta"])
				output = model.predict(input_pattern, asynchronous=sc["async"])
				self.draw_pattern(self.canvases[sc["key"]], output, p_width, lines_height)
				results.append((sc["name"], output))

			fig, axes = plt.subplots(2, 3, figsize=(12, 8))
			fig.suptitle(f"Dataset: {dataset_name} | Pattern Index: {self.current_pattern_index + 1}", fontsize=16)

			def plot_grid(ax, pattern, title):
				if pattern is None:
					ax.axis("off")
					return
				grid = pattern.reshape(lines_height, p_width)
				ax.imshow(grid, cmap="Greys", vmin=-1, vmax=1)
				ax.set_title(title, fontsize=10, fontweight="bold")
				ax.set_xticks([])
				ax.set_yticks([])

			plot_grid(axes[0, 0], target_pattern, "1. Training Pattern (Target)")
			plot_grid(axes[0, 1], results[0][1], "3. " + results[0][0])
			plot_grid(axes[0, 2], results[1][1], "4. " + results[1][0])

			plot_grid(axes[1, 0], input_pattern, "2. Testing Pattern (Input)")
			plot_grid(axes[1, 1], results[2][1], "5. " + results[2][0])
			plot_grid(axes[1, 2], results[3][1], "6. " + results[3][0])

			if not os.path.exists("screenshot"):
				os.makedirs("screenshot")

			filename = f"screenshot/{dataset_name.lower()}_pattern{self.current_pattern_index + 1}.png"
			plt.tight_layout()
			plt.savefig(filename)
			plt.close(fig)

		except Exception as e:
			messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
	root = tk.Tk()
	app = HopfieldGUI(root)
	root.mainloop()
