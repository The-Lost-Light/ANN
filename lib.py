import numpy as np

import lib


def load_patterns(level, lines):
	def load(data):
		with open("Hopfield_dataset/" + data + ".txt", "r") as f:
			patterns_data = [line.strip("\n") for line in f.readlines()]

		pattern_numbers = (len(patterns_data) + 1) // (lines + 1)
		patterns = []
		for n in range(pattern_numbers):
			current_pattern = lib.strings2list(patterns_data[(lines + 1) * n : (lines + 1) * n + lines])
			patterns.append(current_pattern)

		return patterns, len(patterns_data[0])

	train, p = load(level + "_Training")
	test, _ = load(level + "_Testing")
	return train, test, p


def strings2list(patterns):
	current_pattern = "".join(patterns)
	return np.array([1 if char == "1" else -1 for char in current_pattern]).reshape(-1, 1)


def list2string(pattern, step):
	pattern_string = "".join(["1" if n == 1 else " " for n in pattern])
	return "\n".join([pattern_string[t : t + step] for t in range(0, len(pattern_string), step)])
