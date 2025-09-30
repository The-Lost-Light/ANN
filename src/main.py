import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import os
import perceptron
import multilayer_perceptron


def gui(page: ft.Page):
	page.theme_mode = ft.ThemeMode.LIGHT
	page.title = "NN_HW1_104525006"

	path = ""

	def pick_files_result(e: ft.FilePickerResultEvent):
		nonlocal path
		if(e.files):
			selected_files.value = os.path.splitext(e.files[0].name)[0]
			path = e.files[0].path
			selected_files.update()

	pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
	page.overlay.append(pick_files_dialog)

	def multilayer_perceptron_enable(e):
		layer_size.disabled = not switch.value
		layer_size.update()


	def open_dialog(result):
		epoch, accuracy, _, fig = result
		dialog = ft.AlertDialog(
			title=ft.Text("Plot"),
			content=ft.Column(controls=[
				MatplotlibChart(fig, expand=True),
				ft.Row(controls=[
					ft.Text("Epoch: " + str(epoch)),
					ft.Text("Accuracy: " + str(accuracy))
				])
			])
		)
		page.open(dialog)
		plt.close(fig)


	def train(e):
		if(path == ""):
			return

		train_button.disabled = True
		train_button.text = "Training..."
		epoch.value = "Epoch: Training..."
		accuracy.value = "Accuracy: Training..."
		weights.value = "Weights:  Training..."
		page.update()
		result = ['?', '?']
		if(switch.value):
			result = multilayer_perceptron.train(
				file_path=path,
				file_name=selected_files.value,
				learning_rate = (float(learning_rate.value) if learning_rate.value else 0.2),
				epochs = (int(epochs.value) if epochs.value else 100),
				accuracy_limit = (float(accuracy_limit.value) if accuracy_limit.value else 0.95),
				hidden_layers_size = ([int(word.strip()) for word in layer_size.value.split(',')] if layer_size.value else [20, 12, 6])
			)
		else:
			result = perceptron.train(
				file_path=path,
				file_name=selected_files.value,
				learning_rate = (float(learning_rate.value) if learning_rate.value else 0.2),
				epochs = (int(epochs.value) if epochs.value else 100),
				accuracy_limit = (float(accuracy_limit.value) if accuracy_limit.value else 0.95)
			)
		epoch.value = "Epoch: " + str(result[0])
		accuracy.value = "Accuracy: " + str(result[1])
		weights.value = "Weights: " + str(result[2])
		train_button.text = "Train!"
		train_button.disabled = False
		if(result[3]):
			open_dialog(result)
		page.update()

	page.floating_action_button = train_button = ft.FloatingActionButton(
		icon=ft.Icons.PLAY_ARROW_ROUNDED, text="Train!",
		on_click=train
	)

	page.add(
		ft.SafeArea(
			ft.Row(controls=[
				ft.Container(
					ft.Column(controls=[
						ft.Row(controls=[
							ft.ElevatedButton(
								"Pick files",
								icon=ft.Icons.UPLOAD_FILE,
								on_click=lambda _: pick_files_dialog.pick_files(
									allowed_extensions=["txt", "TXT"]
								),
							),
							selected_files := ft.Text()
						]),
						learning_rate := ft.TextField(label="Learning rate", hint_text="0.1"),
						epochs := ft.TextField(label="Epoch", hint_text="100"),
						accuracy_limit := ft.TextField(label="Accuracylimit", hint_text="0.95"),
						ft.Row(controls=[
							switch := ft.Switch(label="Use multilayer perceptron", on_change=multilayer_perceptron_enable),
							layer_size := ft.TextField(label="Hidden layer size", prefix_text='[', suffix_text=']', hint_text="20, 12, 6", disabled=True)
						])
					]),
					alignment=ft.alignment.center,
				),
				ft.Container(
					ft.Column(controls=[
						epoch := ft.Text("Epoch: "),
						accuracy := ft.Text("Accuracy: "),
						weights := ft.Text("Weights: ")
					]),
					alignment=ft.alignment.center
				)
			]),
			expand=True,
		)
	)


ft.app(gui)
