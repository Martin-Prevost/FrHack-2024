import PySimpleGUI as sg

layout = [
    [sg.Text("Bienvenue veuillez chosir les fichiers suivants", size=(35, 1), font=('Helvetica', 25))],
    [sg.Text("Sélectionnez un fichier de mesures :")],
    [sg.InputText(key='mesure'), sg.FileBrowse()],
    [sg.Text("Sélectionnez un shape de zones urbaines :")],
    [sg.InputText(key='urbain'), sg.FileBrowse()],
    [sg.Text("Sélectionnez un shape de zones rurale :")],
    [sg.InputText(key='rurale'), sg.FileBrowse()],
    [sg.Text("Sélectionnez un shape de zones peri-urbain :")],
    [sg.InputText(key='peri_urbain'), sg.FileBrowse()],
    [sg.Button('OK'), sg.Button('Annuler')]
]

window = sg.Window('Choix de fichier', layout)

while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Annuler'):
        break
    elif event == 'OK':
        selected_file = values['-FILE-']
        sg.popup(f'Fichier sélectionné : {selected_file}')
        # Faites ce que vous voulez avec le fichier sélectionné ici
        break

window.close()
