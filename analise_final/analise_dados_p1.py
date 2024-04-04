import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import time
from scipy import signal
from scipy.signal import welch
from mne.time_frequency import tfr_morlet

columns_to_remove = ['Aux 1', 'Aux 2',  'Event Date', 'Event Duration', 'Epoch', 'Time:512Hz']
# Dicionário de mapeamento de colunas antigas para novos nomes
ch_names = {
    'Channel 1': 'AF3',
    'Channel 2': 'F3',
    'Channel 3': 'F8',
    'Channel 4': 'T7',
    'Channel 5': 'FC5',
    'Channel 6': 'P3',
    'Channel 7': 'P7',
    'Channel 8': 'O1',
    'Channel 9': 'AF4',
    'Channel 10': 'F4',
    'Channel 11': 'F7',
    'Channel 12': 'T8',
    'Channel 13': 'FC6',
    'Channel 14': 'P4',
    'Channel 15': 'P8',
    'Channel 16': 'O2',
    'Event Id': 'Event Id'  # Manter o nome da coluna de eventos
}

caminho_arquivo = r"C:\Users\ariog\Downloads\memory_task_eeg_n-back\Coleta2_Gustavo.csv"
raw_data = pd.read_csv(caminho_arquivo)
raw_data = raw_data.drop(columns=columns_to_remove, axis=1)
raw_data = raw_data.rename(columns=ch_names)

# Sequência desejada das colunas
sequencia_colunas = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6", "T7", "T8", "P7", "P3", "P4", "P8", "O1", "O2", "Event Id"]

# Reordenar as colunas conforme a sequência desejada
raw_data = raw_data[sequencia_colunas]

# Mapear os valores originais para os novos valores desejados
mapeamento_valores = {33043.0: 1, 33040.0: 2, 33042.0: 3}

# Renomear os valores na coluna 'Event Id' usando o método replace()
raw_data['Event Id'] = raw_data['Event Id'].replace(mapeamento_valores)

marcador_baseline_close = raw_data[raw_data['Event Id'] == 1].index
marcador_baseline_open = raw_data[raw_data['Event Id'] == 2].index
marcador_start_eventos = raw_data[raw_data['Event Id'] == 3].index


df_splits = np.concatenate((marcador_baseline_close, marcador_baseline_open, marcador_start_eventos))

init_exp = np.array([0])
df_splits_events = np.concatenate((init_exp, df_splits))

event_type = []
for value in df_splits_events:
  if value in marcador_baseline_close:
    event_type.append(1)
  elif value in marcador_baseline_open:
    event_type.append(2)
  elif value in marcador_start_eventos:
    event_type.append(3)
  else:
    event_type.append(0)
    
code_real = {'start_exp':0, 'eyes_close':1, 'eyes_open':2, 'stim':3}

event_mne = np.zeros((df_splits_events.shape[0],3), dtype = 'int')

event_mne[:,0] = df_splits_events
event_mne[:,2] = event_type

event_mne[df_splits_events.shape[0]-1,2] = 3

# Converte a lista de arrays numpy em uma matriz numpy
matriz_eventos = np.array(event_mne)

# Obtém os índices ordenados dos elementos com base no primeiro elemento de cada linha
indices_ordenados = matriz_eventos[:, 0].argsort()

# Ordena a matriz de eventos com base nos índices ordenados
event_mne = matriz_eventos[indices_ordenados]

sfreq = 512
filter = "bandpass"
ordem_filtro = 3
sinal_alvo = [4,8]

ch_names = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6", "T7", "T8", "P7", "P3", "P4", "P8", "O1", "O2",  'Event Id']

channel_types = ['eeg'] * (len(ch_names) - 1)  # Todos os canais, exceto o último, são do tipo 'eeg'
channel_types.append('stim')  # Último canal é do tipo 'stim'
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=channel_types)
samples_scale = raw_data.T * 1e-6
data = raw_felipe = mne.io.RawArray(samples_scale, info)
data.set_montage(montage=montage)

data.set_eeg_reference("average")
data.filter(l_freq=4, h_freq=7)

fig= data.plot_sensors(show_names=True)
plt.title('Plot 1')
plt.show(block=True)

fig = data.plot(scalings=dict(eeg=20e-6))
plt.title('Plot 2')
plt.show(block=True)

ax = data.compute_psd(fmax=30, tmin=220.0, tmax=251.0).plot(picks=list(range(16)), dB =False)
plt.title('Plot 3')
plt.show(block=True)

# fig = mne.viz.plot_events(event_mne, event_id=code_real, sfreq=data.info['sfreq'], first_samp=data.first_samp)
# plt.title('Plot 4')
# plt.show(block=True)

epochs = mne.Epochs(data, event_mne, event_id={"stim": 3}, tmin=0, tmax=2, baseline=None)
epochs.plot()
plt.title('Plot 5')
plt.show(block=True)

# epochs.average().plot()
# plt.show(block=True)

# epochs.compute_psd().plot()
# plt.show(block=True)

freqs =  np.linspace(4, 7, 20)

power = tfr_morlet(
    epochs,
    freqs = freqs,
    n_cycles = freqs/2,
    use_fft=True,
    return_itc=False,
    n_jobs=-1,
    average=True    
)

power.plot_topo(dB=False)