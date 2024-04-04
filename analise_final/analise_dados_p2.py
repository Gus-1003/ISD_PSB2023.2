import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt

def min_max_normalization(data, feature_range):
    """Normaliza os dados em um DataFrame para o intervalo Min-Max especificado."""
    min_value = data.min()
    max_value = data.max()
    normalized_data = (data - min_value) / (max_value - min_value)
    scaled_data = normalized_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return scaled_data

def CAR_filter(signal):
    """Aplica o filtro CAR (Common Average Reference) aos dados EEG."""
    noise = signal.mean(axis=1).to_numpy()
    eeg_car = signal.to_numpy() - noise.reshape((-1, 1))
    return pd.DataFrame(eeg_car, columns=signal.columns)

def bandpass_filter(signal, N=3, f_low=4, f_high=7, fs=512):
    """Aplica um filtro passa-banda aos dados EEG."""
    b, a = butter(N, [f_low, f_high], 'bandpass', fs=fs)
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

def calculate_energy(data):
    """Calcula a energia de cada canal nos dados EEG."""
    return (data ** 2).sum(axis=0) / (2 * data.shape[-1] + 1)

# Parâmetros e constantes configuráveis
columns_to_remove = ['Aux 1', 'Aux 2',  'Event Date', 'Event Duration', 'Epoch', 'Time:512Hz']
map_columns_names = {
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
    'Event Id': 'Event Id'
}
sorting_columns = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6", "T7", "T8", "P7", "P3", "P4", "P8", "O1", "O2", "Event Id"]
signal_channels = ["AF3", "AF4", "F7", "F3", "F4", "F8", "FC5", "FC6", "T7", "T8", "P7", "P3", "P4", "P8", "O1", "O2"]
map_event_id = {33043.0: 1, 33040.0: 2, 33042.0: 3}
duration_event = 2 # segundos
freqs = 512 # frequência de amostragem 
length_event_in_dataset = duration_event * freqs
feature_range=(0, 1)

# Lendo e Pre-processando o Arquivo de Sinal:
caminho_arquivo = r"C:\Users\ariog\Downloads\memory_task_eeg_n-back\Coleta2_Gustavo.csv"
raw_data = pd.read_csv(caminho_arquivo)
raw_data = raw_data.drop(columns=columns_to_remove).rename(columns=map_columns_names)[sorting_columns]
raw_data['Event Id'] = raw_data['Event Id'].replace(map_event_id)

# Lendo e Pre-processando o Arquivo de Eventos:
protocolo_1 = pd.read_csv(r"C:\Users\ariog\Downloads\memory_task_eeg_n-back\n_back1.csv")
protocolo_2 = pd.read_csv(r"C:\Users\ariog\Downloads\memory_task_eeg_n-back\n_back2.csv")
protocolo_3 = pd.read_csv(r"C:\Users\ariog\Downloads\memory_task_eeg_n-back\n_back3.csv")

eventos_concat = pd.concat([protocolo_1, protocolo_2, protocolo_3])
eventos_concat['reacao'] = eventos_concat['reacao'].replace("['space']", 1).fillna(0)
eventos_concat = eventos_concat.reset_index(drop=True)

indices_com_1 = eventos_concat[eventos_concat['reacao'] == 1].index

# Formando Datasets dos sinais
marcador_baseline_close = raw_data[raw_data['Event Id'] == 1].index[0] # Index_Start_baseline - Olhos Fechados
marcador_baseline_open = raw_data[raw_data['Event Id'] == 2].index[0] # Index_Start_baseline - Olhos Abertos
marcador_start_eventos = raw_data[raw_data['Event Id'] == 3].index  # Index_start_events

df_baseline_close = raw_data.loc[marcador_baseline_close:marcador_baseline_open-1] # Baseline - Olhos Fechados
df_baseline_open = raw_data.loc[marcador_baseline_open:marcador_baseline_open+len(df_baseline_close)-1] # Baseline - Olhos Abertos

# Separando os marcadores baseados em cada protocolo - Epochs
## epochs = intervalos de tempo específicos de um sinal EEG
protoc_1 = marcador_start_eventos[0:37]
protoc_2 = marcador_start_eventos[37:75]
protoc_3 = marcador_start_eventos[75:114]

init_exp = np.array([0])
splits_events = np.concatenate((init_exp, marcador_start_eventos))

# Gerando uma estrutura para a marcacao das Epochs na biblioteca MNE
protocol_type = []
for value in splits_events:
  if value in protoc_1:
    protocol_type.append(1)
  elif value in protoc_2:
    protocol_type.append(2)
  elif value in protoc_3:
    protocol_type.append(3)
  else:
    protocol_type.append(0)

event_mne = np.zeros((splits_events.shape[0],3), dtype = 'int')

event_mne[:,0] = splits_events
event_mne[:,2] = protocol_type

## Este é um objeto da biblioteca MNE que contém epochs (segmentos de dados temporais) extraídas de um sinal EEG original.
event_mne[splits_events.shape[0]-1,2] = 3

# Processando o Sinal bruto e atenuando ruidos
raw_signals = raw_data.iloc[:, 0:16].copy()
raw_signals_norm = min_max_normalization(raw_signals, feature_range)
eeg_car = CAR_filter(raw_signals_norm)
raw_signals_filtered = eeg_car.apply(bandpass_filter)

# Calculando o poder em cada canal
eeg_power = calculate_energy(raw_signals_filtered)
eeg_power

# Trabalhando com MNE - Convertendo a estrutura de dataset -> Arquivo MNE:
## cria uma lista channel_types contendo o tipo de canal para cada canal no signal_channels.
channel_types = ['eeg'] * (len(signal_channels))
## criando um objeto montage que representa a configuração padrão dos eletrodos EEG no sistema de montagem 10-20.
montage = mne.channels.make_standard_montage("standard_1020")
## cria um objeto info que contém informações sobre os canais EEG.
info = mne.create_info(ch_names=signal_channels, sfreq=freqs, ch_types=channel_types)
## Nesta linha, os dados brutos (raw_signals) são transpostos (.T) e convertdios de microvolts para volts
samples_scale = raw_signals.T * 1e-6
## um objeto RawArray é criado, este objeto RawArray é uma estrutura de dados MNE que contém os dados EEG prontos para análise.
data = mne.io.RawArray(samples_scale, info)
## associando o objeto montage (representando a configuração padrão dos eletrodos) ao objeto data.
data.set_montage(montage=montage)

information = mne.create_info(ch_names=signal_channels, sfreq=freqs, ch_types=channel_types)
information.set_montage(mne.channels.make_standard_montage('standard_1020'))

# Aplicação de filtros - Dessa vez na versão do arquivo MNE
data.set_eeg_reference("average")
data.notch_filter(freqs=50, notch_widths=1)
data.filter(l_freq=4, h_freq=7)

# Plot - Sensors
## Este gráfico mostra a localização dos sensores EEG no cabeçote padrão 10-20.
fig= data.plot_sensors(show_names=True)
plt.title('Plot 1 - Sensors')
plt.show(block=True)

# Plot - Dados
## Este gráfico mostra os dados brutos em todos os canais EEG ao longo do tempo.
fig = data.plot()
plt.title('Plot 2 - Dados')
plt.show(block=True)

# Plot - Topomap
## Este gráfico mostra um mapa topográfico da distribuição de energia nos canais EEG.
fig, axs = plt.subplots(1,1,figsize=(15,8))
im,cm = mne.viz.plot_topomap(eeg_power,information,extrapolate='head',axes=axs,names=signal_channels, contours=5, show=False)

clb = fig.colorbar(im, format="%.1f")
clb.ax.tick_params(labelsize=7)
clb.ax.set_title("uV²",fontsize=8)
plt.title('Plot 3 - Topomap')
plt.show(block=True)

# Plot - PSD
## Este gráfico mostra a densidade espectral de potência (PSD) dos dados EEG.
ax = data.compute_psd(fmax=15).plot(picks=list(range(16)), dB =False)
plt.title('Plot 4 - PSD')
plt.show(block=True)

# Mapeando para dividir os estimulos (Event Id == 3) em 3 protocolos
stim = {'start_exp':0, 'N_back1':1, 'N_back2':2, 'N_back3':3}

# Plot - Events
## Este gráfico mostra a distribuição temporal dos eventos (estímulos) ao longo do tempo.
fig = mne.viz.plot_events(event_mne, event_id=stim, sfreq=data.info['sfreq'],first_samp=data.first_samp)
plt.title('Plot 5 - Events')
plt.show(block=True)

# Criando as Epochs a partir da biblioteca MNE
epochs_mne = mne.Epochs(data, event_mne, event_id=stim, tmax=2, preload=False)

# Separando as Epochs baseado em cada tipo de protocolo / Tipo de Estimulo
event_focus = ["N_back1", "N_back2"]
epochs_mne.equalize_event_counts(event_focus)

ptc_back1 = epochs_mne["N_back1"]
ptc_back2 = epochs_mne["N_back2"]

## average() é um método da classe Epochs da biblioteca MNE, que calcula a média dos dados em todas as epochs do tipo especificado.
## potencial evocado (ou média evocada) é uma representação média da atividade cerebral em resposta a um estímulo específico ("N_back1" e "N_back2").
ptc_evoked_1 = ptc_back1.average()
ptc_evoked_2 = ptc_back2.average()

# Plot - topomap - Cada protocolo - Utilizando as Epochs - Posterior a um filtro de media
all_times = np.arange(-0.1, 2, 0.1)

## Este gráfico mostra um mapa topográfico da distribuição de energia para o protocolo 1.
fig = ptc_evoked_1.plot_topomap(all_times, ch_type="eeg", ncols=8, nrows="auto")
plt.title('Plot 6 - topomap - Protocolo 1')
plt.show(block=True)

## Este gráfico mostra um mapa topográfico da distribuição de energia para o protocolo 2.
fig = ptc_evoked_2.plot_topomap(all_times, ch_type="eeg", ncols=8, nrows="auto")
plt.title('Plot 7 - topomap - Protocolo 2')
plt.show(block=True)

# Plot - Comparação Evoked
## Este gráfico compara os potenciais evocados (média dos eventos) entre os protocolos 1 e 2.
mne.viz.plot_compare_evokeds(evokeds=[ptc_evoked_1, ptc_evoked_2])

# Plot - topo_image - Procolo 1
## Este gráfico mostra uma imagem topográfica da atividade do cérebro durante o protocolo 1.
ptc_back1.plot_topo_image(title='Topo Image N_back1', colorbar=True, vmin=-10, vmax=10)
plt.title('Plot 9 - Topo_image - Somente Protocolo 1')
plt.show(block=True)

# Plot - topo_image - Procolo 1
## Este gráfico mostra uma imagem topográfica da atividade do cérebro durante o protocolo 2.
ptc_back2.plot_topo_image(title='Topo Image N_back2', colorbar=True, vmin=-10, vmax=10)
plt.title('Plot 10 - Topo_image - Somente Protocolo 2')
plt.show(block=True)