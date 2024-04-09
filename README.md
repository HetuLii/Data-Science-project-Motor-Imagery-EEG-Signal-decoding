# Introduction
This repository contains a replication study of several state-of-the-art models in literatures focusing on motor imagery EEG signals. Further, experiments on some unaddressed questions are carried out and recorded. This project aims to:
* Provide pre-processed, augmented, and ready-to-use motor imagery EEG dataset **BCIC-IV-2A**
* Compare the cross-subject performances of models including fundamental **CNN**, **LSTM** models, hybrid models, **EEGNet-8,2** model, **EEG-TCNet** model, and **EEGNeX** model
* Enable other researchers to conveniently carry out cross-subject comparisons on their datasets
* Investigate the efficacy of incorporating **Attention layer** in the EEGNet-8,2 model
# Dataset: BCIB-IV-2A
## Dataset Description
The **BCIC-IV-2A** dataset is recorded from $9$ subjects. The cue-based BCI paradigm consisted the imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. 
One run consists of $48$ trials ($12$ for each of the four possible classes), yielding a total of $288$ trials per session.

For instance, the data for the first participant are provided in two files labeled as A01T/A01E for the two complete sessions. Each file contains $288$ samples, where 'T' is used for training and 'E' for testing.

The data are collected at a rate of $250 Hz$. The CSV file represents $4$ seconds of data, specifically the data from 2 to 6 seconds in **Figure 1**, totaling $1000$ data points. The data underwent bandpass filtering from $0.5$ to $100 Hz$ and a notch filter at $50 Hz$ to eliminate noise interference.

The dataset utilizes the **international 10-20 system** for electrode placement. There are $25$ channels in total, of which three channels are for Electrooculography (EOG) data. For classifying MI (Motor Imagery) signals, it is necessary to filter the channels during preprocessing, retaining $22$ channels.

![alt text](https://github.com/HetuLii/Data-Science-project-Motor-Imagery-EEG-Signal-decoding/blob/4ff40eac878e5b4b105318adc9da8990800e4b14/cue-based%20BCI%20paradigm.png)
*Figure 1: cue-based BCI paradigm*
## Data Pre-Processing
