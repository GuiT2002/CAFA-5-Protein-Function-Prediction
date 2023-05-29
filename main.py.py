import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.utils import pad_sequences, to_categorical
import numpy as np

def parse_fasta(file_path):
    protein_sequences = {}
    max_sequence_length = 0  
    
    with open(file_path, 'r') as fasta_file:
        entry_id = None
        sequence = ''
        i = 1
        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if entry_id is not None and sequence != '':
                    # Convert sequence letters to numbers
                    sequence_numbers = [ord(letter.upper()) - ord('A') + 1 for letter in sequence]
                    protein_sequences[entry_id] = sequence_numbers
                    max_sequence_length = max(max_sequence_length, len(sequence_numbers))
                entry_id = line[1:].split()[0]
                sequence = ''
            else:
                sequence += line
            
            if i % 100000 == 0:
                print(f'{i} sequences processed')
            i += 1
            
        # Add the last entry to the dictionary
        if entry_id is not None and sequence != '':
            # Convert sequence letters to numbers
            sequence_numbers = [ord(letter.upper()) - ord('A') + 1 for letter in sequence]
            protein_sequences[entry_id] = sequence_numbers
            max_sequence_length = max(max_sequence_length, len(sequence_numbers))
    
    # Pad sequences with zeros to have the same length
    j = 1
    for key in protein_sequences:
        sequence = protein_sequences[key]
        padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post', truncating='post')[0]
        protein_sequences[key] = padded_sequence
        if j % 5000 == 0:
            print(f'{j} sequences padded!')
        j += 1

    return protein_sequences



def process_toplevel_cat(file_path):
    protein_aspects = {}
    with open(file_path, 'r') as tsv_file:
        next(tsv_file)  # Skip the header row
        for line in tsv_file:
            entry_id, term, aspect = line.strip().split('\t')
            if aspect == 'BPO':
                aspect = 1
            elif aspect == 'CCO':
                aspect = 2
            elif aspect == 'MFO':
                aspect = 3
            
            if entry_id not in protein_aspects:
                protein_aspects[entry_id] = []

            if aspect not in protein_aspects[entry_id]:
                protein_aspects[entry_id].append(aspect)

    return protein_aspects


fasta_file_path = 'CAFA 5 Protein Function Prediction/train_sequences.fasta'
tsv_file_path = 'CAFA 5 Protein Function Prediction/train_terms.tsv'

# Processing the data to feed the neural network
sequences = parse_fasta(fasta_file_path)
x_train = []
i = 1
for item in sequences.items():
    x_train.append(sequences[item])

    if i % 10000 == 0:
        print(f'{i} sequences appended!')

    i += 1

labels = process_toplevel_cat(tsv_file_path)
y_train = []
i = 1
for key in sequences.keys():
    y_train.append(to_categorical(labels[key]))

    if i % 10000 == 0:
        print(f'{i} labels appended!')
    i += 1

print(x_train)

# Creating the Neural Network
model = Sequential()

model.add(Conv1D(8, kernel_size=2, padding='valid', activation='relu'))
model.add(Conv1D(8, kernel_size=2, padding='valid', activation='relu'))

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)

model.summary()