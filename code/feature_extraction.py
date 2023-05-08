import warnings
import numpy as np
warnings.filterwarnings("ignore")

def EIIP_protein(seq):
    max_length=15
    seq_list=[]
    protein = {'A': 0.3710, 'C': 0.08292, 'D': 0.12630, 'E': 0.00580, 'F': 0.09460,
                'G': 0.0049, 'H': 0.02415, 'I': 0.0000, 'K': 0.37100, 'L': 0.0000,
                'M': 0.08226, 'N': 0.00359, 'P': 0.01979, 'Q': 0.07606, 'R': 0.95930, 'S': 0.08292,
                'T': 0.09408, 'V': 0.00569, 'W': 0.05481, 'Y': 0.05159}
    for i in range(len(seq)):
        if seq[i] in protein:
            seq_list.append(protein[seq[i]])
    padding=(max_length-len(seq_list))
    seq_list=np.pad(seq_list,(0,padding),'constant')
    return seq_list

def accumulated_amino_frequency_protein(seq):
    max_length=15
    seq_list=[]
    protein = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0,
                       'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
    for i in range(len(seq)):
        if seq[i] in protein:
            protein[seq[i]] +=1
            seq_list.append(protein[seq[i]]/(i+1))
    padding=(max_length-len(seq_list))
    seq_list=np.pad(seq_list,(0, padding), 'constant')
    return seq_list

def trans(seq):
    max_length = 15
    seq_list = []
    dic = {'A': 1, 'C': 5, 'D': 4, 'E': 7, 'F': 14, 'G': 8, 'H': 9, 'I': 10, 'K': 12, 'L': 11,'M': 13, 'N': 3, 'P': 15, 'Q': 6, 'R': 2, 'S': 16, 'T': 17, 'V': 20, 'W': 18, 'Y': 19}
    for i in range(len(seq)):
        if seq[i] in dic:
            seq_list.append(dic[seq[i]])
    padding = (max_length-len(seq_list))
    seq_list = np.pad(seq_list, (0,padding), 'constant')
    return seq_list

def ProssData(seq):
    seq_list1 = []
    seq_list2 = []
    seq_list3 = []
    f = open(seq).readlines()
    for i in range(0, len(f)-1, 2):
        sequence = f[i+1].strip('\n')
        seq_list1.append(EIIP_protein(sequence))
        seq_list2.append(trans(sequence))
        seq_list3.append(accumulated_amino_frequency_protein(sequence))
        np.concatenate([seq_list1, seq_list2, seq_list3], axis=-1)
    return np.concatenate([seq_list1, seq_list2, seq_list3], axis=-1)


def shuffle_data(data,label):
    np.random.seed(42)
    label=label.reshape(-1, 1)
    sum_data=np.concatenate([data,label], axis=-1)
    X=np.array(sum_data[:, :-1])
    y=np.array(sum_data[:, -1])
    np.random.seed(1)
    np.random.shuffle(X)
    np.random.seed(1)
    np.random.shuffle(y)
    return X, y