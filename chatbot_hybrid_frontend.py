from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Constants
MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


###################################################
# DEFINE ENCODER
###################################################
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)  # 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # 2
        outputs, hidden = self.gru(packed, hidden)  # 3
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # 4
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # 5
        return outputs, hidden # 6



###################################################
# DEFINE DECODER'S ATTENTION MODULE
###################################################
# Luong attention layer
class Attn(nn.Module):
#class Attn(torch.jit.ScriptModule):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    #@torch.jit.script_method
    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, max_len) # B x S
        attn_energies = attn_energies.to(device)

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    # Score functions
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy


###################################################
# DEFINE DECODER
###################################################
#class LuongAttnDecoderRNN(nn.Module):
class LuongAttnDecoderRNN(torch.jit.ScriptModule):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    @torch.jit.script_method
    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        # if(embedded.size(0) != 1):
        #     raise ValueError('Decoder input sequence length should be 1')
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden



###################################################
# DATA HANDLING
###################################################
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


###################################################
# DECODE DRIVER
###################################################
# Decode the context vector (decoder_hidden) with attention
def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):
    # Initialize input, words, and attentions
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)
    decoded_words = []

    # Allow output sequences with a max length of max_length
    for _ in range(max_length):
        # Run forward pass though decoder model
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # Take word with highest softmax probability
        _, topi = decoder_output.topk(1)
        ni = topi[0][0]
        # If the recommended word is an EOS token, append the token to the decoded_words list and stop decoding
        if ni == EOS_token:
            break
        # Else, append the string word to decoded_words list
        else:
            decoded_words.append(voc.index2word[ni.item()])

        # Set next decoder input as the chosen decoded word
        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words


###################################################
# EVALUATE INPUTS
###################################################
# Evaluate a sentence
def evaluate(encoder, decoder, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths)

    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Decode sentence
    return decode(decoder, decoder_hidden, encoder_outputs, voc)

# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, voc):
    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, voc, input_sentence)
    output_sentence = ' '.join(output_words)
    print('bot:', output_sentence)


###################################################
# LOAD MODEL
###################################################
save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"

# Configure models
model_name = 'model7'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter))

# Load checkpoint items
#checkpoint = torch.load(loadFilename)
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
embedding_sd = checkpoint['embedding']
voc = checkpoint['voc']

# Initialize Model
checkpoint = None
print('Building encoder and decoder ...')
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)


################################################################
# INITIALIZE MODELS
################################################################
### Initialize encoder
# No tracing
# encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)

# With tracing (this works)
dummy_seq = torch.zeros((1,1), dtype=torch.int64)
dummy_lengths = torch.tensor([1])
encoder = trace(dummy_seq, dummy_lengths)(EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout))

### Initialize decoder
# No tracing
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

# With tracing (does NOT work)
# decoder_dummy_seq = torch.zeros((1,1), dtype=torch.int64)
# decoder_dummy_last_hidden = torch.zeros((decoder_n_layers, 1, hidden_size), dtype=torch.float32)
# decoder_enc_outputs = torch.zeros((2, 1, hidden_size), dtype=torch.float32)
# decoder = trace(decoder_dummy_seq, decoder_dummy_last_hidden, decoder_enc_outputs)(LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout))


# Load checkpoint
# Loading on CPU
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
# Loading on GPU
#checkpoint = torch.load(loadFilename)

# Populate model parameters
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

# Set device
#encoder = encoder.to(device)
#decoder = decoder.to(device)
print('Models built and ready to go!')

# Set dropout layers to eval mode
#encoder.eval()
#decoder.eval()

# Print graphs
print('encoder graph', encoder.__getattr__('forward').graph)
print('decoder graph', decoder.__getattr__('forward').graph)


# Evaluate examples
sentences = ["hello", "what's up?", "who are you?", "where are we?", "where are you from?"]
for s in sentences:
    evaluateExample(s, encoder, decoder, voc)
