import joblib
import torch
import sys
import torch.nn as nn
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from torchtext.data.utils import get_tokenizer
from nltk.lm import Vocabulary
from nltk import pos_tag
import numpy as np

# ------------ ( ONLY FOR DEBUGGING ) -------------------------
# output_file_path = 'output_and_errors.txt'
# sys.stdout = open(output_file_path, 'w')
# sys.stderr = sys.stdout

vocab = torch.load('/Users/tanvikarennavar/dl_group3_project/backend/scripts/vocab.pt')
POS_vocab = torch.load('/Users/tanvikarennavar/dl_group3_project/backend/scripts/POS_vocab.pt')

stopwords = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = get_tokenizer('basic_english')
max_len = 18

# ------------------ EXAMPLE ------------------------
# text = "Building a wall on the U.S.-Mexico border will take literally years."
# speaker = "rick-perry"
# job_title = "Governer"
# state_info = "Texas"
# party = "republican"
# context = "Denver"

# data = {
#     'text':text,
#     'speaker':speaker,
#     'job title':job_title,
#     'state info':state_info,
#     'party':party,
# }

data = {
    'text':sys.argv[1],
    'speaker':sys.argv[2],
    'job title':sys.argv[3],
    'state info':sys.argv[4],
    'party':sys.argv[5],
}

class HybridAttentionLSTM(torch.nn.Module):
    def __init__(self, input_size, input_size_POS, input_size_Meta, embedding_dim, hidden_size, lstm_hidden_size, output_size, batch_size):

        super(HybridAttentionLSTM, self).__init__()
        self.word_embeddings = nn.Embedding(input_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_size, batch_first=True, num_layers=1, dropout=0.8, bidirectional=True)
        self.Meta_fc = nn.Linear(input_size_Meta, 64)
        self.Meta_dropout = nn.Dropout(p=0.5)

        self.POS_word_embeddings = nn.Embedding(input_size_POS, embedding_dim)
        self.POS_lstm = nn.LSTM(embedding_dim, lstm_hidden_size, batch_first=True, num_layers=1, dropout=0.8, bidirectional=True)

        self.fc1 = nn.Linear(hidden_size*2 + 64, 100)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, output_size)
        self.softmax = nn.Softmax()

    def forward(self, X, POS, Meta, val = False):
        embedded = self.word_embeddings(X)
        rnn_out, (h_n, c_n) = self.lstm(embedded)

        embedded_POS = self.POS_word_embeddings(POS)
        rnn_POS, (h_POS,s_POS) = self.POS_lstm(embedded_POS)

        rnn_out = torch.cat([rnn_out,rnn_POS],dim = 2)
        h_n = torch.cat([h_n, h_POS], dim = 0)

        # attention
        merged_state = torch.cat([s for s in h_n],1).squeeze(0).unsqueeze(2)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        attn_out = torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

        out_meta = self.Meta_dropout(nn.functional.relu(self.Meta_fc(Meta)))
        out = torch.cat([attn_out,out_meta],dim=1)

        out = self.dropout1(nn.functional.relu(self.fc1(out)))
        out = self.fc2(out)
        return self.softmax(out)

input_size = len(vocab)
input_size_POS = len(POS_vocab)
input_size_Meta = 84

embedding_dim = 100
hidden_size = 100
output_size = 6

hidden_dim = 128
batch_size = 4
vocab_size = len(vocab)

embeddings = "glove.6b.300"

model = HybridAttentionLSTM( input_size, input_size_POS, input_size_Meta, embedding_dim, hidden_size, hidden_size // 2, output_size, batch_size)

model.load_state_dict(torch.load('/Users/tanvikarennavar/dl_group3_project/backend/scripts/model.pth'))

s = data["text"]
s = s.lower()
s = ''.join([c for c in s if c not in string.punctuation])
s = nltk.word_tokenize(s)
s = [word for word in s if word not in stopwords]
pos = pos_tag(s)
pos = [tup[1] for tup in pos]
s = [lemmatizer.lemmatize(word) for word in s]
s = ' '.join(s)
max_len = 50

def get_meta_data(query):
  top_categories_speaker = ['barack-obama', 'donald-trump', 'hillary-clinton', 'mitt-romney', 'scott-walker', 'john-mccain', 'rick-perry', 'chain-email', 'marco-rubio', 'rick-scott', 'ted-cruz', 'bernie-s', 'facebook-posts', 'chris-christie', 'newt-gingrich', 'charlie-crist', 'jeb-bush', 'joe-biden', 'blog-posting', 'paul-ryan']
  top_categories_job_title =['President', 'U.S. Senator', 'Governor', 'President-Elect', 'U.S. senator', 'Presidential candidate', 'Former governor', 'U.S. Representative', 'Milwaukee County Executive', 'Senator', 'State Senator', 'U.S. House of Representatives', 'U.S. representative', 'Congressman', 'Attorney', 'Social media posting', 'Governor of New Jersey', 'Co-host on CNN\'s "Crossfire"', 'State Representative', 'State representative']
  top_categories_state_info =['Texas', 'Florida', 'Wisconsin', 'New York', 'Illinois', 'Ohio', 'Georgia', 'Virginia', 'Rhode Island', 'New Jersey', 'Oregon', 'Massachusetts', 'Arizona', 'California', 'Washington, D.C.', 'Vermont', 'Pennsylvania', 'New Hampshire', 'Arkansas', 'Kentucky']
  top_categories_party =['republican', 'democrat', 'none', 'organization', 'independent', 'newsmaker', 'libertarian', 'activist', 'journalist', 'columnist', 'talk-show-host', 'state-official', 'labor-leader', 'tea-party-member', 'business-leader', 'green', 'education-official', 'liberal-party-canada', 'government-body', 'Moderate']
  top_categories_categories =['candidates-biography', 'crime', 'economy', 'education', 'elections', 'energy', 'environment', 'federal-budget', 'foreign-policy', 'guns', 'health-care', 'history', 'immigration', 'job-accomplishments', 'jobs', 'legal-issues', 'military', 'state-budget', 'taxes', 'workers']

  speaker_ohe = [0] * (len(top_categories_speaker)+1)
  try:
    index = top_categories_speaker.index(query['speaker'])
  except ValueError:
    index = len(top_categories_speaker)
  speaker_ohe[index] = 1

  job_title_ohe = [0] * (len(top_categories_job_title)+1)
  try:
    index = top_categories_job_title.index(query['job title'])
  except ValueError:
    index = len(top_categories_job_title)
  job_title_ohe[index] = 1

  state_info_ohe = [0] * (len(top_categories_state_info)+1)
  try:
    index = top_categories_state_info.index(query['state info'])
  except ValueError:
    index = len(top_categories_state_info)
  state_info_ohe[index] = 1

  party_ohe = [0] * (len(top_categories_party)+1)
  try:
    index = top_categories_party.index(query['party'])
  except ValueError:
    index = len(top_categories_party)
  party_ohe[index] = 1

  meta_data = np.hstack((speaker_ohe,job_title_ohe,state_info_ohe,party_ohe))

  return meta_data



def preprocessing(sample):
  if len(sample["Text"]) > 18:
    sample["Text"] = sample["Text"][:max_len].to(torch.int)
    sample["POS"] = sample["POS"][:max_len].to(torch.int)
  else:
    sample["Text"] = torch.cat((sample["Text"],torch.tensor([0 for i in range(len(sample["Text"]),max_len)])), dim = 0).to(torch.int)
    sample["POS"] = torch.cat((sample["POS"],torch.tensor([0 for i in range(len(sample["POS"]),max_len)])), dim = 0).to(torch.int)

  for key in sample:
    if torch.is_tensor(sample[key]):
      sample[key] = sample[key]
    else:
      sample[key] = torch.tensor(sample[key])
  return sample

Meta = get_meta_data(data)

sample = {"Text": s, "POS": pos,"Meta": Meta}
sample["Text"] = torch.tensor(vocab(tokenizer(sample["Text"])), dtype=torch.int)
sample["POS"] = torch.tensor(POS_vocab(sample["POS"]), dtype=torch.int)
sample["Meta"] = torch.tensor(sample["Meta"], dtype=torch.float)

sample = preprocessing(sample)
sample

model.eval()
with torch.no_grad():
  inputs = sample["Text"].expand(4, -1)
  POS = sample["POS"].expand(4, -1)
  Meta = sample["Meta"].expand(4, -1)
  outputs = model(inputs,POS,Meta, val = True)
  pred = torch.argmax(outputs,dim = 1)
  result = pred[0].cpu().numpy()

if result==0:
    print('Barely-True')
elif result==1:
    print('False')
elif result==2:
    print('Half-True')
elif result==3:
    print('Mostly-True')
elif result==4:
    print('Pants On Fire')
else:
    print('True')