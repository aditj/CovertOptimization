import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from nn import BERTClass
from transformers import AlbertTokenizer, BertTokenizer
import pandas as pd
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2', do_lower_case=True)
from torch.utils.data import Dataset, DataLoader
### suppress ptorch transformer warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data['list']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        # make string of a list as a list
        targets = self.targets[index]
        targets = [float(i) for i in targets]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }
## Class for FL client 
class Client():
    def __init__(self,cid,network,train_batch_size = 16,valid_batch_size = 80,max_len = 300,epochs = 1,learning_rate = 1e-04,device = "mps",n_classes = 2):
        self.cid = cid
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.max_len = max_len
        self.n_classes = n_classes 
        self.load_data()
        self.model = network
        self.model.to(self.device)
       
        print("Client", self.cid, "initialized with ",len(self.train_dataset),"train samples and ",len(self.valid_dataset),"valid samples")

    def load_data(self):
        df = pd.read_csv(f'data/client_datasets/client_{self.cid}.csv')
        df['list'] = df['list'].apply(lambda x: x.strip('][').split(','))
        df['list'] = df['list'].apply(lambda x: [float(i) for i in x])
        train_df = df.sample(frac=0.8, random_state=200)
        valid_df = df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        self.train_dataset = CustomDataset(train_df, tokenizer, self.max_len)
        self.valid_dataset = CustomDataset(valid_df, tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_dataset, shuffle = True, batch_size=self.train_batch_size, num_workers=0)
        self.n_batch_per_client = len(self.train_loader)//self.train_batch_size
        self.valid_loader = DataLoader(self.valid_dataset, shuffle = True, batch_size=self.valid_batch_size, num_workers=0)
    def train(self,n_samples):
        ## Adam with weight decay AdamW
        #optimizer = torch.optim.SGD(params =  self.model.parameters(), lr=self.learning_rate)

        optimizer = torch.optim.AdamW(params =  self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            ## compute number of batches
            n_batches = len(self.train_loader)
            n_batches_max = int(n_samples//self.train_batch_size)
            batch_indices = np.random.choice(n_batches,n_batches_max,replace=False)
            # take subset of train  loader from n_batches_start to n_batches_end                    
            ## randomly sample n_batches_max batches

            for _,data in enumerate(self.train_loader,0):
                
                if _ not in batch_indices:
                    continue
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                optimizer.zero_grad()
                
                loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #print(f'Client: {self.cid}, Batch: {_}, Loss:  {loss.item()}')
    def evaluate(self,batch_size):
        
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        loss = []
        with torch.no_grad():
            for _, data in enumerate(self.valid_loader):
                ids = data['ids'].to(self.device,torch.long)
                mask = data['mask'].to(self.device,torch.long)
                token_type_ids = data['token_type_ids'].to(self.device,torch.long)
                targets = data['targets'].to(self.device,torch.float)
                outputs = self.model(ids, mask, token_type_ids)
               # pos_weights = torch.ones([self.n_classes])*10
               # pos_weights = pos_weights.to(self.device)
                #loss+=[torch.nn.BCEWithLogitsLoss(pos_weight = pos_weights)(outputs, targets)]
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs = np.array(fin_outputs) >= 0.5
        fin_targets = np.array(fin_targets)
        ## create weights for samples with all 0 classes to avoid bias
       # f1_score = metrics.f1_score(fin_targets, outputs, average='weighted',zero_division=1)
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        #balanced_accuracy = metrics.balanced_accuracy_score(fin_targets, outputs, adjusted=True)       
        return accuracy
    def get_parameters(self):
        return self.model.state_dict()
    def set_parameters(self,parameters_state_dict):
        self.model.load_state_dict(parameters_state_dict, strict=True)
    def are_parameters_equal(self,parameters_state_dict):
        for layer in self.model.state_dict():
            if not torch.equal(self.model.state_dict()[layer],parameters_state_dict[layer]):
                print("Parameters are not equal")
                return False
        return True