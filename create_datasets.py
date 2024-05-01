### Create datasets
#### Partition the entire dataset into N Clients
import pandas as pd
import numpy as np
## Function to create datasets
def create_datasets_clients(N_device = 100, fraction_of_data = 1,batch_size = 40):
    df = pd.read_csv("./data/df_treated_comment.csv")

##    df = df.sample(frac=fraction_of_data).reset_index(drop=True)
    ## Composition before balancing the dataset
    df['target'] = np.where(df['target'] >= 0.5, 1, 0)
    print("Composition of dataset before balancing: ",df['target'].value_counts())

    ## Balance the dataset
    df_1 = df[df['target'] == 1]
    df_0 = df[df['target'] == 0].sample(n=df_1.shape[0]).reset_index(drop=True)
    print("Balanced dataset with shape: ",df_1.shape,df_0.shape)
    df = pd.concat([df_1,df_0]).reset_index(drop=True)

    ## Take fraction of data
    df = df.sample(frac=fraction_of_data).reset_index(drop=True)
    ## Create a list of the labels
    df['list'] = df[df.columns[1:]].values.tolist()
    df.rename(columns={'treated_comment': 'comment_text'}, inplace=True)
    df = df[['comment_text', 'list']].copy()
    #### Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    N_total = df.shape[0]
    N_batch = int(N_total/N_device)

    #### Partion the dataset into N devices disjoint datasets and save dataset
    #### Serially partition the dataset
    for i in range(N_device):
        client_dataset = df.iloc[i*N_batch:(i+1)*N_batch,:]
        client_dataset = client_dataset.reset_index(drop=True)
        client_dataset.to_csv(f"./data/client_datasets/client_{i}.csv")
    print("Datasets created for ",N_device," clients")
    ## Create dataset for eavesdropper
    # Filter out the df if list column has 0 
    # have 10% 0 and 90% 1
    df_eav_1 = df[df['list'].apply(lambda x: 1 in x)].sample(frac=0.1)
    df_eav_0 = df[df['list'].apply(lambda x: 0 in x)].sample(frac=0.9)
    df_eav = pd.concat([df_eav_1,df_eav_0])
    
    # train_df_eav = df_eav.sample(n = int(N_batch*0.8))
    # valid_df_eav = df.drop(train_df_eav.index).reset_index(drop=True).sample(n = int(N_batch*0.2))
    # train_df_eav = train_df_eav.reset_index(drop=True)
    # print("Eavesdropper dataset created with shape: ",train_df_eav.shape,valid_df_eav.shape)
    # train_df_eav.to_csv(f"./data/client_datasets/client_eav_train.csv",index=False)
    # valid_df_eav.to_csv(f"./data/client_datasets/client_eav_valid.csv",index=False)