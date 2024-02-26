import os, sys, re
import pickle as pkl
import pandas as pd
import torch
from sklearn.utils import shuffle
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

#gives id2label and label2id mappings
def get_label_mappings(labels):
    id2label = {}
    label2id = {}
    for i, label in enumerate(set(labels)):
        id2label[i] = label
        label2id[label] = i
    return id2label, label2id


# custom function to load the representations stored in the pickle file. the pickle file was written in an append mode
def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        representation_data = []
        while True: #change this to True when loading the entire data
            try:
                data = pkl.load(pickle_file)
                representation_data.append(data)
            except EOFError:
                break
    return representation_data


# Usage
print("[INFO] Loading the representations of images from the pickle file...")
pickle_file_path = '<load the test pickle file with pre- and post- projection embeddings of the images>'
representation_data_train = load_pickle_file(pickle_file_path)

# Usage
print("[INFO] Loading the representations of images from the pickle file...")
pickle_file_path = '<load the test pickle file with pre- and post- projection embeddings of the images>'
representation_data_test = load_pickle_file(pickle_file_path)

# get the labels using representation_data_test and representation_data_train's first element
def get_labels(representation_data, files):
    labels = []
    for data in representation_data:
        labels.append(data[0].split('/')[-2].strip().replace('_', ' '))
    return labels

train_labels = get_labels(representation_data_train, [x[0] for x in representation_data_train])
test_labels = get_labels(representation_data_test, [x[0] for x in representation_data_test])
id2label, label2id = get_label_mappings(train_labels)

train_labels = [label2id[label] for label in train_labels]
test_labels = [label2id[label] for label in test_labels]

# given the entries in train/val/test_files, match them with representation_data[0] and construct two set of train/validation/test_embeddings (pre and post) for the second and third elements of representation_data
def get_embeddings(representation_data, setting):
    embeddings = []
    for data in representation_data:
        if setting == 'pre':
            embeddings.append(data[1])
        elif setting == 'post':
            embeddings.append(data[2])       
    # reshape the embeddings to (N, 1, 1024) to (N, 1024) for 'pre' and [N, 1, 256, 4096] to (N, 256, 4096) for 'post'
    if setting == 'pre':
        embeddings = [embedding.reshape(embedding.shape[1]) for embedding in embeddings]
    elif setting == 'post':
        embeddings = [embedding.reshape(embedding.shape[1], embedding.shape[2]) for embedding in embeddings]
    return [embeddings]

train_data_pre = get_embeddings(representation_data_train, 'pre')
train_embeddings_pre, train_labels = np.array(train_data_pre[0]), np.array(train_labels)
test_data_pre = get_embeddings(representation_data_test, 'pre')
test_embeddings_pre, test_labels = np.array(test_data_pre[0]), np.array(test_labels)

train_data_post = get_embeddings(representation_data_train, 'post')
train_embeddings_post, _ = np.array(train_data_post[0]), np.array(train_labels)
test_data_post = get_embeddings(representation_data_test, 'post')
test_embeddings_post, _ = np.array(test_data_post[0]), np.array(test_labels)

# a multi layer perceptron to train on the representations
# the input to the MLP is of the size [1, 1024] and the number of classes are suppied as input to the function
def mlp_model(input_size, num_classes):
    # define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 3600),
        torch.nn.ReLU(),
        torch.nn.Linear(3600, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 600),
        torch.nn.ReLU(),
        torch.nn.Linear(600, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes)
    )
    return model

# a multi layer perceptron to train on the representations
# the input to the MLP is of the size [1, 4096] and the number of classes are suppied as input to the function
def mlp_model2(input_size, num_classes):
    # define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1536),
        torch.nn.ReLU(),
        torch.nn.Linear(1536, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes)
    )
    return model


# prepare the pre data for training the MLP
train_tensors_pre = torch.FloatTensor(train_embeddings_pre)
test_tensors_pre = torch.FloatTensor(test_embeddings_pre)

train_labels_tensor = torch.LongTensor(train_labels)
test_labels_tensor = torch.LongTensor(test_labels)




# train the model, given the model_name
def train_model(model, train_data, train_labels, val_data, val_labels, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(batch_idx, epoch)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # print(loss)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                all_preds.extend(pred.view(-1).cpu().numpy())
                # print(all_preds)
                all_labels.extend(target.view(-1).cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
                # print(correct)

        val_loss /= len(val_loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch}, Accuracy: {100. * correct / len(val_loader.dataset)}, Macro F1: {macro_f1}')

def evaluate_model(model, test_data, test_labels, batch_size=32):
    # Create DataLoader for test set
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()  # Set the model to evaluation mode
    
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_labels.extend(target.view(-1).cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate metrics
    accuracy = 100. * correct / len(test_loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Test Accuracy: {accuracy}, Macro F1 Score: {macro_f1}')


# Define MLP model
input_size = 1024  # Adjust based on your pre-embedding size
num_classes = len(set(train_labels))  # Number of unique labels
mlp = mlp_model(input_size, num_classes)

# Convert list of embeddings and labels to PyTorch tensors
train_tensors_pre = torch.FloatTensor(train_embeddings_pre)
train_labels_tensor = torch.LongTensor(train_labels)
test_tensors_pre = torch.FloatTensor(test_embeddings_pre)
test_labels_tensor = torch.LongTensor(test_labels)

# Train the model
train_model(mlp, train_tensors_pre, train_labels_tensor, test_tensors_pre, test_labels_tensor, num_epochs=100)
# Evaluate the model
evaluate_model(mlp, test_tensors_pre, test_labels_tensor)


# now train the MLP with the post embeddings
train_tensors_post = torch.FloatTensor(train_embeddings_post)
test_tensors_post = torch.FloatTensor(test_embeddings_post)

# shrink the second dimension of train_tensors_post by taking average along that dimension -- this is for mlp2
train_tensors_post = torch.mean(train_tensors_post, dim=1)
test_tensors_post = torch.mean(test_tensors_post, dim=1)

# # Define MLP2 model with input size as [4096]
input_size = 4096
num_classes = len(set(train_labels))  # Number of unique labels
mlp2 = mlp_model2(input_size, num_classes)

# Train the model on a GPU
train_model(mlp2, train_tensors_post, train_labels_tensor, test_tensors_post, test_labels_tensor, num_epochs=100)
# Evaluate the model on a GPU
evaluate_model(mlp2, test_tensors_post, test_labels_tensor)