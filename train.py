import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

# Argument parser for Graph WaveNet hyperparameters and experimental settings
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:3',help='Device for training')
parser.add_argument('--data',type=str,default='data/METR-LA',help='Path to preprocessed traffic data (METR-LA or PEMS-BAY)')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='Predefined adjacency matrix path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='Type of predefined adjacency: forward, backward, or doubletransition')
parser.add_argument('--gcn_bool',action='store_true',help='Enable graph convolution layer for spatial dependencies')
parser.add_argument('--aptonly',action='store_true',help='Use only adaptive adjacency matrix (no predefined graph)')
parser.add_argument('--addaptadj',action='store_true',help='Add adaptive adjacency matrix for learning hidden spatial dependencies')
parser.add_argument('--randomadj',action='store_true',help='Randomly initialize adaptive adjacency matrix')
parser.add_argument('--seq_length',type=int,default=12,help='Input sequence length (paper uses 12 for 1 hour with 5-min intervals)')
parser.add_argument('--nhid',type=int,default=32,help='Hidden dimension after initial linear transformation')
parser.add_argument('--in_dim',type=int,default=2,help='Input feature dimension (current + previous timestep)')
parser.add_argument('--num_nodes',type=int,default=207,help='Number of sensors/nodes in the graph (207 for METR-LA)')
parser.add_argument('--batch_size',type=int,default=64,help='Training batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='Initial learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='Dropout rate for regularization')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='L2 regularization weight')
parser.add_argument('--epochs',type=int,default=100,help='Number of training epochs')
parser.add_argument('--print_every',type=int,default=50,help='Print frequency during training')
parser.add_argument('--save',type=str,default='./garage/metr',help='Model save path')
parser.add_argument('--expid',type=int,default=1,help='Experiment identifier')

args = parser.parse_args()

def main():
    # Graph WaveNet training pipeline following the paper's methodology
    
    # Setup device for computation
    device = torch.device(args.device)
    
    # Load predefined adjacency matrix and sensor information
    # Paper: "The explicit graph structure does not necessarily reflect the true dependency"
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    
    # Load dataset with sliding window approach
    # Input: past 12 time steps (1 hour), Output: future 12 time steps (1 hour)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']  # Normalization scaler for inverse transform during evaluation
    
    # Convert adjacency matrices to tensors
    # Paper supports multiple adjacency matrices: forward, backward, and adaptive
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    # Initialize adaptive adjacency matrix
    # Paper: "learn it through node embedding" for adaptive dependency matrix
    if args.randomadj:
        adjinit = None  # Random initialization of adaptive matrix
    else:
        adjinit = supports[0]  # Initialize from predefined adjacency

    # Option to use only adaptive adjacency matrix
    # Paper: comparing different adjacency configurations (Table 3)
    if args.aptonly:
        supports = None

    # Initialize Graph WaveNet trainer
    # Contains the complete architecture: Gated TCN + GCN + Skip connections
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("start training...", flush=True)
    
    # Training history tracking
    his_loss = []
    val_time = []
    train_time = []
    
    # Main training loop
    for i in range(1, args.epochs + 1):
        # Training phase
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        
        # Shuffle training data for each epoch
        dataloader['train_loader'].shuffle()
        
        # Iterate through training batches
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # Data preparation: [batch, time, nodes, features] → [batch, features, nodes, time]
            # Paper processes data as [N, D, S] (nodes, features, sequence)
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)  # Transpose for model input format
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            
            # Forward pass and backward pass
            # Paper uses MAE loss: L = (1/TND) ∑∑∑ |X̂(t+i)_jk - X(t+i)_jk|
            metrics = engine.train(trainx, trainy[:, 0, :, :])  # Only predict first feature (traffic flow)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        
        t2 = time.time()
        train_time.append(t2 - t1)
        
        # Validation phase
        # Paper: "Unlike previous works, our GraphWaveNet outputs X̂(t+1):(t+T) as a whole
        # rather than generating X̂(t) recursively through T steps"
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            
            # Non-autoregressive prediction (all 12 future steps at once)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        # Calculate epoch averages
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        
        # Save model checkpoint for each epoch
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Testing phase with best model
    # Load the best model based on validation loss
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    # Prediction on test set
    # Paper: outputs all T future time steps simultaneously
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # Ground truth for comparison

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            # Forward pass: [N, D, S] → [N, D, T] where T=12 future time steps
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    # Concatenate all predictions
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # Evaluation for each prediction horizon (1 to 12 steps ahead)
    # Paper evaluates performance at different prediction horizons
    amae = []
    amape = []
    armse = []
    
    for i in range(12):
        # Inverse transform predictions back to original scale
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        
        # Calculate metrics: MAE, MAPE, RMSE
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    # Overall performance across all horizons
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    
    # Save the best model
    torch.save(engine.model.state_dict(), args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
