import os
import torch

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, device, patience, epochs, result_path, fold_logger):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
    
    def train(self):
        for epoch in range(1,self.epochs+1):
            loss_train, acc_train = self.train_step()
            loss_val, acc_val = self.valid_step()

            self.logger.info(f'Epoch({epoch}) t_loss:{loss_train:.3f} t_acc:{acc_train:2f} v_loss:{loss_val:.3f} v_acc:{acc_val:.2f}')

            if loss_val < best:
                best = loss_val
                torch.save(self.model.state_dict(), self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break
        
        loss_test, acc_test = self.test_step()
        self.logger.info(f'Test loss:{loss_test:.3f} acc:{acc_test:.2f}')
        return loss_test, acc_test
    
    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0
        correct = 0
        for batch in self.train_loader:
            x,y = batch
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item(0) * x.shape[0] #reduction of criterion -> mean
            correct +=  sum(output.argmax(dim=1) == y).item()
        
        return total_loss/len(self.train_loader), correct/len(self.train_loader)
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for batch in self.valid_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                
                loss = self.loss_fn(output, y)
                total_loss += loss.item(0) * x.shape[0] #reduction of criterion -> mean
                correct +=  sum(output.argmax(dim=1) == y).item()
                
        return total_loss/len(self.train_loader), correct/len(self.train_loader)

    def test(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()

        with torch.no_grad():
            result = []
            for batch in self.test_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)

                result.append(output)

        return torch.cat(result,dim=0).cpu().numpy()