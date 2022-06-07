```bash
pip install wandb

wandb login
```



```bash
cd my_project
wandb init

python wandb_tutorial.py

# 실행한 뒤, wandb 사이트 접속해서 로그 확인
```



wandb 예시 : https://github.com/wandb/examples



wandb 사용법

- `wandb.init()`
  - cmd에서 `wandb init` 과 동일한 역할. 하나만해도 됨
- `wandb.config.update(args)`
- `wandb.watch(model)`

```python
import wandb


# main()함수 앞에
def main():
  #======================================================#
	wandb.init(project="project-name", reinit=True)
  wandb.run.name = 'my_run1'
  wandb.run.save()
  
  #======================================================#
  # 방법 1) 기존대로 + wandb.config.update
  parser = argparse.ArgumentParser(description = 'abc')
  args = parser.parse_args()
  args.epochs = 4
  args.batch_size = 32
  wandb.config.update(args)
  
  # 방법 2) 기존대로 + wandb.config.update
  parser = argparse.ArgumentParser(description = 'abc')
  parser.add_argument('-b', '--epochs', type=int, default=4, metavar='N',
                     help='input batch size for training (default: 8)')
  parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                     help='input batch size for training (default: 8)')
	args = parser.parse_args()
  wandb.config.update(args)
  
  # 방법 3) wandb.config.하이퍼파라미터  
  wandb.config.epochs = 4
  wandb.config.batch_size = 32
  
  # 방법 4) 처음부터
  wandb.init(config={'epochs':4,})
  #======================================================#
  model = MyModel()
  opt = optim.Adam()
  loss_fun = my_loss_function()
  
  wandb.watch(model, loss_fun, log="all", log_freq=10)
  #======================================================#
  
  for epoch in range(total_epochs):
    ....
    
    avg_loss = xxx
    wandb.log({"loss":avg_loss}, step = epoch)
```



프로젝트 명 1 ( `wandb.init(project="project-name", reinit=True)`  )

- 실행 명 1-1 ( `wandb.run.name = 'my_run1'` )
- 실행 명 1-2 ( `wandb.run.name = 'my_run2'` )

- …



로그 기록하기 ( `wandb.log()` )

```python
# result
wandb.log({"Test Accuracy" : test_acc,
          "Test Loss" : test_loss})

# plot 1
wandb.log({"gradients": wandb.Histogram(numpy_array_or_sequence)})
wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})

# plot 2
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some interesting numbers')
wandb.log({"chart": plt})

# plot 3
wandb.log({'pr': wandb.plots.precision_recall(y_test, y_probas, labels)})
```



종합

```python
def train(model, loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0  
    for epoch in tqdm(range(config.epochs)):
        cumu_loss = 0
        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)
    
            outputs = model(images)
            loss = criterion(outputs, labels)
            cumu_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            example_ct +=  len(images)

        avg_loss = cumu_loss / len(loader)
        wandb.log({"loss": avg_loss}, step=epoch)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")
        
        
def run(config=None):
    wandb.init(project='test-pytorch', entity='pebpung', config=config)
      
    config = wandb.config

    train_loader = make_loader(batch_size=config.batch_size, train=True)
    test_loader = make_loader(batch_size=config.batch_size, train=False)

    model = ConvNet(config.kernels, config.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train(model, train_loader, criterion, optimizer, config)
    test(model, test_loader)
    return model
```

