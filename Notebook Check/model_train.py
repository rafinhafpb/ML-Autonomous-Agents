# TASK 2 IMITATION LEARNING

model_il = TinyTransformerDecoder(input_dim, output_dim, embedding_dim=embedding_dim, num_layers=num_layers, num_heads=num_heads)
dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_il.parameters(), lr=0.001)

# Total number of epochs
num_epochs = 500

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//2, gamma=0.1)

# Training loop
best_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs+1):
    model_il.train()
    running_loss = 0.0
    for observations, actions in dataloader:
        optimizer.zero_grad()
        outputs = model_il(observations.float())
        loss = criterion(outputs.view(-1, output_dim), actions.view(-1, output_dim).argmax(dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    
    # Save the model state if it has the lowest loss so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state = model_il.state_dict()

    # Step the scheduler
    scheduler.step()

# Load the best model state
if best_model_state is not None:
    model_il.load_state_dict(best_model_state)

