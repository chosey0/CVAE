
def init_model(lr, beta, latent_dim, metric, model_init_config, training_config):
    # Make the model
    model = training_config.model(latent_dim=latent_dim, **model_init_config)

    # Make the loss and optimizer
    criterion = training_config.criterion(beta=beta, metric=metric)
    optimizer = training_config.optimizer(model.parameters(), lr=lr)
    
    model.to(training_config.device)
    criterion.to(training_config.device)
    
    training_config.__setattr__("model", model)
    training_config.__setattr__("criterion", criterion)
    training_config.__setattr__("optimizer", optimizer)
    return model, criterion, optimizer