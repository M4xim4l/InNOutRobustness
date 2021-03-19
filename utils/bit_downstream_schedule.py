import numpy as np
import utils.train_types.schedulers as schedulers
import utils.train_types.optimizers as optimizers

def get_bit_scheduler_optim_configs(dataset_size, dataloader_length, lr=0.001, decay=0, nesterov=False):
    # Bit hyperrule
    # https://github.com/google-research/big_transfer/blob/0bb237d6e34ab770b56502c90424d262e565a7f3/bit_hyperrule.py#L30
    if dataset_size < 20_000:
        decay_steps = [200, 300, 400]
        total_updates = 500
        warmup_steps = 100
    elif dataset_size < 250_000:
        decay_steps = [3000, 6000, 9000]
        total_updates = 10000
        warmup_steps = 500
    # elif dataset_size < 500_000:
    #     decay_steps = [4500, 9000, 13500]
    #     total_updates = 15000
    #     warmup_steps = 500
    else:
        decay_steps = [6000, 12_000, 18_000]
        total_updates = 20000
        warmup_steps = 500

    decay_rate = 0.1
    epochs = int(np.ceil(total_updates / dataloader_length))

    # convert from batch to epoch
    decay_epochs = [decay_step / dataloader_length for decay_step in decay_steps]
    warmup_length_epochs = warmup_steps / dataloader_length

    scheduler_config = schedulers.create_piecewise_consant_scheduler_config(epochs, decay_epochs, decay_rate,
                                                                            warmup_length=warmup_length_epochs)

    #########OPTIMIZER
    optimizer_config = optimizers.create_optimizer_config('SGD', lr, momentum=0.9,
                                                          weight_decay=decay, nesterov=nesterov)

    return scheduler_config, optimizer_config, epochs

def get_ssL_bit_scheduler_optim_configs(dataset_size, dataloader_length, lr=0.001, decay=0, nesterov=False):
    # Bit hyperrule
    # https://github.com/google-research/big_transfer/blob/0bb237d6e34ab770b56502c90424d262e565a7f3/bit_hyperrule.py#L30
    if dataset_size < 10_000:
        decay_steps = [200, 300, 400]
        total_updates = 500
        warmup_steps = 100
    if dataset_size < 20_000:
        decay_steps = [200, 300, 400]
        total_updates = 500
        warmup_steps = 100
    elif dataset_size < 250_000:
        decay_steps = [3000, 6000, 9000]
        total_updates = 10000
        warmup_steps = 500
    # elif dataset_size < 500_000:
    #     decay_steps = [4500, 9000, 13500]
    #     total_updates = 15000
    #     warmup_steps = 500
    else:
        decay_steps = [6000, 12_000, 18_000]
        total_updates = 20000
        warmup_steps = 500

    decay_rate = 0.1
    epochs = int(np.ceil(total_updates / dataloader_length))

    # convert from batch to epoch
    decay_epochs = [decay_step / dataloader_length for decay_step in decay_steps]
    warmup_length_epochs = warmup_steps / dataloader_length

    scheduler_config = schedulers.create_piecewise_consant_scheduler_config(epochs, decay_epochs, decay_rate,
                                                                            warmup_length=warmup_length_epochs)

    #########OPTIMIZER
    optimizer_config = optimizers.create_optimizer_config('SGD', lr, momentum=0.9,
                                                          weight_decay=decay, nesterov=nesterov)

    return scheduler_config, optimizer_config, epochs
