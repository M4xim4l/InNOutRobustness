def create_optimizer_config(optimizer_type, lr, weight_decay=0, momentum=0, nesterov=False, mixed_precision=False):
    optimizer_config = {'optimizer_type': optimizer_type, 'lr': lr, 'weight_decay': weight_decay,
                        'mixed_precision': mixed_precision}
    if optimizer_type == 'SGD':
        optimizer_config['momentum'] = momentum
        optimizer_config['nesterov'] = nesterov
    return optimizer_config


def add_cosine_swa_to_optimizer_config(epochs, cycle_length, update_frequency,
                                       virtual_schedule_length, virtual_schedule_swa_end,
                                       virtual_schedule_lr, scheduler_config):
    swa_config = {'epochs': epochs, 'cycle_length': cycle_length,
                  'update_frequency': update_frequency,
                  'swa_schedule_type': 'cosine',
                  'virtual_schedule_length': virtual_schedule_length,
                  'virtual_schedule_swa_end': virtual_schedule_swa_end,
                  'virtual_schedule_lr': virtual_schedule_lr}
    scheduler_config['swa_config'] = swa_config


def add_constant_swa_to_optimizer_config(epochs, update_frequency,
                                       virtual_schedule_lr, scheduler_config):
    swa_config = {'epochs': epochs,
                  'update_frequency': update_frequency,
                  'swa_schedule_type': 'constant',
                  'virtual_schedule_lr': virtual_schedule_lr}
    scheduler_config['swa_config'] = swa_config