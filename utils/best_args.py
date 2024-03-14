best_args = {
    "seq-cifar10": {
        "sgd": {-1: {"lr": 0.03, "batch_size": 10, "n_epochs": 1}},
        "onlinevt": {
            200: {
                "lr": 0.03,
                "minibatch_size": 10,
                "alpha": 0.4,  # 0.1
                "beta": 0.5,  # 0.5.
                "weight_decay":0.2,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0,
                "L1": 0,
            },
            500: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.2,  # 0.2,
                "beta": 0.5,  # 0.5,
                "batch_size": 10,
                "distill_ce": 0.1,
                "ce": 1,
                "n_epochs": 1,
                "wd_reg": 0,  # 0.00001
                "L1": 0,  # 0.00001  0.00005
            },
            1000: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.1,
                "beta": 0.5,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0.0,  # 0.00001
                "L1": 0.000,  # 0.00001  0.00005 0.00001
            },
        },
    },
    "seq-cifar100": {
        "sgd": {-1: {"lr": 0.03, "batch_size": 10, "n_epochs": 1}},
        "onlinevt": {
            200: {
                "lr": 0.03,
                "minibatch_size": 10,
                "alpha": 0.4,  # 0.1
                "beta": 0.5,  # 0.5.
                "weight_decay":0.2,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0,
                "L1": 0,
            },
            500: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.2,  # 0.2,
                "beta": 0.5,  # 0.5,
                "batch_size": 10,
                "distill_ce": 0.1,
                "ce": 1,
                "n_epochs": 1,
                "wd_reg": 0,  # 0.00001
                "L1": 0,  # 0.00001  0.00005
            },
            1000: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.1,
                "beta": 0.5,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0.0,  # 0.00001
                "L1": 0.000,  # 0.00001  0.00005 0.00001
            },
        },
    },
    "seq-imagenet-animals": {
        "sgd": {-1: {"lr": 0.1, "batch_size": 10, "n_epochs": 1}},
        "onlinevt": {
            200: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.4,  # 0.1
                "beta": 0.5,  # 0.5.
                #"weight_decay":0.0005,
                #"momentum":0.2,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0,
                "L1": 0,
            },
            500: {
                "lr": 0.1,
                "minibatch_size": 36,
                "alpha": 0.2,  # 0.2,
                "beta": 0.5,  # 0.5,
                "batch_size": 10,
                "distill_ce": 0.1,
                "ce": 1,
                "n_epochs": 1,
                "wd_reg": 0,  # 0.00001
                "L1": 0,  # 0.00001  0.00005
            },
            1000: {
                "lr": 0.1,
                "minibatch_size": 32,
                "alpha": 0.1,
                "beta": 0.5,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0.0,  # 0.00001
                "L1": 0.000,  # 0.00001  0.00005 0.00001
            },
        },
    },

    "seq-imagenet-r": {
        "sgd": {-1: {"lr": 0.1, "batch_size": 10, "n_epochs": 1}},
        "onlinevt": {
            200: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.4,  # 0.1
                "beta": 0.5,  # 0.5.
                #"weight_decay":0.0005,
                #"momentum":0.2,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0,
                "L1": 0,
            },
            600: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.2,  # 0.2,
                "beta": 0.5,  # 0.5,
                "batch_size": 10,
                "distill_ce": 0.1,
                "ce": 1,
                "n_epochs": 1,
                "wd_reg": 0,  # 0.00001
                "L1": 0,  # 0.00001  0.00005
            },
            1000: {
                "lr": 0.1,
                "minibatch_size": 10,
                "alpha": 0.1,
                "beta": 0.5,
                "distill_ce": 0.1,
                "ce": 1,
                "batch_size": 10,
                "n_epochs": 1,
                "wd_reg": 0.0,  # 0.00001
                "L1": 0.000,  # 0.00001  0.00005 0.00001
            },
        },
    },
}

