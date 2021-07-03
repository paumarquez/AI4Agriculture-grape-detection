from .weakly_supervised import WeakSemanticSegmentationLoss

def create_loss(opt):
    if opt.model == 'fasterRCNN':
        loss = None
    elif opt.model == "ResidualUNet":
        loss = WeakSemanticSegmentationLoss(
            opt.weak_loss_init_t,
            opt.weak_loss_d,
            opt.weak_loss_margins,
            opt.weak_loss_lambda,
            opt.weak_loss_alpha,
            opt.weak_loss_cache_folder
        )
    else:
        raise Exception(f'Loss from model {opt.model} not found')

    print("loss from model [%s] was created" % (opt.model))

    return loss
