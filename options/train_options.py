from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=1000, help='frequency for showing training results on screen in steps')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency for saving the latest results in steps')
        self.parser.add_argument('--save_periodically_freq', type=int, default=50000, help='frequency for saving the checkpoint periodically in steps')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--mode', default='train', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--validate_train_split', type=bool, nargs= '?', const=True, default=False, help='')

        # for training
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--nepochs', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--max_val_samples', type=int, default=50, help='max number of samples to take into account when validating')
        
        # for constraints loss time scheduler
        self.parser.add_argument('--update_constraints_scheduler_freq', type=int, default=60, help='Frequency (in ticks) to update the temporal parameter of the losses')
        self.parser.add_argument('--constraints_time_factor', type=float, default=1.1, help='Factor for which the temporal parameter of the losses is multiplied')

        # optimizer
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of optimizer')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for optimizer')
        self.parser.add_argument('--lr_power', type=float, default=0.0, help='power of learning rate policy')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
        self.parser.add_argument('--wd', type=float, default=0.0005, help='weight decay for sgd')
        
        # optimizer scheduler
        self.parser.add_argument('--optim_metric_name', type=str, default="val_mask_ap", help='Metric to store best checkpoint and adjust lr scheduler')
        self.parser.add_argument('--optimizer', type=str, default="SGD", help='Optimizer used (SGD supported for now)')
        self.parser.add_argument('--lr_scheduler', type=bool, nargs= '?', const=True, default=False, help='')
        self.parser.add_argument('--lr_scheduler_method', type=str, default="plateau", help='[plateau|tickBased] scheduler based on a value to max/minimise or number of ticks')
        self.parser.add_argument('--lr_scheduler_nticks', type=str, default=None, help='Number of ticks to update learning rate')
        self.parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='Scheduler factor new_lr := factor * lr')
        self.parser.add_argument('--lr_scheduler_patience', type=int, default=10, help='Number of epochs with no improvement after which learning rate will be reduced')
        self.parser.add_argument('--lr_scheduler_threshold', type=float, default=0.01, help='Threshold for measuring the new optimum (absolute value)')
        self.parser.add_argument('--lr_scheduler_mode', type=str, default="max", help='[min|max] whether to mimize or maximise the metric given')
        self.parser.add_argument('--lr_scheduler_eps', type=float, default=5e-5, help='minimum learning rate that can be acheived')

        self.isTrain = True
