import argparse


class Options():
    def __init__(self, cuda='true', task_name='facescrub', epoch_size=5000, batch_size=64, learning_rate=0.0002,
                 model_arch='distancegan', image_size=64, gan_curriculum=10000, starting_rate=0.01, default_rate=0.5,
                 style_A=None, style_B=None, constraint=None, constraint_type=None, n_test=200, update_interval=3,
                 log_interval=50, image_save_interval=1000, model_save_interval=10000, result_path='./results/',
                 model_path='./models/', use_self_distance=False, unnormalized_distances=False, max_items=300,
                 use_reconst_loss=False, num_layers=4, num_layers_second_gan=4, starting_correlation_rate=1,
                 default_correlation_rate=1, number_of_samples=500, not_all_samples=False, port=8097, test_mode=False,
                 which_epoch_load=3, one_sample_index=0, continue_training=False, indiv_gan_rate=1,fixed_g1=False,
                 pretrained_g1_path_A=None, pretrained_g1_path_B=None, pretrained_g2_path_A=None, pretrained_g2_path_B=None,
                 start_from_pretrained_g1=False, start_from_pretrained_g2=False, one_sample_train=False):
        self.cuda = cuda
        self.task_name = task_name
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_arch = model_arch
        self.image_size = image_size
        self.gan_curriculum = gan_curriculum
        self.starting_rate = starting_rate
        self.default_rate = default_rate
        self.style_A = style_A
        self.style_B = style_B
        self.constraint = constraint
        self.constraint_type = constraint_type
        self.n_test = n_test
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.image_save_interval = image_save_interval
        self.model_save_interval = model_save_interval
        self.result_path = result_path
        self.model_path = model_path
        self.use_self_distance = use_self_distance
        self.unnormalized_distances = unnormalized_distances
        self.max_items = max_items
        self.use_reconst_loss = use_reconst_loss
        self.num_layers = num_layers
        self.num_layers_second_gan = num_layers_second_gan
        self.starting_correlation_rate = starting_correlation_rate
        self.default_correlation_rate = default_correlation_rate
        self.number_of_samples = number_of_samples
        self.not_all_samples = not_all_samples
        self.port = port
        self.test_mode = test_mode
        self.which_epoch_load = which_epoch_load
        self.one_sample_index = one_sample_index
        self.continue_training = continue_training
        self.indiv_gan_rate = indiv_gan_rate
        self.fixed_g1 = fixed_g1
        self.pretrained_g1_path_A = pretrained_g1_path_A
        self.pretrained_g1_path_B = pretrained_g1_path_B
        self.pretrained_g2_path_A = pretrained_g2_path_A
        self.pretrained_g2_path_B = pretrained_g2_path_B
        self.start_from_pretrained_g1 = start_from_pretrained_g1
        self.start_from_pretrained_g2 = start_from_pretrained_g2
        self.one_sample_train = one_sample_train

    @classmethod
    def from_cmd(cls):
        cmd_opt = CmdOptions()
        args = cmd_opt.get_args()
        args.__class__ = Options
        return args


class CmdOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch implementation of DistanceGAN based on DiscoGAN')
        self.parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
        self.parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
        self.parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
        self.parser.add_argument('--model_arch', type=str, default='distancegan',
                                 help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
        self.parser.add_argument('--image_size', type=int, default=64,
                                 help='Image size. 64 for every experiment in the paper')
        self.parser.add_argument('--gan_curriculum', type=int, default=10000,
                                 help='Strong GAN loss for certain period at the beginning')
        self.parser.add_argument('--starting_rate', type=float, default=0.01,
                                 help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
        self.parser.add_argument('--default_rate', type=float, default=0.5,
                                 help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
        self.parser.add_argument('--style_A', type=str, default=None,
                                 help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--style_B', type=str, default=None,
                                 help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--constraint', type=str, default=None,
                                 help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
        self.parser.add_argument('--constraint_type', type=str, default=None,
                                 help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
        self.parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')
        self.parser.add_argument('--update_interval', type=int, default=3, help='')
        self.parser.add_argument('--log_interval', type=int, default=50,
                                 help='Print loss values every log_interval iterations.')
        self.parser.add_argument('--image_save_interval', type=int, default=1000,
                                 help='Save test results every log_interval iterations.')
        self.parser.add_argument('--model_save_interval', type=int, default=10000,
                                 help='Save models every log_interval iterations.')
        self.parser.add_argument('--result_path', type=str, default='./results/')
        self.parser.add_argument('--model_path', type=str, default='./models/')
        self.parser.add_argument('--use_self_distance', action='store_true',
                                 help="use distance for top and bottom half of the image")
        self.parser.add_argument('--unnormalized_distances', action='store_true',
                                 help='do not normalize distances by expecatation and std')
        self.parser.add_argument('--max_items', type=int, default=300,
                                 help='maximum number of items to use for expectation and std calculation')
        self.parser.add_argument('--use_reconst_loss', action='store_true',
                                 help='add reconstruction loss in addition to distance loss')
        self.parser.add_argument('--num_layers', type=int, default=4,
                                 help='Number of convolutional layers in G (equal number of deconvolutional layers exist)')
        self.parser.add_argument('--num_layers_second_gan', type=int, default=4,
                                 help='Number of convolutional layers in G_2 (equal number of deconvolutional layers exist)')

        self.parser.add_argument('--starting_correlation_rate', type=float, default=1)
        self.parser.add_argument('--default_correlation_rate', type=float, default=1)
        self.parser.add_argument('--number_of_samples', type=int, default=500)
        self.parser.add_argument('--not_all_samples', action='store_true')
        self.parser.add_argument('--port', type=int, default=8097)
        self.parser.add_argument('--test_mode', action='store_true')
        self.parser.add_argument('--which_epoch_load', type=int, default=3)
        self.parser.add_argument('--one_sample_index', type=int, default=0)
        self.parser.add_argument('--continue_training', action='store_true')
        self.parser.add_argument('--indiv_gan_rate', type=float, default=1)
        self.parser.add_argument('--fixed_g1', action='store_true')
        self.parser.add_argument('--pretrained_g1_path_A', type=str, default=None)
        self.parser.add_argument('--pretrained_g1_path_B', type=str, default=None)
        self.parser.add_argument('--pretrained_g2_path_A', type=str, default=None)
        self.parser.add_argument('--pretrained_g2_path_B', type=str, default=None)
        self.parser.add_argument('--start_from_pretrained_g1', action='store_true')
        self.parser.add_argument('--start_from_pretrained_g2', action='store_true')
        self.parser.add_argument('--one_sample_train', action='store_true')
        
    def get_args(self):
        return self.parser.parse_args()


class AnglePairingOptions(CmdOptions):

    def initialize(self):
        self.parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
        self.parser.add_argument('--task_name', type=str, default='car2car', help='Set data name')
        self.parser.add_argument('--epoch_size', type=int, default=10000, help='Set epoch size')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
        self.parser.add_argument('--model_arch', type=str, default='distancegan',
                                 help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
        self.parser.add_argument('--image_size', type=int, default=64,
                                 help='Image size. 64 for every experiment in the paper')
        self.parser.add_argument('--gan_curriculum', type=int, default=10000,
                                 help='Strong GAN loss for certain period at the beginning')
        self.parser.add_argument('--starting_rate', type=float, default=0.9,
                                 help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
        self.parser.add_argument('--default_rate', type=float, default=0.9,
                                 help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
        self.parser.add_argument('--style_A', type=str, default=None,
                                 help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--style_B', type=str, default=None,
                                 help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
        self.parser.add_argument('--constraint', type=str, default=None,
                                 help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
        self.parser.add_argument('--constraint_type', type=str, default=None,
                                 help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
        self.parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')
        self.parser.add_argument('--update_interval', type=int, default=3, help='')
        self.parser.add_argument('--log_interval', type=int, default=50,
                                 help='Print loss values every log_interval iterations.')
        self.parser.add_argument('--image_save_interval', type=int, default=500,
                                 help='Save test results every log_interval iterations.')
        self.parser.add_argument('--model_save_interval', type=int, default=10000,
                                 help='Save models every log_interval iterations.')
        self.parser.add_argument('--result_path', type=str, default='./results/')
        self.parser.add_argument('--model_path', type=str, default='./models/')
        self.parser.add_argument('--log_path', type=str, default='./logs/')
        self.parser.add_argument('--use_self_distance', action='store_true',
                                 help="use distance for top and bottom half of the image")
        self.parser.add_argument('--unnormalized_distances', action='store_true',
                                 help='do not normalize distances by expecatation and std')
        self.parser.add_argument('--max_items', type=int, default=900,
                                 help='maximum number of items to use for expectation and std calculation')
        self.parser.add_argument('--use_reconst_loss', action='store_true',
                                 help='add reconstruction loss in addition to distance loss')
        self.parser.add_argument('--num_layers', type=int, default=5,
                                 help='Number of convolutional layers in G (equal number of deconvolutional layers exist)')
