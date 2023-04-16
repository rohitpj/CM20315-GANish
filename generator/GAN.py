from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, gzip, torch
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse, os, torch, time, pickle
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from utils import *
#from models import *
#import classifier as Classifier
#functions to help with the training
class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 120)
    self.fc2 = nn.Linear(120, 120)
    self.fc3 = nn.Linear(120,10)
    #self.fc4 = nn.Linear(64,10)
    #defining the 20% dropout
    self.dropout = nn.Dropout(0.2)

  def forward(self,x):
    x = x.view(x.shape[0],-1)
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.dropout(F.relu(self.fc2(x)))
    #x = self.dropout(F.relu(self.fc3(x)))
    #not using dropout on output layer
    x = F.log_softmax(self.fc3(x), dim=1)
    return x   

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image = (image * 255).astype(np.uint8)
    return imageio.imwrite(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['G_loss']))

    y2 = hist['G_loss']

    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

"""## The generator network. 
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32, batch_size=1):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.batch_size = batch_size

        ## Create a simple ANN that takes as input self.input_dim has a hidden layer and returns (1 * 28 * 28 )
        ## -- Don't forget batch size ( which is handled automatically )

        

    def forward(self, input):
        x = ## Complete using what you've defined in __init__
        ## The output is a long vector of size 1 * 28 * 28, reshape to image dimentions before return here (again, don't forget batch size). 
        x = ## Complete
        return x
"""
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32, batch_size=1):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.batch_size = batch_size

        ## Create a simple ANN that takes as input self.input_dim has a hidden layer and returns (1 * 28 * 28 )
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim*self.input_size*self.input_size),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, self.output_dim, self.input_size, self.input_size)
        return x
## Load it up.
discriminator = Classifier()
discriminator.load_state_dict(torch.load('model.pt'))

# This is the main trainer class that will generate the image.

class GANish(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type + '_' + str(args.batch_size)
        self.input_size = args.input_size
        self.z_dim = 100
        self.gen_target= args.gen_target

        # Load the trained discriminator here

        self.D = Classifier()
        #self.D = discriminator(x=self.input_size)
        self.D.load_state_dict(torch.load('model.pt'))

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

        # Fixed noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

def train(self):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        target = torch.tensor(self.gen_target)
        if self.gpu_mode: 
          target = target.cuda()

        print('Training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.save()
            self.G.train()
            epoch_start_time = time.time()
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                z_ = z_.cuda()

            # Update G network
            self.G_optimizer.zero_grad()

            G_ = self.G(z_)
            D_fake = self.D(G_)
            G_loss = self.Loss(D_fake.squeeze(), target)

            self.train_hist['G_loss'].append(G_loss.item())

            G_loss.backward()
            self.G_optimizer.step()

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1), fix=False)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        self.save()
        generate_animation(self.result_dir + '/' + self.model_name + '/' + self.model_name,  self.epoch )
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

def visualize_results(self, epoch, fix=True):
        self.G.eval()
        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = 20

        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

"""

class GANish(object):
    def __init__(self, args):
        # parameters
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type + '_' + str(args.batch_size)
        self.input_size = args.input_size
        self.z_dim = 62
        self.gen_target= args.gen_target

        ## Load a TRAINED classifier here. 
        self.D = discriminator 

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size, batch_size=self.batch_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            ## Pick a loss here. 
            self.Loss  = ## 
        else:
            ## Same loss here but no cuda. 
            self.Loss  = ## 

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')


        # Fixed noise
        ## This is the input to the generator -- you can change shape if you feel like. 
        ## Should be batch_size x z_dim
        torch.manual_seed( 20 ) 
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        ## This is what we want to generate (0 to 9 in the FMNIST dataset).
        target = # Set this to an array which defines the output class based on what you want to generate. 
        if self.gpu_mode: 
          target = target.cuda()

        print('Training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.save()     ## Helper function will save this for you. 
            self.G.train()  ## Set the generator to "train". 
            epoch_start_time = time.time()

            ## Create the input noise - keep this the same dimentions as above. 
            ## Should be batch_size x z_dim
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                z_ = z_.cuda()

            ## -------------------------------------------------------------------------- ##
            ##                            Add your code here                              ##
            ## -------------------------------------------------------------------------- ##

            ## Zero grad the Generator's optimizer.
            

            ## First the forward step (Generate some image).
            G_     = # 
            ## Now predict the category using the discriminator (the classifier you have).
            D_fake = # 
            ## The loss is how far the prediction was from what it should be 
            ## -- You have a target (defined above) based on the image you are generating. 
            ## -- D_fake is what you got as the class of the fake image.
            G_loss = # 

            ## Save the loss for inspection here. Change from .item() to whatever else depending on what you are saving.
            self.train_hist['G_loss'].append(G_loss.item())

            ## Backwards on Generator (not classifier)
            

            ## Step through the optimizer on the Genrator 

  
            ## -------------------------------------------------------------------------- ##
            ##                    No changes required below this                          ##
            ## -------------------------------------------------------------------------- ##
          

            ## Save stuff here. 
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                ## This will save the image to the correct folder. [If this does not work, make sure self.G(self.sample_z_) is generating an image]
                self.visualize_results((epoch+1), fix=False)

        ## Some stats here. 
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        ## Save both models (You don't need the discriminator, that was not updated)
        self.save()

        ## This will create a cool gif for you. 
        generate_animation(self.result_dir + '/' + self.model_name + '/' + self.model_name,  self.epoch )

        ## Plot the loss [Use this to determine how much training is required]
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)


    ## Helper functions. 
    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = 20

        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            
            samples = self.G(self.sample_z_)
        else:
            
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
# This is the main trainer class that will generate the image.

class GANish(object):
    def __init__(self, args):
        # parameters
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type + '_' + str(args.batch_size)
        self.input_size = args.input_size
        self.z_dim = 62
        self.gen_target= args.gen_target

        ## Load a TRAINED classifier here. 
        self.D = discriminator 

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size, batch_size=self.batch_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            ## Pick a loss here. 
            self.Loss  = ## 
        else:
            ## Same loss here but no cuda. 
            self.Loss  = ## 

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')


        # Fixed noise
        ## This is the input to the generator -- you can change shape if you feel like. 
        ## Should be batch_size x z_dim
        torch.manual_seed( 20 ) 
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        ## This is what we want to generate (0 to 9 in the FMNIST dataset).
        target = # Set this to an array which defines the output class based on what you want to generate. 
        if self.gpu_mode: 
          target = target.cuda()

        print('Training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.save()     ## Helper function will save this for you. 
            self.G.train()  ## Set the generator to "train". 
            epoch_start_time = time.time()

            ## Create the input noise - keep this the same dimentions as above. 
            ## Should be batch_size x z_dim
            z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                z_ = z_.cuda()

            ## -------------------------------------------------------------------------- ##
            ##                            Add your code here                              ##
            ## -------------------------------------------------------------------------- ##

            ## Zero grad the Generator's optimizer.
            

            ## First the forward step (Generate some image).
            G_     = # 
            ## Now predict the category using the discriminator (the classifier you have).
            D_fake = # 
            ## The loss is how far the prediction was from what it should be 
            ## -- You have a target (defined above) based on the image you are generating. 
            ## -- D_fake is what you got as the class of the fake image.
            G_loss = # 

            ## Save the loss for inspection here. Change from .item() to whatever else depending on what you are saving.
            self.train_hist['G_loss'].append(G_loss.item())

            ## Backwards on Generator (not classifier)
            

            ## Step through the optimizer on the Genrator 

  
            ## -------------------------------------------------------------------------- ##
            ##                    No changes required below this                          ##
            ## -------------------------------------------------------------------------- ##
          

            ## Save stuff here. 
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                ## This will save the image to the correct folder. [If this does not work, make sure self.G(self.sample_z_) is generating an image]
                self.visualize_results((epoch+1), fix=False)

        ## Some stats here. 
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        ## Save both models (You don't need the discriminator, that was not updated)
        self.save()

        ## This will create a cool gif for you. 
        generate_animation(self.result_dir + '/' + self.model_name + '/' + self.model_name,  self.epoch )

        ## Plot the loss [Use this to determine how much training is required]
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)



    ## Helper functions. 
    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = 20

        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            samples = self.G(self.sample_z_)
        else:

            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
"""



"""parsing and configuration"""
def parse_args():

    desc = "Pytorch Generative Network PGN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='PGN',
                        choices=['PGN'], help='The type of Network')
    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--gen_target', type=int, default=9)

    return check_args(parser.parse_args(args=[]))



"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args






"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # declare instance for GAN
    gan = GANish(args)
    
    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")







if __name__ == '__main__':
    main()