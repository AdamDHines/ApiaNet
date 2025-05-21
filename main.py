#MIT License

#Copyright (c) 2025 Adam Hines & Andrew Barron

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import argparse

from apianet.src.train import TrainVision, TrainGustatory, TrainMotor

def apianet_eval(args):
    pass

def apianet_train(args):
    if args.module == 'vision':
        # Initialize the training class
        trainer = TrainVision(args)
        trainer.train()
    elif args.module == 'gustatory':
        # Initialize the training class
        trainer = TrainGustatory(args)
        trainer.train()
    # Train the motor module
    elif args.module == 'motor':
        # Initialize the training class
        trainer = TrainMotor(args)
        trainer.train()
    else:
        raise ValueError(f"Module {args.module} not recognized. Choose from ['vision', 'motor', 'gustatory', 'association']")


def parse_args():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for default configuration")

    # Training or evaluation mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode to run: training or evaluation network')
    
    # If training, specify module to train
    parser.add_argument('--module', type=str, default='vision', choices=['vision', 'motor', 'gustatory', 'association'],
                        help='ApiaNet module to train - association training requires pretrained modules')
    
    # Training parameters
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs to train modules')
    
    # Directory paths
    parser.add_argument('--models_dir', type=str, default='./apianet/models/',
                        help='Directory to save and load models')
    
    # Model names
    parser.add_argument('--vision_model', type=str, default='VisionModel.pth',
                        help='Name of the vision model for saving/loading')
    parser.add_argument('--gustatory_model', type=str, default='GustatoryModel.pth',
                        help='Name of the gustatory model for saving/loading')
    parser.add_argument('--motor_model', type=str, default='MotorModel.pth',
                        help='Name of the motor model for saving/loading')
    
    # Output base configuration
    args = parser.parse_args()

    # Run the user-chosen mode
    if args.mode == 'train':
        apianet_train(args)
    elif args.mode == 'eval':
        apianet_eval(args)

if __name__ == "__main__":
    args = parse_args()