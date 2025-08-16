from argparse import ArgumentParser
import torch 


parser = ArgumentParser()
parser.add_argument('weights_path', help='Path to the weights of the pretrained backbone')
parser.add_argument('output_path', help='Path to save the output checkpoint file (optional)', required=False, default=None)

args = parser.parse_args()
args.output_path = args.output_path or args.weights_path.replace('.pth', '_dinov3_checkpoint.pth')

sd = torch.load(args.weights_path)
sd = {
    f'backbone.{key}': val for key, val in sd.items()
}
sd = {
    "teacher": sd
}
torch.save(sd, args.output_path)