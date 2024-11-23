import torch
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from finetune import ViTLinear, VPTDeep, inference, Trainer
import yaml

from datasets import get_flower102

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'vpt_deep',
                    'The experiment with corresponding hyperparameters to run. See config.yaml')
flags.DEFINE_string('output_dir', 'run1', 'Output Directory')
flags.DEFINE_string('encoder', 'vit_b_32',
                    'Select the encoder from linear, resnet18, resnet50, vit_b_16, vit_b_32')
flags.DEFINE_string('data_dir', './flower-dataset-reduced', 'Directory with coco data')


def get_config(exp_name, encoder):
    dir_name = f'{FLAGS.output_dir}/runs-{encoder}-flower102/demo-{exp_name}'

    encoder_registry = {
        'ViTLinear': ViTLinear,
        'VPTDeep': VPTDeep,
    }

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    wd = config['wd']
    epochs = config['epochs']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    momentum = config['momentum']
    net_class = encoder_registry[config['net_class']]
    batch_size = config['batch_size']

    num_prompts = config.get('num_prompts', 10)

    return net_class, dir_name, (optimizer, lr, wd, momentum), (scheduler, epochs), batch_size, num_prompts


def main(_):
    torch.set_num_threads(2)
    torch.manual_seed(0)

    print(f"Running: {FLAGS.exp_name} with encoder {FLAGS.encoder} ------------------")

    net_class, dir_name, \
        (optimizer, lr, wd, momentum), \
        (scheduler, epochs), batch_size, num_prompts = \
        get_config(FLAGS.exp_name, FLAGS.encoder)

    train_data = get_flower102(FLAGS.data_dir, 'train')
    val_data = get_flower102(FLAGS.data_dir, 'val')
    test_data = get_flower102(FLAGS.data_dir, 'test')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    tmp_file_name = dir_name + '/best_model.pth'
    device = torch.device('cuda:0')

    writer = SummaryWriter(f'{dir_name}/lr{lr:0.6f}_wd{wd:0.6f}', flush_secs=10)


    model = VPTDeep(num_classes=102, encoder_type=FLAGS.encoder, num_prompt_tokens=num_prompts)
    model.to(device)

    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                     optimizer=optimizer, lr=lr, wd=wd, momentum=momentum,
                     scheduler=scheduler, epochs=epochs,
                     device=device)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"lr: {lr:0.7f}, wd: {wd:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")

    print("Training complete--------------------")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{dir_name}/best_model.pth'))
    inference(test_dataloader, model, device, result_path=dir_name + '/test_predictions.txt')


if __name__ == '__main__':
    app.run(main)