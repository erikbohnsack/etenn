import torch


def get_windows(vis, time_str):
    loss_window = vis.line(
        Y=torch.zeros((1)),
        X=torch.zeros((1)),
        opts=dict(xlabel='epoch', ylabel='Loss', title=('Loss\n' + time_str), legend=['Training', 'Validation']))

    sub_loss_window = vis.line(
        Y=torch.zeros((1)),
        X=torch.zeros((1)),
        opts=dict(xlabel='epoch', ylabel='Sub-Loss', title=('Sub-Loss\n' + time_str),
                  legend=['TrainReg', 'ValidReg',
                          'TrainEul', 'ValidEul',
                          'TrainCla', 'ValidCla']))

    recall_window = vis.line(
        Y=torch.zeros((1)),
        X=torch.zeros((1)),
        opts=dict(xlabel='epoch', ylabel='Recall', title=('Recall\n' + time_str), legend=['Training', 'Validation']))

    precision_window = vis.line(
        Y=torch.zeros((1)),
        X=torch.zeros((1)),
        opts=dict(xlabel='epoch', ylabel='Precision', title=str('Precision\n' + time_str),
                  legend=['Training', 'Validation']))

    return loss_window, sub_loss_window, recall_window, precision_window


def push_data(epoch, vis,
              loss_window, sub_loss_window, recall_window, precision_window,
              train_mean_loss, eval_mean_loss,
              train_scaled_L1_mean, train_classification_loss,
              train_scaled_euler_mean, eval_scaled_euler_mean,
              eval_scaled_L1_mean, eval_classification_loss,
              train_mean_recall, eval_mean_recall,
              train_mean_precision, eval_mean_precision):
    # Visualize Loss
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_mean_loss]).unsqueeze(0).cpu(),
             win=loss_window,
             name='Training',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_mean_loss]).unsqueeze(0).cpu(),
             win=loss_window,
             name='Validation',
             update='append')

    # Visualize the sub parts of the loss function, i.e. L1 and Classification Loss
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_scaled_L1_mean]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='TrainReg',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_scaled_L1_mean]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='ValidReg',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_scaled_euler_mean]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='TrainEul',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_scaled_euler_mean]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='ValidEul',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_classification_loss]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='TrainCla',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_classification_loss]).unsqueeze(0).cpu(),
             win=sub_loss_window,
             name='ValidCla',
             update='append')

    # Visualize Recall
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_mean_recall]).unsqueeze(0).cpu(),
             win=recall_window,
             name='Training',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_mean_recall]).unsqueeze(0).cpu(),
             win=recall_window,
             name='Validation',
             update='append')

    # Visualize Precision
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([train_mean_precision]).unsqueeze(0).cpu(),
             win=precision_window,
             name='Training',
             update='append')
    vis.line(X=torch.ones((1, 1)).cpu() * epoch,
             Y=torch.Tensor([eval_mean_precision]).unsqueeze(0).cpu(),
             win=precision_window,
             name='Validation',
             update='append')
