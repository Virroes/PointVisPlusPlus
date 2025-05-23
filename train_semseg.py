"""
Author: Benny
Date: Nov 2019
Modified: May 2023 - Added support for VisDataSet with spherical coordinates
Modified: May 2025 - Added focal loss and binary classification metrics
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
from data_utils.VisDataLoader import VisDataSet  # Import our new dataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
write = tqdm.write
import provider
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd  # Add pandas for CSV export
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Visibility classes - occluded=0, visible=1
vis_classes = ['occluded', 'visible']  
class2label = {cls: i for i, cls in enumerate(vis_classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay')
    
    # VisDataSet specific parameters
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--train_dir', type=str, default='data/train/', help='Directory with training data')
    parser.add_argument('--val_dir', type=str, default='data/val/', help='Directory with validation data')
    parser.add_argument('--k_neighbors', type=int, default=4096, help='Number of neighbors per point')
    parser.add_argument('--use_spherical', type=str2bool, default=True,
                        help='Use spherical coordinates (r, theta, phi) instead of Cartesian (x, y, z)')
    parser.add_argument('--voxel_size', type=float, default=None,
                       help='Voxel size for downsampling (None=no downsampling)')
    parser.add_argument('--max_samples_per_scene', type=int, default=None,
                       help='Max number of center points per scene (None=use all points)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Gamma parameter for focal loss (higher = more focus on hard examples)')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    def log_write(str):
        write(str)
        logger.info(str)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('vis_seg')  # Changed from sem_seg to vis_seg
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Create metrics directory
    metrics_dir = experiment_dir.joinpath('metrics/')
    metrics_dir.mkdir(exist_ok=True)
    
    # Create plots directory
    plots_dir = experiment_dir.joinpath('plots/')
    plots_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = len(vis_classes)  # Number of visibility classes
    BATCH_SIZE = args.batch_size
    POSITIVE_CLASS = 1  # 'visible' class is positive
    
    log_string(f"Using binary classification: Positive class={vis_classes[POSITIVE_CLASS]} (1), Negative class={vis_classes[1-POSITIVE_CLASS]} (0)")

    print("Start loading training data...")
    TRAIN_DATASET = VisDataSet(
        split='train', 
        data_dir=args.train_dir,
        k_neighbors=args.k_neighbors,
        use_spherical=args.use_spherical,
        voxel_size=args.voxel_size,
        max_samples_per_scene=args.max_samples_per_scene,
        seed=args.seed
    )
    TRAIN_DATASET.save_debug_samples(num_samples=10, output_dir="debug_point_clouds_train")
    
    print("Start loading val data...")
    VAL_DATASET = VisDataSet(
        split='val', 
        data_dir=args.val_dir,
        k_neighbors=args.k_neighbors,
        use_spherical=args.use_spherical,
        voxel_size=args.voxel_size,
        max_samples_per_scene=args.max_samples_per_scene,
        seed=args.seed
    )
    VAL_DATASET.save_debug_samples(num_samples=10, output_dir="debug_point_clouds_val")

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=24,
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
        persistent_workers=True,
        prefetch_factor=3
    )
    
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=24,
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    
    # Get class weights for loss weighting
    class_weights = TRAIN_DATASET.get_label_weights()
    weights = torch.FloatTensor(class_weights).cuda()
    log_string(f"Using class weights: {class_weights}")
    
    def log_batch_stats(batch_labels, prefix=""):
        """Log statistics about a batch of labels."""
        unique_labels, counts = np.unique(batch_labels, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        log_write(f"{prefix} Batch class distribution: {label_dist}")

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of val data is: %d" % len(VAL_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    # Create model instance with the correct number of classes
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    # Pass gamma parameter to use Focal Loss
    criterion = MODEL.get_loss(gamma=args.focal_gamma).cuda()
    log_string(f"Using Focal Loss with gamma={args.focal_gamma}")
    
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_accuracy = 0
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    epochs = []

    for epoch in range(start_epoch, args.epoch):
        '''Train on point-centered neighborhoods'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        # For collecting all training labels
        all_train_targets = []

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Rotate points in data augmentation if using xyz coordinates
            if not args.use_spherical:
                points = points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
            
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # [batch, features, npoints]

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * args.k_neighbors)
            loss_sum += loss.item()
            # Collect all training labels
            all_train_targets.extend(batch_label)
                
        # Calculate train metrics for this epoch
        train_loss = loss_sum / num_batches
        train_acc = total_correct / float(total_seen)
        
        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy: %f' % train_acc)
        
        # Store training metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on val data'''
        with torch.no_grad():
            num_batches = len(valDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            classifier = classifier.eval()
            
            # For binary metrics calculation
            all_preds = []
            all_targets = []

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                points = points.float().cuda()
                target = target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * args.k_neighbors)
                
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                # Collect all predictions and targets for binary metrics
                all_preds.extend(pred_val.flatten())
                all_targets.extend(batch_label.flatten())

            # Calculate overall metrics
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            val_loss = loss_sum / float(num_batches)
            val_acc = total_correct / float(total_seen)
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate binary metrics for the positive class (visible=1)
            precision = precision_score(all_targets, all_preds, pos_label=POSITIVE_CLASS)
            recall = recall_score(all_targets, all_preds, pos_label=POSITIVE_CLASS)
            f1 = f1_score(all_targets, all_preds, pos_label=POSITIVE_CLASS)
            
            log_string(f'Binary classification metrics for {vis_classes[POSITIVE_CLASS]} class:')
            log_string(f'  - Precision: {precision:.4f}')
            log_string(f'  - Recall: {recall:.4f}')
            log_string(f'  - F1 Score: {f1:.4f}')
            
            # Log class distribution
            class_dist = np.bincount(all_targets.astype(int))
            pred_dist = np.bincount(all_preds.astype(int))
            log_string(f'Class distribution in validation data: {dict(enumerate(class_dist))}')
            log_string(f'Predicted class distribution: {dict(enumerate(pred_dist))}')
            
            # Log evaluation metrics
            log_string('Eval mean loss: %f' % val_loss)
            log_string('Eval accuracy: %f' % val_acc)

            # Store validation metrics
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(f1)
            epochs.append(epoch)
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame({
                'epoch': epochs,
                'train_loss': train_losses,
                'train_accuracy': train_accs,
                'val_loss': val_losses,
                'val_accuracy': val_accs,
                'val_precision': val_precisions,
                'val_recall': val_recalls,
                'val_f1': val_f1s
            })
            
            # Save metrics as CSV
            metrics_df.to_csv(f"{metrics_dir}/training_metrics.csv", index=False)
            
            # Create plots for different metrics
            
            # 1. Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, '-o', label='Training Loss')
            plt.plot(epochs, val_losses, '-o', label='Validation Loss')
            plt.title(f'Loss vs. Epochs (Focal γ={args.focal_gamma})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_dir}/loss.png")
            plt.close()
            
            # 2. Accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_accs, '-o', label='Training Accuracy')
            plt.plot(epochs, val_accs, '-o', label='Validation Accuracy')
            plt.title('Accuracy vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_dir}/accuracy.png")
            plt.close()
            
            # 3. Binary metrics plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, val_precisions, '-o', label='Precision')
            plt.plot(epochs, val_recalls, '-o', label='Recall')
            plt.plot(epochs, val_f1s, '-o', label='F1 Score')
            plt.title(f'Binary Metrics for {vis_classes[POSITIVE_CLASS]} Class')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)
            plt.savefig(f"{plots_dir}/binary_metrics.png")
            plt.close()

            # Save class distributions to CSV
            train_class_dist = np.bincount(np.array(all_train_targets).astype(int))
            val_class_dist = class_dist
            val_pred_dist = pred_dist
            
            # Create distribution dataframe
            dist_data = {
                'class': list(vis_classes),
                'train_count': [train_class_dist[i] if i < len(train_class_dist) else 0 for i in range(NUM_CLASSES)],
                'train_percent': [100 * train_class_dist[i] / sum(train_class_dist) if i < len(train_class_dist) else 0 for i in range(NUM_CLASSES)],
                'val_count': [val_class_dist[i] if i < len(val_class_dist) else 0 for i in range(NUM_CLASSES)],
                'val_percent': [100 * val_class_dist[i] / sum(val_class_dist) if i < len(val_class_dist) else 0 for i in range(NUM_CLASSES)],
                'val_pred_count': [val_pred_dist[i] if i < len(val_pred_dist) else 0 for i in range(NUM_CLASSES)],
                'val_pred_percent': [100 * val_pred_dist[i] / sum(val_pred_dist) if i < len(val_pred_dist) else 0 for i in range(NUM_CLASSES)]
            }
            

            
            # Also save latest distribution as a separate file for easy access
            pd.DataFrame(dist_data).to_csv(f"{metrics_dir}/latest_class_distribution.csv", index=False)
            
            log_string(f"Class distribution saved to {metrics_dir}/class_distribution_epoch_{epoch}.csv")
            
            
            # Save model if we have better accuracy
            if val_acc >= best_accuracy:
                best_accuracy = val_acc
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best accuracy: %f' % best_accuracy)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)