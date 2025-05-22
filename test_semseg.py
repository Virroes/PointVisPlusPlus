"""
Test script for PointVisPlusPlus visibility classification model.
Loads a trained model and evaluates it on test data.
"""
import argparse
import os
import sys
import torch
import numpy as np
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils.VisDataLoader import VisDataSet
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj
    
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
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--test_dir', type=str, default='data/test/', help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    
    # VisDataSet parameters
    parser.add_argument('--k_neighbors', type=int, default=1024, help='Number of neighbors per point')
    parser.add_argument('--use_spherical', type=str2bool, default=True, help='Use spherical coordinates')
    parser.add_argument('--voxel_size', type=float, default=None, help='Voxel size for downsampling')
    parser.add_argument('--max_samples_per_scene', type=int, default=None, help='Max samples per scene')
    
    # Output parameters
    parser.add_argument('--out_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--visualize', type=str2bool, default=True, help='Generate visualization plots')
    
    return parser.parse_args()

def test(args):
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Create output directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Visibility classes (using the corrected ordering)
    vis_classes = ['occluded', 'visible']
    NUM_CLASSES = len(vis_classes)
    
    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    TEST_DATASET = VisDataSet(
        split='test',
        data_dir=args.test_dir,
        k_neighbors=args.k_neighbors,
        use_spherical=args.use_spherical,
        voxel_size=args.voxel_size,
        max_samples_per_scene=args.max_samples_per_scene
    )
    
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint_path}")
    try:
        # Try the safer approach first
        torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
        checkpoint = torch.load(args.checkpoint_path)
    except Exception as e:
        print(f"Warning: Couldn't load with weights_only=True. Trying legacy mode: {e}")
        # Fall back to legacy mode (less secure but more compatible)
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    print(f"Model was trained for {checkpoint['epoch']} epochs, class_avg_iou: {checkpoint.get('class_avg_iou', 'N/A')}")
    
    # Testing
    print("Running model inference...")
    with torch.no_grad():
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        
        # For detailed metrics
        all_preds = []
        all_targets = []
        
        for batch_id, (points, target) in enumerate(tqdm(testDataLoader)):
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)  # [batch, features, npoints]
            
            # Forward pass
            seg_pred, _ = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)  # [batch_size, num_points]
            
            # Copy to CPU for evaluation
            batch_label = target.cpu().data.numpy()
            
            # Compute metrics
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += points.size()[0] * points.size()[2]
            
            # Collect predictions and targets for detailed metrics
            all_preds.extend(pred_val.flatten())
            all_targets.extend(batch_label.flatten())
            
            # Per-class metrics
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate overall metrics
        accuracy = total_correct / float(total_seen)
        iou_per_class = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
        mIoU = np.mean(iou_per_class)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
        
        # Print results
        print("\n===== RESULTS =====")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Mean IoU: {mIoU:.4f}")
        
        print("\nPer-Class Performance:")
        for i, cls_name in enumerate(vis_classes):
            print(f"Class {i} ({cls_name}):")
            print(f"  - IoU: {iou_per_class[i]:.4f}")
            print(f"  - Precision: {precision[i]:.4f}")
            print(f"  - Recall: {recall[i]:.4f}")
            print(f"  - F1 Score: {f1[i]:.4f}")
            print(f"  - Samples: {total_seen_class[i]}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'mean_iou': float(mIoU),
            'iou_per_class': iou_per_class.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'samples_per_class': [int(x) for x in total_seen_class],  # Convert to regular Python ints
            'confusion_matrix': cm.tolist()
        }

        # Convert any remaining numpy types
        results = convert_numpy_types(results)

        # Save metrics to file
        import json
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualizations
        if args.visualize:
            # Confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=vis_classes, yticklabels=vis_classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(out_dir / 'confusion_matrix.png')
            
            # Precision and Recall plot
            plt.figure(figsize=(10, 6))
            x = np.arange(len(vis_classes))
            width = 0.35
            plt.bar(x - width/2, precision, width, label='Precision')
            plt.bar(x + width/2, recall, width, label='Recall')
            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title('Precision and Recall by Class')
            plt.xticks(x, vis_classes)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(out_dir / 'precision_recall.png')
            
            # IoU plot
            plt.figure(figsize=(10, 6))
            plt.bar(x, iou_per_class)
            plt.xlabel('Classes')
            plt.ylabel('IoU')
            plt.title('IoU by Class')
            plt.xticks(x, vis_classes)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(out_dir / 'iou.png')
            
        print(f"Results saved to {out_dir}")

if __name__ == '__main__':
    args = parse_args()
    test(args)