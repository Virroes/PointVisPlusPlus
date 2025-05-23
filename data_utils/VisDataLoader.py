import os
import math
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from torch.utils.data import Dataset
import gc
from scipy.spatial import cKDTree
import open3d as o3d


class VisDataSet(Dataset):
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "data/train/",
        k_neighbors: int = 64,  # Number of neighbors per point
        use_spherical: bool = True,
        voxel_size: float = None,  # None = no downsampling, float value = downsample
        precompute_neighbors: bool = True,
        max_samples_per_scene: int = None,  # None = use all points as centers
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.k_neighbors = k_neighbors
        print(f"Using {self.k_neighbors} neighbors")
        self.use_spherical = use_spherical
        self.voxel_size = voxel_size
        self.precompute_neighbors = precompute_neighbors
        self.max_samples_per_scene = max_samples_per_scene
        self.rng = np.random.default_rng(seed)
        
        # Memory-optimized containers
        self.all_scenes = []  # List of (points, labels, spherical_coords) tuples
        self.all_trees = []   # KD trees for each scene (for angular space)
        self.all_indices = []  # List of (scene_idx, point_idx) tuples for center points
        self.neighbor_cache = {}  # Cache for precomputed neighbors
        
        # Class distribution tracking
        self.label_counts = Counter()
        self.neighbor_label_counts = Counter()
        
        self._load_data()
        self._select_center_points()
        
        if self.precompute_neighbors:
            self._precompute_neighbors()
            
        # Print final class distribution summary
        self._analyze_class_distribution()
    
    def _voxel_downsample_o3d(self, points, labels):
        """Downsample point cloud using Open3D voxel grid method.
        
        Args:
            points: Numpy array of shape (N, 5) with [x, y, z, u, v]
            labels: Numpy array of shape (N,) with point labels
            
        Returns:
            downsampled_points: Numpy array of shape (M, 5) where M < N
            downsampled_labels: Numpy array of shape (M,) with corresponding labels
            original_indices: Mapping from downsampled to original points
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Store uv coordinates as colors (for retrieval later)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([
            points[:, 3:5],  # u, v coordinates
            np.zeros(len(points))  # Padding for third dimension
        ]))
        
        # Create label point cloud with same points but with label info
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        label_pcd.colors = o3d.utility.Vector3dVector(np.column_stack([
            labels.reshape(-1, 1),  # Labels as first component
            np.zeros((len(labels), 2))  # Padding for other components
        ]))
        
        # Perform voxel downsampling on both point clouds
        print(f"Voxel downsampling with voxel_size={self.voxel_size:.5f}")
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downsampled_label_pcd = label_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Extract downsampled points
        pts_down = np.asarray(downsampled_pcd.points)  # xyz
        uv_down = np.asarray(downsampled_pcd.colors)[:, :2]  # uv only
        
        # Combine into downsampled point array
        pts_downsampled = np.column_stack([pts_down, uv_down])
        
        # Extract labels - we use the first color channel where we stored the labels
        lbls_downsampled = np.round(np.asarray(downsampled_label_pcd.colors)[:, 0]).astype(int)
        
        # Debug - Log label distribution
        orig_counts = np.bincount(labels.astype(int))
        down_counts = np.bincount(lbls_downsampled.astype(int))
        
        print("Original label counts:", dict(enumerate(orig_counts)))
        print("Downsampled label counts:", dict(enumerate(down_counts)))
        
        # Calculate percentage change in proportion
        if len(orig_counts) == len(down_counts):
            orig_prop = orig_counts / np.sum(orig_counts)
            down_prop = down_counts / np.sum(down_counts)
            
            for i in range(len(orig_counts)):
                if orig_prop[i] > 0:
                    change = (down_prop[i] - orig_prop[i]) / orig_prop[i] * 100
                    print(f"Class {i} proportion change: {change:.1f}%")
        
        # Find mapping from downsampled to original points
        tree = o3d.geometry.KDTreeFlann(pcd)
        original_indices = []
        
        for i, point in enumerate(pts_down):
            # Find single nearest neighbor in original point cloud
            [_, idx, _] = tree.search_knn_vector_3d(point, 1)
            original_indices.append(idx[0])
        
        return pts_downsampled, lbls_downsampled, np.array(original_indices)
    
    def _load_data(self) -> None:
        """Load and preprocess all point cloud data."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        print(f"Loading point clouds from {self.data_dir}...")
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".xyz"):
                continue
                
            fpath = os.path.join(self.data_dir, fname)
            try:
                pts, lbls = self._read_xyz_file(fpath)
            except Exception as exc:
                print(f"✗  error processing {fpath}: {exc}")
                continue
                
            if pts.size == 0:
                continue  # empty file – skip
            
            print(f"Loading {fname}...")
            print(f"Original point count: {len(pts)}")
            
            # Log initial class distribution
            unique_labels, counts = np.unique(lbls, return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"Original class distribution: {label_dist}")
            for label, count in label_dist.items():
                self.label_counts[label] += count
            
            # Check if we should downsample
            if self.voxel_size is not None:
                # Voxel downsample BEFORE normalization
                pts_downsampled, lbls_downsampled, original_indices = self._voxel_downsample_o3d(pts, lbls)
                print(f"Downsampled point count: {len(pts_downsampled)} (reduction: {100 * (1 - len(pts_downsampled) / len(pts)):.1f}%)")
            else:
                # No downsampling, use original points
                print("Skipping downsampling (voxel_size=None)")
                pts_downsampled = pts
                lbls_downsampled = lbls
                original_indices = np.arange(len(pts))  # Identity mapping
            
            # NOW normalize the point cloud
            centroid = pts_downsampled[:, :3].mean(axis=0, keepdims=True)
            pts_downsampled[:, :3] -= centroid              # translate to origin
            scale = np.linalg.norm(pts_downsampled[:, :3], axis=1).max()
            if scale > 0:
                pts_downsampled[:, :3] /= scale             # scale to unit sphere
            
            # Convert to spherical coordinates
            spherical_coords = self._to_spherical(pts_downsampled)
            
            # Store scene data
            scene_idx = len(self.all_scenes)
            self.all_scenes.append((pts_downsampled, lbls_downsampled, spherical_coords))
            
            # Create KD tree in angular space (theta-phi)
            angular_coords = spherical_coords[:, 1:3]  # Just theta and phi
            tree = cKDTree(angular_coords)
            self.all_trees.append(tree)

        print(f"Loaded {len(self.all_scenes)} scenes with total {sum(len(scene[0]) for scene in self.all_scenes)} points")
        
        # Clear label counts since we'll build it from the downsampled points
        self.label_counts.clear()
    
    def _select_center_points(self):
        """Select points as potential centers, with optional sample limit."""
        if self.max_samples_per_scene is None:
            print("Using all points as potential centers...")
        else:
            print(f"Using up to {self.max_samples_per_scene} points per scene as centers...")
        
        # Clear existing indices
        self.all_indices = []
        
        # Track class distribution of center points
        center_label_counts = Counter()
        
        for scene_idx, (pts, lbls, _) in enumerate(self.all_scenes):
            # Count class distribution for this scene
            unique_labels, counts = np.unique(lbls, return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"Scene {scene_idx} class distribution: {label_dist}")
            
            # Update total label counts
            for label, count in label_dist.items():
                self.label_counts[label] += count
            
            total_points = len(pts)
            
            # Determine which points to use as centers
            if self.max_samples_per_scene is None or total_points <= self.max_samples_per_scene:
                # Use all points
                points_to_use = range(total_points)
                print(f"Scene {scene_idx}: Using all {total_points} points as centers")
            else:
                # Randomly sample points ensuring class balance
                points_to_use = self.rng.choice(
                    total_points, 
                    size=self.max_samples_per_scene, 
                    replace=False
                )
                print(f"Scene {scene_idx}: Randomly selected {len(points_to_use)} of {total_points} points as centers")
            
            # Add the selected points as centers
            for point_idx in points_to_use:
                self.all_indices.append((scene_idx, point_idx))
                center_label_counts[lbls[point_idx]] += 1
        
        print(f"Total center points across all scenes: {len(self.all_indices)}")
        print(f"Each sample will contain a center point and up to {self.k_neighbors-1} neighbors")
        print(f"Center points class distribution: {dict(center_label_counts)}")
        
    def _precompute_neighbors(self):
        """Precompute and cache neighbors for all center points."""
        print("Precomputing neighbors for center points...")
        
        # Reset neighbor label counter
        self.neighbor_label_counts = Counter()
        
        for idx, (scene_idx, point_idx) in enumerate(tqdm(self.all_indices)):
            scene_points, scene_labels, scene_spherical = self.all_scenes[scene_idx]
            
            # Query the angular KD tree and store results
            angular_coords = scene_spherical[point_idx, 1:3]  # theta and phi
            _, neighbor_indices = self.all_trees[scene_idx].query(
                angular_coords.reshape(1, -1),
                k=min(self.k_neighbors, len(scene_spherical))
            )
            neighbor_indices = neighbor_indices[0]  # Flatten
            
            # Cache the result
            self.neighbor_cache[(scene_idx, point_idx)] = neighbor_indices
            
            # Count neighbor labels
            for idx in neighbor_indices:
                self.neighbor_label_counts[scene_labels[idx]] += 1
        
        print(f"Precomputed neighbors for {len(self.neighbor_cache)} center points")
    
    def _analyze_class_distribution(self):
        """Analyze and print class distribution statistics."""
        print("\n" + "="*50)
        print(f"CLASS DISTRIBUTION ANALYSIS - {self.split.upper()} SET")
        print("="*50)
        
        # Overall class distribution in dataset
        total_points = sum(self.label_counts.values())
        print("\n1. OVERALL CLASS DISTRIBUTION:")
        for label, count in sorted(self.label_counts.items()):
            percentage = 100 * count / total_points if total_points > 0 else 0
            print(f"   Class {label}: {count} points ({percentage:.2f}%)")
        
        # Class distribution in neighborhoods
        total_neighbors = sum(self.neighbor_label_counts.values())
        print(f"\n2. CLASS DISTRIBUTION IN {self.split.upper()} NEIGHBORHOODS:")
        for label, count in sorted(self.neighbor_label_counts.items()):
            percentage = 100 * count / total_neighbors if total_neighbors > 0 else 0
            print(f"   Class {label}: {count} points ({percentage:.2f}%)")
            
            # Calculate ratio compared to original distribution
            original_percentage = 100 * self.label_counts[label] / total_points if total_points > 0 else 0
            if original_percentage > 0:
                ratio = percentage / original_percentage
                print(f"     Ratio to original: {ratio:.2f}x") 
        
        print("="*50 + "\n")

    def __len__(self) -> int:
        return len(self.all_indices)

    def __getitem__(self, idx: int):
        # Get scene and point index
        scene_idx, point_idx = self.all_indices[idx]
        scene_points, scene_labels, scene_spherical = self.all_scenes[scene_idx]
        
        # Use cached neighbors if available
        if self.precompute_neighbors and (scene_idx, point_idx) in self.neighbor_cache:
            neighbor_indices = self.neighbor_cache[(scene_idx, point_idx)]
        else:
            # Fallback to computing on-the-fly
            if len(scene_spherical) <= self.k_neighbors:
                neighbor_indices = np.arange(len(scene_spherical))
            else:
                angular_coords = scene_spherical[point_idx, 1:3]
                _, neighbor_indices = self.all_trees[scene_idx].query(
                    angular_coords.reshape(1, -1),
                    k=min(self.k_neighbors, len(scene_spherical))
                )
                neighbor_indices = neighbor_indices[0]
        
        # Extract neighbor points and labels
        neighbor_points = scene_points[neighbor_indices]
        neighbor_labels = scene_labels[neighbor_indices]
        
        # Use spherical coordinates if requested
        if self.use_spherical:
            return_points = np.column_stack([
                scene_spherical[neighbor_indices],  # r, theta, phi
                neighbor_points[:, 3:]              # u, v
            ])
        else:
            return_points = neighbor_points
            
        return return_points, neighbor_labels

    @staticmethod
    def _read_xyz_file(path: str):
        pts, lbls = [], []
        with open(path, "r") as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 6:
                    continue
                x, y, z, u, v, lab = map(float, vals)
                pts.append([x, y, z, u, v])
                lbls.append(int(lab))
        return np.asarray(pts, np.float32), np.asarray(lbls, np.int64)
    
    def _to_spherical(self, pts: np.ndarray) -> np.ndarray:
        """Convert points from Cartesian to spherical coordinates.
        
        Input shape: (N, 5) with [x, y, z, u, v]
        Output shape: (N, 3) with [r, theta, phi]
        """
        xyz = pts[:, :3]
        
        # Calculate spherical coordinates
        r = np.linalg.norm(xyz, axis=1)  # Radius
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])  # Azimuth angle (-π, π]
        phi = np.arcsin(xyz[:, 2] / (r + 1e-12))  # Elevation angle (-π/2, π/2]
        
        # Stack the spherical coordinates
        spherical = np.column_stack([r, theta, phi])
        
        return spherical
    
    def get_label_weights(self):
        """Calculate inverse class frequencies for loss weighting.
        
        Returns:
            weights: numpy array where weights[i] is inversely proportional to
                    the frequency of class i in the neighborhood samples
        """
        if not self.neighbor_label_counts:
            print("Warning: No neighborhood label counts available. Using uniform weights.")
            return np.ones(len(self.label_counts))
        
        # Get sorted class labels
        classes = sorted(self.neighbor_label_counts.keys())
        total = sum(self.neighbor_label_counts.values())
        
        # Inverse frequency weighting
        weights = np.zeros(len(classes))
        for i, label in enumerate(classes):
            count = self.neighbor_label_counts[label]
            weights[i] = total / (count + 1e-6)  # avoid division by zero
        
        # Normalize weights to sum to number of classes
        weights = weights / weights.sum() * len(classes)
        
        return weights
    
    def save_debug_samples(self, num_samples=10, output_dir="debug_samples"):
        """Save a subset of center points and their neighbors for debugging.
        
        Args:
            num_samples: Number of samples to save (default: 10)
            output_dir: Directory to save the debug samples
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random samples
        if len(self) <= num_samples:
            indices = list(range(len(self)))
        else:
            indices = self.rng.choice(len(self), size=num_samples, replace=False)
        
        print(f"Saving {len(indices)} debug samples to {output_dir}...")
        
        for i, idx in enumerate(indices):
            # Get points and labels using __getitem__
            points, labels = self[idx]
            scene_idx, point_idx = self.all_indices[idx]
            
            # If we're using spherical coordinates, convert back to Cartesian
            if self.use_spherical:
                # Extract r, theta, phi
                r = points[:, 0]
                theta = points[:, 1]
                phi = points[:, 2]
                
                # Convert to Cartesian
                x = r * np.cos(phi) * np.cos(theta)
                y = r * np.cos(phi) * np.sin(theta)
                z = r * np.sin(phi)
                
                # Get u, v coordinates
                u = points[:, 3]
                v = points[:, 4]
                
                save_data = np.column_stack([x, y, z, u, v, labels])
            else:
                # Already in Cartesian format
                save_data = np.column_stack([points, labels])
            
            # Save as XYZ file
            filename = f"sample_{i}_scene_{scene_idx}_center_{point_idx}.xyz"
            filepath = os.path.join(output_dir, filename)
            
            np.savetxt(
                filepath,
                save_data,
                fmt="%.6f %.6f %.6f %.6f %.6f %d",
                header=f"x y z u v label - Debug sample {i}, Scene {scene_idx}, Center point {point_idx}",
                comments=""
            )
            
            # Highlight center point in a separate file
            center_point = save_data[0].reshape(1, -1)
            center_filepath = os.path.join(output_dir, f"center_{i}_scene_{scene_idx}_point_{point_idx}.xyz")
            np.savetxt(
                center_filepath,
                center_point,
                fmt="%.6f %.6f %.6f %.6f %.6f %d",
                header=f"x y z u v label - Center point for sample {i}, Scene {scene_idx}",
                comments=""
            )
            
            print(f"Saved {filepath} with {len(points)} points")
        
        print(f"Successfully saved {len(indices)} debug samples to {output_dir}")