import os
import json
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
import pdb

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

def _elem_to_array(elem, dtype=float, expected_len=None):
    """
    把单个元素转换为 1D numpy 数组（dtype）。
    - 支持 map/generator/list/tuple/ndarray
    - 支持 '1 2 3' 字符串或 b'...' bytes
    - 支持单个数值
    expected_len 可选，用于提高效率（用于 np.fromiter 的 count）
    """
    # already ndarray
    if isinstance(elem, np.ndarray):
        return elem.astype(dtype)
    # list/tuple
    if isinstance(elem, (list, tuple)):
        return np.asarray(elem, dtype=dtype)
    # string or bytes: 尝试用 fromstring 分割
    if isinstance(elem, (str, bytes)):
        s = elem.decode() if isinstance(elem, bytes) else elem
        arr = np.fromstring(s.strip(), sep=' ', dtype=dtype)
        if arr.size == 0:
            # 可能是单个数的字符串
            try:
                return np.array([float(s)], dtype=dtype)
            except Exception:
                raise ValueError(f"无法解析字符串元素为数值: {repr(s)}")
        return arr.astype(dtype)
    # map/generator/其它可迭代（包括 map 对象）
    try:
        # 尝试用 fromiter（若 expected_len 已知效率更高）
        if expected_len is not None:
            arr = np.fromiter(elem, dtype=dtype, count=expected_len)
            # 如果 fromiter 返回空，可能是 elem 已被消费过，退回到 list() 方案
            if arr.size == 0:
                vals = list(elem)
                return np.asarray(vals, dtype=dtype)
            return arr
        else:
            # expected_len 不知时，用 list() 保守处理（会消费迭代器）
            vals = list(elem)
            return np.asarray(vals, dtype=dtype)
    except TypeError:
        # 不是可迭代，尝试当作单个数值
        try:
            return np.array([float(elem)], dtype=dtype)
        except Exception:
            raise ValueError(f"无法把元素转换为数值数组: {repr(elem)}")

def stack_object_array(obj_arr, expected_len=None, dtype=np.float32, allow_broadcast=False):
    """
    将 object dtype 的数组（每个元素是可迭代的数值序列或字符串等）转换为一个 2D/1D numpy 数组。
    - 如果 expected_len 给出，最终返回形状 (N, expected_len)
    - 否则会尝试推断每个元素的长度并检查一致性，然后 vstack
    - allow_broadcast: 若 expected_len 给出但某些元素长度为1，可广播到 expected_len（例如 color 给出单个灰度）
    """
    if not isinstance(obj_arr, np.ndarray):
        obj_arr = np.asarray(obj_arr, dtype=object)

    # 如果已经是数值数组，可以直接 astype
    if np.issubdtype(obj_arr.dtype, np.number):
        arr = obj_arr.astype(dtype)
        # 如果一维且 expected_len 给出，尝试 reshape
        if expected_len is not None and arr.ndim == 1:
            if arr.size == expected_len * obj_arr.shape[0]:
                return arr.reshape((-1, expected_len))
        return arr

    N = obj_arr.shape[0]
    parsed = []
    lengths = []
    for i, el in enumerate(obj_arr):
        a = _elem_to_array(el, dtype=dtype, expected_len=expected_len)
        parsed.append(a)
        lengths.append(a.size)

    if expected_len is not None:
        # 检查长度兼容性
        out = np.empty((N, expected_len), dtype=dtype)
        for i, a in enumerate(parsed):
            if a.size == expected_len:
                out[i, :] = a
            elif a.size == 1 and allow_broadcast:
                out[i, :] = a[0]
            else:
                raise ValueError(f"元素 {i} 长度 {a.size} 无法匹配 expected_len={expected_len}")
        return out
    else:
        # expected_len 未给出，所有长度必须相同 -> vstack
        uniq = set(lengths)
        if len(uniq) != 1:
            raise ValueError(f"元素长度不一致，无法 vstack: 长度集合={uniq}")
        L = lengths[0]
        if L == 1:
            return np.asarray([a.item() for a in parsed], dtype=dtype)
        return np.vstack(parsed).astype(dtype)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            
            # pdb.set_trace()
            # tvec = np.array(list(im.tvec.item()), dtype=float)
            # trans = tvec.reshape(3, 1)
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        # points = manager.points3D.astype(np.float32)
        # points_err = manager.point3D_errors.astype(np.float32)
        # points_rgb = manager.point3D_colors.astype(np.uint8)
        # points: 期望 (N,3) float32
        points = stack_object_array(manager.points3D, expected_len=3, dtype=np.float32)
        # points_err: 可能是标量数组 -> 得到 (N,) float32
        # 如果管理器里每个元素是长度为1的数组或单值，allow_broadcast=True 可接受
        points_err = stack_object_array(manager.point3D_errors, expected_len=1, dtype=np.float32).reshape(-1)
        # points_rgb: 期望 (N,3) uint8；允许输入灰度单值（广播）
        points_rgb = stack_object_array(manager.point3D_colors, expected_len=3, dtype=np.uint8, allow_broadcast=True)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        # for point_id, data in manager.point3D_id_to_images.items():
        #     for image_id, _ in data:
        #         image_name = image_id_to_name[image_id]
        #         point_idx = manager.point3D_id_to_point3D_idx[point_id]
        #         point_indices.setdefault(image_name, []).append(point_idx)
        # point_indices = {
        #     k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        # }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        self.scene_scale_nerf2mesh = 1.0 / np.min(dists)
        print(f"[Parser] Scene scale: {self.scene_scale:.2f}")
        print(f"[Parser] Scene scale (nerf2mesh): {self.scene_scale_nerf2mesh:.2f}")


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        depth_mode: str = "pts",
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.depth_mode = depth_mode
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
        self.num_depth_sample = 10000

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            if self.depth_mode=="pts":
                # projected points to image plane to get depths
                worldtocams = np.linalg.inv(camtoworlds)
                image_name = self.parser.image_names[index]
                point_indices = self.parser.point_indices[image_name]
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)
                # filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()
            elif self.depth_mode=="npz":
                depth_npz_path = self.parser.image_paths[index].replace("images","npz_depths")
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
                file_ext = os.path.splitext(depth_npz_path)[1].lower()
                if file_ext in image_extensions:
                    depth_npz_path = os.path.splitext(depth_npz_path)[0] + '.npz'
                else:
                    print("unknown ext for img")
                data_np = np.load(depth_npz_path)
                depth_np = data_np['arr_0']
                # h,w = depth_np.shape
                # x_coords = np.arange(w)  # [0, 1, 2, ..., w-1]
                # y_coords = np.arange(h)  # [0, 1, 2, ..., h-1]
                # xx, yy = np.meshgrid(x_coords, y_coords)
                # pixel_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
                
                #depth_np = depth_np.reshape(-1)
                #data["points"] = torch.from_numpy(pixel_coords).float()
                data["depths"] = torch.from_numpy(depth_np).float()

            else:
                print("unknown depth_mode when loading depth gt for dataseet")
            

        return data

class BlenderDataset:
    """ A simple synthetic Blender dataset class. """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        bg_color: Tuple[float, float, float] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.bg_color = None if bg_color is None else np.array(bg_color)
        self.image_size = 800
        
        # Loads json file that defines camtoworlds and intrinrics
        json_path = os.path.join(self.data_dir, f"transforms_{self.split}.json")
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        # Compute camera intrinsics
        self.camera_angle = json_data["camera_angle_x"] * 0.5
        c = self.image_size // 2 # pricipal point in pixels
        f = c / np.tan(self.camera_angle)
        self.K = np.array([
            [f, 0, c],
            [0, f, c],
            [0, 0, 1]
        ], dtype=np.float32)

        # Load images and camera extrinsics
        self.image_ids = []
        self.images = []
        self.camtoworlds = []
        self.alphas = []
        for frame_data in json_data["frames"]:
            image_id = frame_data["file_path"].split("/")[-1]
            image_file_path = os.path.join(self.data_dir, self.split, f"{image_id}.png")
            rgba = imageio.imread(image_file_path)
            image = self.add_bg_color(rgba)
            camtoworld = np.array(frame_data["transform_matrix"])

            # Adjust OpenGL v.s. COLMAP coordinate convension (Y-up, Z-back to Y-down, Z-forward)
            camtoworld[:3, 1:3] *= -1

            self.image_ids.append(image_id)
            self.images.append(image)
            self.camtoworlds.append(camtoworld)
            self.alphas.append(rgba[..., 3] / 255.0)
        self.camtoworlds = np.array(self.camtoworlds, dtype=np.float32)
                
    def add_bg_color(self, rgba):
        if self.bg_color is None:
            return rgba[..., :3]
        rgb = rgba[..., :3] # [0, 255]
        alpha = rgba[..., 3:4] / 255.0
        image = rgb * alpha + self.bg_color * (1 - alpha)
        return image
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index:int) -> Dict[str, Any]:
        data = {
            "K": torch.tensor(self.K, dtype=torch.float).float(),
            "camtoworld": torch.tensor(self.camtoworlds[index], dtype=torch.float).float(),
            "image": torch.tensor(self.images[index], dtype=torch.float).float(),
            "alpha": torch.tensor(self.alphas[index], dtype=torch.float).float(),
            "image_id": index,  # the index of the image in the dataset
        }
        return data
    
if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()