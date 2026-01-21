import numpy as np
from pathlib import Path
import struct
from typing import Optional, Tuple, List
import warnings

def ply_to_colmap_points3d(
    ply_path: str,
    output_path: Optional[str] = None,
    min_track_length: int = 2,
    max_error: float = 999999.0,
    binary_format: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    将 PLY 点云文件转换为 COLMAP points3D.txt 格式
    
    Args:
        ply_path: 输入的 PLY 文件路径
        output_path: 输出的 points3D.txt 文件路径，如果为 None 则只返回数据
        min_track_length: 最小轨迹长度（默认每个点至少被2个图像看到）
        max_error: 最大重投影误差（默认设为很大值）
        binary_format: 如果为 True，同时生成 points3D.bin 文件
        
    Returns:
        points: (N, 3) 点坐标数组
        colors: (N, 3) RGB 颜色数组 (0-255)
        track_lengths: 每个点的轨迹长度列表（如果 PLY 中没有，则使用默认值）
    
    Raises:
        FileNotFoundError: 如果 PLY 文件不存在
        ValueError: 如果 PLY 文件格式不支持
    """
    
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")
    
    # 读取 PLY 文件
    points, colors, extra_data = read_ply_file(ply_path)
    
    if points is None or len(points) == 0:
        raise ValueError(f"PLY 文件中没有点数据: {ply_path}")
    
    print(f"读取到 {len(points)} 个点")
    
    # 生成轨迹长度（如果 PLY 中没有轨迹信息，使用默认值）
    if 'track_length' in extra_data:
        track_lengths = extra_data['track_length']
    else:
        # 如果没有轨迹信息，为每个点生成合理的轨迹长度
        track_lengths = [min_track_length] * len(points)
        print(f"警告: PLY 中没有轨迹信息，使用默认轨迹长度: {min_track_length}")
    
    # 生成点 ID（从 1 开始）
    point_ids = np.arange(1, len(points) + 1)
    
    # 生成重投影误差（如果 PLY 中没有，使用默认值）
    if 'error' in extra_data:
        errors = extra_data['error']
    else:
        errors = np.ones(len(points)) * 0.1  # 默认误差
    
    # 如果指定了输出路径，写入文件
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文本格式
        write_points3d_txt(
            output_path.with_suffix('.txt'),
            point_ids,
            points,
            colors,
            track_lengths,
            errors
        )
        
        # 如果需要，同时写入二进制格式
        if binary_format:
            write_points3d_bin(
                output_path.with_suffix('.bin'),
                point_ids,
                points,
                colors,
                track_lengths,
                errors
            )
    
    return points, colors, track_lengths


def read_ply_file(ply_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    读取 PLY 文件，支持 ASCII 和二进制格式
    
    Returns:
        points: (N, 3) 点坐标
        colors: (N, 3) RGB 颜色
        extra_data: 额外数据字典
    """
    
    with open(ply_path, 'rb') as f:
        # 读取文件头
        header_lines = []
        line = f.readline().decode('ascii').strip()
        header_lines.append(line)
        
        if line != 'ply':
            raise ValueError(f"不是有效的 PLY 文件: {line}")
        
        format_ascii = True
        vertex_count = 0
        properties = []
        header_end = False
        
        while not header_end:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line.startswith('format'):
                if 'ascii' in line:
                    format_ascii = True
                elif 'binary' in line:
                    format_ascii = False
                else:
                    raise ValueError(f"不支持的 PLY 格式: {line}")
            
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
            
            elif line == 'end_header':
                header_end = True
        
        # 记录当前位置（数据开始位置）
        data_start = f.tell()
        
        print(f"PLY 格式: {'ASCII' if format_ascii else 'Binary'}")
        print(f"顶点数量: {vertex_count}")
        print(f"属性: {[p[0] for p in properties]}")
    
    # 根据格式读取数据
    if format_ascii:
        return read_ply_ascii(ply_path, vertex_count, properties, data_start)
    else:
        return read_ply_binary(ply_path, vertex_count, properties, data_start)


def read_ply_ascii(ply_path: Path, vertex_count: int, properties: List[Tuple[str, str]], data_start: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读取 ASCII 格式的 PLY 文件"""
    
    # 读取 ASCII 数据
    with open(ply_path, 'r') as f:
        # 跳转到数据开始位置
        f.seek(data_start)
        
        data = []
        for i in range(vertex_count):
            line = f.readline().strip()
            if not line:
                break
            values = line.split()
            data.append(values)
    
    if len(data) == 0:
        return None, None, {}
    
    # 解析数据
    points = []
    colors = []
    extra_data = {
        'track_length': [],
        'error': []
    }
    
    # 查找属性索引
    prop_names = [p[0] for p in properties]
    
    # 标准属性映射
    x_idx = prop_names.index('x') if 'x' in prop_names else -1
    y_idx = prop_names.index('y') if 'y' in prop_names else -1
    z_idx = prop_names.index('z') if 'z' in prop_names else -1
    
    # 颜色属性可能有不同的名称
    color_indices = []
    for color_name in ['red', 'green', 'blue', 'r', 'g', 'b', 'diffuse_red', 'diffuse_green', 'diffuse_blue']:
        if color_name in prop_names:
            color_indices.append(prop_names.index(color_name))
    
    # 如果找不到标准的颜色属性，尝试其他可能的位置
    if len(color_indices) == 0:
        # 假设颜色在位置 3-5 或 6-8（如果有法向量）
        if len(prop_names) >= 6:
            color_indices = [3, 4, 5]  # 假设在 xyz 之后
    
    # 轨迹长度和误差
    track_idx = prop_names.index('track_length') if 'track_length' in prop_names else -1
    error_idx = prop_names.index('error') if 'error' in prop_names else -1
    
    for row in data:
        # 提取坐标
        if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
            point = [float(row[x_idx]), float(row[y_idx]), float(row[z_idx])]
            points.append(point)
        else:
            # 如果没有明确的 xyz，假设前三个属性是坐标
            point = [float(row[0]), float(row[1]), float(row[2])]
            points.append(point)
        
        # 提取颜色
        if len(color_indices) >= 3:
            # 确保索引不越界
            if max(color_indices) < len(row):
                color = [
                    min(255, max(0, int(float(row[color_indices[0]])))),
                    min(255, max(0, int(float(row[color_indices[1]])))),
                    min(255, max(0, int(float(row[color_indices[2]]))))
                ]
                colors.append(color)
            else:
                colors.append([255, 255, 255])  # 默认白色
        else:
            colors.append([255, 255, 255])  # 默认白色
        
        # 提取轨迹长度
        if track_idx >= 0 and track_idx < len(row):
            extra_data['track_length'].append(int(float(row[track_idx])))
        else:
            extra_data['track_length'].append(2)  # 默认值
        
        # 提取误差
        if error_idx >= 0 and error_idx < len(row):
            extra_data['error'].append(float(row[error_idx]))
        else:
            extra_data['error'].append(0.1)  # 默认值
    
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)
    
    return points, colors, extra_data


def read_ply_binary(ply_path: Path, vertex_count: int, properties: List[Tuple[str, str]], data_start: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读取二进制格式的 PLY 文件"""
    
    # 创建数据类型映射
    type_map = {
        'float': 'f4', 'float32': 'f4',
        'double': 'f8', 'float64': 'f8',
        'int': 'i4', 'int32': 'i4',
        'uint': 'u4', 'uint32': 'u4',
        'uchar': 'u1', 'uint8': 'u1',
        'ushort': 'u2', 'uint16': 'u2',
    }
    
    # 构建 dtype
    dtype_list = []
    for prop_name, prop_type in properties:
        numpy_type = type_map.get(prop_type, 'f4')  # 默认 float32
        dtype_list.append((prop_name, numpy_type))
    
    dtype = np.dtype(dtype_list)
    
    # 读取二进制数据
    with open(ply_path, 'rb') as f:
        f.seek(data_start)  # 跳转到数据开始位置
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    
    if len(data) == 0:
        return None, None, {}
    
    # 提取数据
    points = []
    colors = []
    extra_data = {
        'track_length': [],
        'error': []
    }
    
    # 查找属性
    prop_names = [p[0] for p in properties]
    
    # 提取坐标
    if 'x' in data.dtype.names and 'y' in data.dtype.names and 'z' in data.dtype.names:
        points = np.column_stack([data['x'], data['y'], data['z']])
    else:
        # 假设前三个属性是坐标
        first_three_names = [name for name in data.dtype.names[:3]]
        if len(first_three_names) >= 3:
            points = np.column_stack([data[name] for name in first_three_names])
        else:
            raise ValueError("无法找到坐标属性")
    
    # 提取颜色
    color_found = False
    for color_set in [['red', 'green', 'blue'], ['r', 'g', 'b'], ['diffuse_red', 'diffuse_green', 'diffuse_blue']]:
        if all(color in data.dtype.names for color in color_set):
            colors = np.column_stack([
                np.clip(data[color_set[0]], 0, 255).astype(np.uint8),
                np.clip(data[color_set[1]], 0, 255).astype(np.uint8),
                np.clip(data[color_set[2]], 0, 255).astype(np.uint8)
            ])
            color_found = True
            break
    
    if not color_found:
        # 如果没有找到颜色，创建白色
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
        print("警告: 未找到颜色属性，使用默认白色")
    
    # 提取轨迹长度
    if 'track_length' in data.dtype.names:
        extra_data['track_length'] = data['track_length'].astype(int).tolist()
    else:
        extra_data['track_length'] = [2] * len(points)
    
    # 提取误差
    if 'error' in data.dtype.names:
        extra_data['error'] = data['error'].tolist()
    else:
        extra_data['error'] = [0.1] * len(points)
    
    return points, colors, extra_data


def write_points3d_txt(
    output_path: Path,
    point_ids: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    track_lengths: List[int],
    errors: np.ndarray
):
    """
    写入 COLMAP points3D.txt 文本格式
    
    格式:
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}, mean track length: {:.2f}\n".format(
            len(points), np.mean(track_lengths) if track_lengths else 0.0
        ))
        
        # 写入每个点
        for i in range(len(points)):
            point_id = point_ids[i]
            x, y, z = points[i]
            r, g, b = colors[i]
            error = errors[i] if i < len(errors) else 0.1
            track_len = track_lengths[i] if i < len(track_lengths) else 2
            
            # 写入基本信息
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f}")
            
            # 写入轨迹信息（这里生成虚拟的轨迹，因为 PLY 中通常没有）
            # 在实际应用中，你可能需要从其他地方获取轨迹信息
            if track_len > 0:
                # 生成虚拟轨迹（在实际应用中应该使用真实数据）
                for j in range(track_len):
                    # 生成虚拟的图像 ID 和点索引
                    # 注意：这里只是示例，实际应该使用真实的轨迹数据
                    image_id = 1000 + (i * track_len + j) % 100  # 虚拟图像 ID
                    point2d_idx = (i + j) % 1000  # 虚拟点索引
                    f.write(f" {image_id} {point2d_idx}")
            
            f.write("\n")
    
    print(f"✅ 已写入文本格式: {output_path}")
    print(f"   点数: {len(points)}")


def write_points3d_bin(
    output_path: Path,
    point_ids: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    track_lengths: List[int],
    errors: np.ndarray
):
    """
    写入 COLMAP points3D.bin 二进制格式
    
    二进制格式:
    - 8 字节: 点数 (uint64)
    - 对每个点:
      - 8 字节: 点 ID (uint64)
      - 24 字节: XYZ 坐标 (3 × double)
      - 3 字节: RGB 颜色 (3 × uint8)
      - 8 字节: 误差 (double)
      - 8 字节: 轨迹长度 (uint64)
      - 对每个轨迹元素:
        - 8 字节: 图像 ID (uint64)
        - 8 字节: 点2D索引 (uint64)
    """
    
    with open(output_path, 'wb') as f:
        # 写入点数
        num_points = len(points)
        f.write(struct.pack('Q', num_points))
        
        for i in range(num_points):
            point_id = point_ids[i]
            x, y, z = points[i]
            r, g, b = colors[i]
            error = errors[i] if i < len(errors) else 0.1
            track_len = track_lengths[i] if i < len(track_lengths) else 2
            
            # 写入点基本信息
            f.write(struct.pack('Q', point_id))  # 点 ID
            f.write(struct.pack('ddd', x, y, z))  # XYZ 坐标
            f.write(struct.pack('BBB', r, g, b))  # RGB 颜色
            f.write(struct.pack('d', error))      # 误差
            
            # 写入轨迹长度
            f.write(struct.pack('Q', track_len))
            
            # 写入轨迹信息（虚拟数据）
            for j in range(track_len):
                # 生成虚拟的图像 ID 和点索引
                image_id = 1000 + (i * track_len + j) % 100
                point2d_idx = (i + j) % 1000
                f.write(struct.pack('QQ', image_id, point2d_idx))
    
    print(f"✅ 已写入二进制格式: {output_path}")
    print(f"   点数: {len(points)}")


def create_dummy_tracks(
    points: np.ndarray,
    images_count: int = 100,
    min_track_length: int = 2,
    max_track_length: int = 10
) -> List[List[Tuple[int, int]]]:
    """
    为点云创建虚拟轨迹（用于演示）
    
    在实际应用中，应该使用真实的轨迹数据
    """
    np.random.seed(42)  # 固定随机种子以便重现
    
    tracks = []
    for i in range(len(points)):
        # 随机确定轨迹长度
        track_len = np.random.randint(min_track_length, max_track_length + 1)
        
        track = []
        # 随机选择图像
        image_ids = np.random.choice(range(1, images_count + 1), size=track_len, replace=False)
        
        for img_id in image_ids:
            # 随机生成点2D索引
            point2d_idx = np.random.randint(0, 10000)
            track.append((int(img_id), int(point2d_idx)))
        
        tracks.append(track)
    
    return tracks


def visualize_points(points: np.ndarray, colors: np.ndarray = None):
    """可视化点云（需要安装 open3d）"""
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            # 确保颜色在 0-1 范围内
            colors_normalized = colors.astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # 创建坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd, coord_frame])
        
    except ImportError:
        print("未安装 Open3D，跳过可视化")
        print("安装: pip install open3d")
    except Exception as e:
        print(f"可视化时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 示例 1: 基本使用
    print("示例 1: 基本转换")
    try:
        # 替换为你的 PLY 文件路径
        ply_file = "data/hall_old/sparse/0/points3D.ply"
        
        # 转换并保存为文本格式
        points, colors, track_lengths = ply_to_colmap_points3d(
            ply_path=ply_file,
            output_path="data/hall_old/sparse/0/points3D.txt",
            min_track_length=2
        )
        
        print(f"转换完成!")
        print(f"点数: {len(points)}")
        print(f"颜色范围: {colors.min()} - {colors.max()}")
        print(f"轨迹长度范围: {min(track_lengths)} - {max(track_lengths)}")
        
        # 可视化（可选）
        # visualize_points(points, colors)
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("正在创建示例 PLY 文件...")
        
        # 创建示例 PLY 文件
        create_sample_ply("points3d.ply", num_points=1000)
        print("已创建示例文件，请重新运行")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def create_sample_ply(filename: str, num_points: int = 1000):
    """创建示例 PLY 文件用于测试"""
    import random
    
    with open(filename, 'w') as f:
        # 写入 PLY 头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uint track_length\n")
        f.write("property float error\n")
        f.write("end_header\n")
        
        # 写入数据
        for i in range(num_points):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(-5, 5)
            
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            
            track_len = random.randint(2, 10)
            error = random.uniform(0.01, 1.0)
            
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {track_len} {error:.6f}\n")
    
    print(f"✅ 已创建示例 PLY 文件: {filename}")