import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import rawpy  # Required for Nikon RAW support

class ImageProcessor:
    """Base class for image processing operations"""
    
    SUPPORTED_RAW_EXTENSIONS = ('.nef',)  # Nikon RAW extensions
    DEFAULT_IMG_PREFIX = 'DSC_'
    DEFAULT_SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp') + SUPPORTED_RAW_EXTENSIONS
    
    @classmethod
    def get_image_files(cls, input_dir, img_prefix=None, img_ext=None):
        """Get sorted image files from directory with support for RAW files"""
        img_prefix = img_prefix or cls.DEFAULT_IMG_PREFIX
        image_files = []
        
        for f in os.listdir(input_dir):
            # Skip directories
            if os.path.isdir(os.path.join(input_dir, f)):
                continue
                
            # Check prefix
            if not f.startswith(img_prefix):
                continue
                
            f_lower = f.lower()
            
            # Check extension
            if img_ext:
                # Use specified extension
                if not f_lower.endswith(img_ext.lower()):
                    continue
            else:
                # Use default supported extensions
                if not f_lower.endswith(cls.DEFAULT_SUPPORTED_EXTENSIONS):
                    continue
            
            image_files.append(f)
        
        return sorted(image_files)
    
    @classmethod
    def open_image(cls, image_path):
        """Open image with support for RAW files"""
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext in cls.SUPPORTED_RAW_EXTENSIONS:
            # Process Nikon RAW file
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8
                )
                return Image.fromarray(rgb)
        else:
            # Process standard image file
            return Image.open(image_path)
    
    @classmethod
    def select_images(cls, image_files, start=0, end=0, step=1):
        """Select images with given step interval, starting from the specified element"""
        if step < 1:
            raise ValueError("Step must be positive integer")
        if not (0 <= start < len(image_files)):
            raise ValueError("Start index out of range")
        if end<=0 or end >= len(image_files):
            return image_files[start::step]
        return image_files[start:end:step]
    @classmethod
    def select_imagesx(cls, image_files, input_dir=None, start=0, end=None, step=1, 
                     brightness_threshold=None, min_brightness_diff=10.0):
        """
        选择图像序列，支持亮度差异过滤
        
        参数:
            image_files: 图像文件列表
            input_dir: 图像所在目录 (亮度计算需要)
            start: 起始索引 (默认0)
            end: 结束索引 (None表示到末尾)
            step: 选择步长 (默认1)
            brightness_threshold: 亮度阈值 (None表示不过滤)
            min_brightness_diff: 最小亮度差异 (默认10.0)
        
        返回:
            选择后的图像文件列表
        """
        # 验证参数
        if step < 1:
            raise ValueError("Step must be positive integer")
        if start < 0 or start >= len(image_files):
            raise ValueError(f"Start index {start} out of range [0, {len(image_files)-1}]")
        
        # 确定结束索引
        if end is None or end <= 0 or end > len(image_files):
            end = len(image_files)
        
        # 获取子集
        subset = image_files[start:end]
        
        # 使用亮度差异过滤
        if brightness_threshold is not None and input_dir:
            return cls._select_by_brightness(subset, input_dir, min_brightness_diff, brightness_threshold)
        
        # 使用步长选择
        return subset[::step]
    
    @classmethod
    def _select_by_brightness(cls, image_files, input_dir, min_diff, brightness_threshold):
        """
        基于亮度差异选择图像
        
        参数:
            image_files: 图像文件列表
            input_dir: 图像所在目录
            min_diff: 最小亮度差异
            brightness_threshold: 亮度阈值
        
        返回:
            选择后的图像文件列表
        """
        selected_files = []
        last_brightness = None
        
        # 使用tqdm显示进度条
        for idx, filename in enumerate(tqdm(image_files, desc="亮度分析")):
            img_path = os.path.join(input_dir, filename)
            
            try:
                # 计算当前图像亮度
                brightness = cls._calculate_brightness(img_path)
                
                # 检查是否满足亮度阈值
                if brightness_threshold is not None and brightness < brightness_threshold:
                    continue
                    
                # 首次添加或亮度差异足够大
                if last_brightness is None or abs(brightness - last_brightness) >= min_diff:
                    selected_files.append(filename)
                    last_brightness = brightness
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
        
        print(f"亮度过滤: 从 {len(image_files)} 张中选出 {len(selected_files)} 张图像")
        return selected_files
    
    @classmethod
    def _calculate_brightness(cls, image_path, resize_factor=0.2):
        """
        计算图像平均亮度 (高效方法)
        
        参数:
            image_path: 图像路径
            resize_factor: 缩放因子 (加速处理)
        
        返回:
            平均亮度值 (0-255)
        """
        # 打开图像并转换为灰度
        with cls.open_image(image_path) as img:
            # 转换为灰度图
            gray_img = img.convert("L")
            
            # 缩小图像以加快处理速度
            if resize_factor < 1.0:
                new_size = (int(gray_img.width * resize_factor), 
                            int(gray_img.height * resize_factor))
                gray_img = gray_img.resize(new_size, Image.LANCZOS)
            
            # 转换为numpy数组计算平均值
            img_array = np.array(gray_img)
            return np.mean(img_array)


class VideoCreator(ImageProcessor):
    """Create video from image sequence"""
    
    @classmethod
    def create_video(cls, input_dir, output_filename, start=0, end=0, step=1, fps=30, duration=None,
                    img_prefix=None, img_ext=None, loop=False, overwrite=False):
        """
        Create video from images with progress bar and RAW support
        """
        # Check output file
        if os.path.exists(output_filename) and not overwrite:
            raise FileExistsError(f"Output file exists: {output_filename} (use --overwrite)")
        
        # Get and select images
        image_files = cls.get_image_files(input_dir, img_prefix, img_ext)
        if not image_files:
            raise FileNotFoundError(f"No images found with prefix '{img_prefix}' and ext '{img_ext}'")
        
        selected_files = cls.select_imagesx(image_files, input_dir, start, end, step)
        if not selected_files:
            raise ValueError(f"Step value ({step}) too large, no images selected")
        
        print(f"Found {len(image_files)} images, selected {len(selected_files)} with step {step}")
        
        # Get dimensions from first image
        first_img_path = os.path.join(input_dir, selected_files[0])
        with cls.open_image(first_img_path) as first_img:
            width, height = first_img.size
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_filename}")
        
        # Calculate frame count
        if duration is not None:
            total_frames = int(fps * duration)
            if not loop and total_frames > len(selected_files):
                print(f"Warning: Required frames ({total_frames}) exceed available images ({len(selected_files)})")
        else:
            total_frames = len(selected_files)
        
        try:
            # Generate frames with progress bar
            for frame_idx in tqdm(range(total_frames), desc="Generating video"):
                img_idx = frame_idx % len(selected_files) if loop or duration is None else min(frame_idx, len(selected_files) - 1)
                
                img_path = os.path.join(input_dir, selected_files[img_idx])
                with cls.open_image(img_path) as img:
                    # Convert PIL to OpenCV format
                    frame_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_cv)
            
            print(f"Video created: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"Error during video creation: {e}")
            raise
        finally:
            video_writer.release()


class VerticalCompositor(ImageProcessor):
    """Create vertical composite image from multiple images"""
    
    @classmethod
    def create_composite(cls, input_dir, output_filename, start=0, end=0, step=1, min_diff=0,
                        img_prefix=None, img_ext=None, overwrite=False):
        """Create vertical composite image"""
        # Check output file
        if os.path.exists(output_filename) and not overwrite:
            raise FileExistsError(f"Output file exists: {output_filename} (use --overwrite)")
        
        # Get and select images
        image_files = cls.get_image_files(input_dir, img_prefix, img_ext)
        if not image_files:
            raise FileNotFoundError(f"No images found with prefix '{img_prefix}' and ext '{img_ext}'")
        
        selected_files = cls.select_imagesx(image_files, input_dir, start, end, step, 0, min_diff)
        if not selected_files:
            raise ValueError(f"Step value ({step}) too large, no images selected")
        
        print(f"Found {len(image_files)} images, selected {len(selected_files)} with step {step}")
        
        # Get dimensions from first image
        first_img_path = os.path.join(input_dir, selected_files[0])
        with cls.open_image(first_img_path) as first_img:
            width, height = first_img.size
        
        # Calculate column distribution
        cols_per_image = width // len(selected_files)
        remaining_cols = width % len(selected_files)
        
        # Create new image
        new_image = Image.new('RGB', (width, height))
        current_col = 0
        
        try:
            for img_idx in tqdm(range(len(selected_files)), desc="Generating image"):
                image_file = selected_files[img_idx]
                cols = cols_per_image + (1 if img_idx < remaining_cols else 0)
                img_path = os.path.join(input_dir, image_file)
                
                with cls.open_image(img_path) as img:
                    # Extract vertical strip
                    region = img.crop((current_col, 0, current_col + cols, height))
                    new_image.paste(region, (current_col, 0))
                
                current_col += cols
            
            new_image.save(output_filename)
            print(f"Composite created: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"Error during composite creation: {e}")
            raise


def parse_arguments():
    """Parse command line arguments with common options"""
    parser = argparse.ArgumentParser(description='Image Processing Tool: Video Creator & Vertical Compositor')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Common arguments for both commands
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('input_dir', help='Input directory containing images')
    common_args.add_argument('output_filename', help='Output filename')
    common_args.add_argument('-S', '--start', type=int, default=1,
                           help='Set first image (default: 1)')
    common_args.add_argument('-E', '--end', type=int, default=0,
                           help='Set last image (default: 0,no limitation)')
    common_args.add_argument('-N', '--step', type=int, default=1,
                           help='Select every Nth image (default: 1)')
    common_args.add_argument('--prefix', default=ImageProcessor.DEFAULT_IMG_PREFIX,
                           help=f'Image filename prefix (default: {ImageProcessor.DEFAULT_IMG_PREFIX})')
    common_args.add_argument('--ext', 
                           help='Specific image extension to look for (default: any supported format)')
    common_args.add_argument('--overwrite', action='store_true',
                           help='Overwrite existing output file')
    
    # Video command
    video_parser = subparsers.add_parser('video', parents=[common_args], 
                                        help='Create video from image sequence')
    video_parser.add_argument('--fps', type=int, default=30,
                            help='Output video frames per second (default: 30)')
    video_parser.add_argument('--duration', type=float,
                            help='Output video duration in seconds (default: based on image count)')
    video_parser.add_argument('--loop', action='store_true',
                            help='Loop images to fill specified duration')
    
    # Vertical command
    vertical_parser = subparsers.add_parser('vertical', parents=[common_args],
                                          help='Create vertical composite image')
    vertical_parser.add_argument('--brightness', type=float, default=None,
                       help='Minimum brightness threshold (0-255)')
    vertical_parser.add_argument('--min-diff', type=float, default=10.0,
                       help='Minimum brightness difference between consecutive images')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    try:
        if args.command == 'video':
            VideoCreator.create_video(
                input_dir=args.input_dir,
                output_filename=args.output_filename,
                start=args.start - 1,
                end=args.end,
                step=args.step,
                fps=args.fps,
                duration=args.duration,
                img_prefix=args.prefix,
                img_ext=args.ext,
                loop=args.loop,
                overwrite=args.overwrite
            )
        elif args.command == 'vertical':
            VerticalCompositor.create_composite(
                input_dir=args.input_dir,
                output_filename=args.output_filename,
                start=args.start - 1,
                end=args.end,
                step=args.step,
                min_diff=args.min_diff,
                img_prefix=args.prefix,
                img_ext=args.ext,
                overwrite=args.overwrite
            )
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Operation failed. See error details above.")


if __name__ == "__main__":
    main()
