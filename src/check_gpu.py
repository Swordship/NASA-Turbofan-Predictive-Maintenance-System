"""
Check GPU availability and setup TensorFlow to use GPU
"""

import tensorflow as tf
import sys

print("\n" + "=" * 80)
print("🖥️ GPU SETUP CHECK")
print("=" * 80)

# Check TensorFlow version
print(f"\n📦 TensorFlow Version: {tf.__version__}")

# List all physical devices
print("\n🔍 Available Devices:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"   - {device.device_type}: {device.name}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🎮 GPU Devices Found: {len(gpus)}")

if len(gpus) > 0:
    print("✅ GPU is available!")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        
    # Get GPU details
    try:
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices()
        for device in local_devices:
            if device.device_type == 'GPU':
                print(f"\n   GPU Details:")
                print(f"   Name: {device.physical_device_desc}")
                print(f"   Memory: {device.memory_limit / 1024**3:.2f} GB")
    except:
        pass
        
    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
    print("\n⚙️ Configuring GPU memory growth...")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("   ✅ Memory growth enabled")
    except RuntimeError as e:
        print(f"   ⚠️ {e}")
        
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("   ✅ GPU computation successful!")
        print(f"   Result:\n{c.numpy()}")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        
else:
    print("❌ No GPU found - TensorFlow will use CPU")
    print("\n💡 To enable GPU:")
    print("   1. Install NVIDIA drivers for your GPU")
    print("   2. Install CUDA Toolkit (11.8 recommended)")
    print("   3. Install cuDNN (8.6 for CUDA 11.8)")
    print("   4. Install TensorFlow GPU:")
    print("      pip install tensorflow[and-cuda]")
    print("\n   Or for older TensorFlow:")
    print("      pip install tensorflow-gpu==2.13.0")
    print("\n   Verify installation:")
    print("      python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'")

print("\n" + "=" * 80)

# Check CUDA version if available
print("\n🔧 CUDA Information:")
if tf.test.is_built_with_cuda():
    print("   ✅ TensorFlow built with CUDA support")
    try:
        print(f"   CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"   cuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    except:
        print("   ⚠️ Could not retrieve CUDA/cuDNN versions")
else:
    print("   ❌ TensorFlow NOT built with CUDA support")
    print("   You need to install tensorflow-gpu or tensorflow[and-cuda]")

print("\n" + "=" * 80)

# Recommendations
print("\n📋 RECOMMENDATIONS:")
print("=" * 80)

if len(gpus) > 0:
    print("✅ Your system is ready for GPU training!")
    print("   Models will automatically use GPU when available.")
else:
    print("⚠️ CPU-only mode detected")
    print("\nOption 1: Quick Fix (Skip Random Forest)")
    print("   - LSTM model already trained successfully")
    print("   - Random Forest not needed for demo")
    print("   - Continue with next steps")
    print("\nOption 2: Enable GPU (Better Performance)")
    print("   - Install CUDA Toolkit 11.8")
    print("   - Install cuDNN 8.6")
    print("   - Reinstall: pip install tensorflow[and-cuda]")
    print("\nOption 3: Use Lighter Model")
    print("   - Reduce LSTM units (128→64, 64→32)")
    print("   - Reduce epochs (50→20)")
    print("   - Smaller sequence length (30→15)")

print("\n" + "=" * 80)