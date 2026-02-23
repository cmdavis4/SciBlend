# Adding New Dependencies to SciBlend Extension

This guide walks through adding xarray, dask, scikit-image, and pyopenvdb to the extension.

## Step 1: Download Wheels

Run the wheel fetching script:

```bash
cd /home/cmdavis4/projects/SciBlend
bash scripts/fetch_wheels.sh
```

This will download wheels for all platforms based on the updated constraint files.

## Step 2: Verify Downloaded Wheels

Check that wheels were downloaded:

```bash
# Check what was downloaded
find wheels/ -name "*.whl" | sort

# Specifically check for new dependencies
find wheels/ -name "*xarray*"
find wheels/ -name "*dask*"
find wheels/ -name "*scikit_image*"
find wheels/ -name "*scikit-image*"
find wheels/ -name "*networkx*"
find wheels/ -name "*imageio*"
```

## Step 3: About openvdb (No Action Needed!)

**Good news:** The `openvdb` module is bundled with official Blender 4.4+ downloads!

- Volume mode will work automatically with official Blender builds
- No need to add openvdb to constraint files
- No wheels to download
- **Only caveat:** Distro-packaged Blender (flatpak, snap, apt) may not include it

If users get an "openvdb not available" error, they should:
1. Download official Blender from blender.org instead of using distro packages
2. Or manually install: `$BLENDER_PYTHON -m pip install openvdb` (may require building from source)

## Step 4: Generate Wheel List for Manifest

Create a script to list all wheels:

```bash
cat > scripts/list_wheels.sh << 'EOF'
#!/bin/bash
echo "wheels = ["

# Common wheels (noarch)
for whl in ./wheels/common/*.whl; do
    [ -f "$whl" ] && echo "  \"${whl#./}\","
done

# Platform-specific wheels
for plat in windows-x64 linux-x64 macos-x64 macos-arm64; do
    for whl in ./wheels/${plat}/*.whl; do
        [ -f "$whl" ] && echo "  \"${whl#./}\","
    done
done

echo "]"
EOF

chmod +x scripts/list_wheels.sh
bash scripts/list_wheels.sh > wheels_list.txt
```

## Step 5: Update blender_manifest.toml

1. Open `blender_manifest.toml`
2. Find the `wheels = [...]` section
3. Replace it with the output from `wheels_list.txt`
4. **Important:** Make sure paths use forward slashes and start with `./wheels/`

Example format:
```toml
wheels = [
  "./wheels/common/xarray-2024.11.0-py3-none-any.whl",
  "./wheels/common/dask-2024.12.0-py3-none-any.whl",
  "./wheels/linux-x64/scikit_image-0.24.0-cp311-cp311-manylinux_2_17_x86_64.whl",
  # ... etc
]
```

## Step 6: Test the Extension

### Quick Test (Development Mode)
```bash
# Install directly from source without building
blender --command extension install-file /home/cmdavis4/projects/SciBlend --enable
```

### Full Build Test
```bash
cd /home/cmdavis4/projects/SciBlend/..
zip -r sciblend-dev.zip SciBlend/ -x "SciBlend/.git/*" "SciBlend/__pycache__/*" "SciBlend/*/__pycache__/*"
blender --command extension install-file sciblend-dev.zip --enable
```

## Step 7: Verify Dependencies in Blender

Open Blender and run in the Python console:

```python
import sys
print(f"Python: {sys.executable}")

# Check each dependency
try:
    import xarray
    print(f"✓ xarray {xarray.__version__}")
except ImportError as e:
    print(f"✗ xarray: {e}")

try:
    import dask
    print(f"✓ dask {dask.__version__}")
except ImportError as e:
    print(f"✗ dask: {e}")

try:
    from skimage import measure
    print(f"✓ scikit-image available")
except ImportError as e:
    print(f"✗ scikit-image: {e}")

try:
    import openvdb
    print(f"✓ openvdb available (bundled with Blender)")
except ImportError as e:
    print(f"⚠ openvdb: {e} (use official Blender build)")
```

## Troubleshooting

### Wheels not downloading
- Check internet connection
- Try manually with: `pip download <package> --dest ./wheels/linux-x64 --only-binary=:all: --python-version=3.11 --platform=manylinux2014_x86_64`

### Wrong Python version wheels
- Blender 4.5 uses Python 3.11, make sure `--python-version=3.11` is used

### Import errors in Blender
- Check that wheel paths in manifest are correct
- Verify wheels are actually in the `wheels/` directory
- Check Blender console for specific import errors

### openvdb not available
- This means the user is using a distro-packaged Blender
- Recommend downloading official Blender from blender.org
- Volume mode will gracefully fail with clear error message
- Users can manually install if needed (may require building from source)

## Final Checklist

- [ ] Updated all constraint files (base.txt, linux-x64.txt, macos-x64.txt, macos-arm64.txt)
- [ ] Ran `fetch_wheels.sh` successfully
- [ ] Verified wheels downloaded to `wheels/` directory
- [ ] Updated `blender_manifest.toml` with new wheel paths
- [ ] Tested extension loads in Blender
- [ ] Verified xarray/dask imports work
- [ ] Tested 2D surface mode (should work without changes)
- [ ] Tested isosurface mode (requires scikit-image)
- [ ] Tested volume mode (requires official Blender build with openvdb)
- [ ] Updated version number in `blender_manifest.toml` to 1.3.0 or appropriate

## Notes

- The constraints files now include all dependencies needed for xarray, dask, and scikit-image
- `openvdb` is NOT in constraints because it's bundled with official Blender 4.4+ downloads
- Volume mode requires official Blender builds (not distro packages like flatpak/snap)
- If users report "openvdb not available" errors, recommend downloading from blender.org
