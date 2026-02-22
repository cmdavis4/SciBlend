# Testing the NetCDF 3D Visualization Updates

## Quick Testing (Without Building Wheels)

### Step 1: Prepare Dependencies

Since we changed from netCDF4 to xarray/dask, you need to install these in Blender's Python environment:

```bash
# Find your Blender Python (usually in Blender installation directory)
# On Linux, typically:
BLENDER_PYTHON="/path/to/blender/4.5/python/bin/python3.11"

# Or find it this way:
blender --background --python-expr "import sys; print(sys.executable)"

# Install required packages
$BLENDER_PYTHON -m pip install xarray dask scikit-image pyopenvdb
```

### Step 2: Install Extension in Development Mode

**Method A: Using Blender UI**
1. Open Blender 4.5+
2. Go to `Edit > Preferences > Add-ons`
3. Click the dropdown arrow next to the `Install...` button
4. Select `Install from Disk (Development Mode)`
5. Navigate to `/home/cmdavis4/projects/SciBlend/`
6. Click `Install from Disk`
7. Enable the "SciBlend" addon in the list

**Method B: Using Symlink (Linux/Mac)**
```bash
# Find Blender extensions directory
BLENDER_EXT_DIR=~/.config/blender/4.5/extensions/user_default

# Create symlink
ln -s /home/cmdavis4/projects/SciBlend $BLENDER_EXT_DIR/sciblend

# Restart Blender and enable the addon
```

**Method C: Using Blender Command Line**
```bash
blender --command extension install-file /home/cmdavis4/projects/SciBlend --enable
```

### Step 3: Test the Changes

1. In Blender, open the SciBlend panel (View3D sidebar, SciBlend tab)
2. Click "NetCDF" button
3. Select a .nc file with 3D data
4. You should see the new UI with:
   - **Visualization Mode** dropdown with 3 options:
     - 2D Surface (original)
     - Isosurface (new)
     - Volume (new)

### Step 4: Test Each Mode

**Test 2D Surface (Original Behavior):**
- Select "2D Surface" mode
- Choose a 2D variable
- Should work exactly as before

**Test Isosurface:**
- Select "Isosurface" mode
- Choose a 3D variable (e.g., temperature with time, lev, lat, lon)
- Set `z_dimension` to your vertical dimension name (e.g., "lev", "depth")
- Enable "Auto Iso Value" for automatic threshold detection
- Import and you should see a 3D mesh surface

**Test Volume:**
- Select "Volume" mode
- Choose a 3D variable
- Set `z_dimension` to your vertical dimension name
- Select resolution (start with 128³)
- Import and you should see a volume object

---

## Full Build (For Distribution)

If you need to create a distributable .zip extension with all dependencies:

### Step 1: Update Constraints Files

Add the new dependencies to `/home/cmdavis4/projects/SciBlend/constraints/linux-x64.txt`:

```bash
# Add these lines:
xarray==2024.11.0
dask==2024.11.2
scikit-image==0.24.0
pyopenvdb==11.0.0  # Note: This may not be available as wheel for all platforms
```

Do the same for:
- `constraints/base.txt` (Windows)
- `constraints/macos-x64.txt`
- `constraints/macos-arm64.txt`

### Step 2: Download Wheels

```bash
cd /home/cmdavis4/projects/SciBlend
bash scripts/fetch_wheels.sh
```

**Note:** `pyopenvdb` might not be available as a pre-built wheel for all platforms. You may need to:
- Build it from source for each platform, OR
- Document that users need to install it manually, OR
- Use a fallback approach if pyopenvdb is not available

### Step 3: Update blender_manifest.toml

After running `fetch_wheels.sh`, list all new .whl files:

```bash
find wheels/ -name "*.whl" -type f | sort
```

Add the new wheel paths to the `wheels = [...]` section in `blender_manifest.toml`.

### Step 4: Package Extension

```bash
cd /home/cmdavis4/projects/SciBlend/..
zip -r sciblend-1.2.0.zip SciBlend/ -x "SciBlend/.git/*" "SciBlend/__pycache__/*" "SciBlend/*/__pycache__/*"
```

### Step 5: Install Packaged Extension

```bash
blender --command extension install-file sciblend-1.2.0.zip --enable
```

---

## Troubleshooting

### Missing xarray/dask Error
If you see: "xarray/dask is not available"
- Install in Blender Python: `$BLENDER_PYTHON -m pip install xarray dask`

### Missing scikit-image Error
If you see: "scikit-image is not available" in Isosurface mode
- Install: `$BLENDER_PYTHON -m pip install scikit-image`

### Missing pyopenvdb Error
If you see: "pyopenvdb is not available" in Volume mode
- Install: `$BLENDER_PYTHON -m pip install pyopenvdb`
- If installation fails, you may need to build from source or install via conda

### Extension Doesn't Show Up
- Make sure you're using Blender 4.5.1 or newer (check `blender_manifest.toml` line 19)
- Check Blender console for error messages
- Verify the extension is in the correct directory

### Syntax Errors
The code has been validated with `python3 -m py_compile`, but if you see Python errors:
- Check the Blender console (Window > Toggle System Console on Windows/Linux)
- Look for import errors or syntax issues

---

## Testing Checklist

- [ ] Extension loads without errors
- [ ] NetCDF import dialog shows new "Visualization Mode" dropdown
- [ ] 2D Surface mode works (backward compatibility)
- [ ] Isosurface mode extracts 3D surfaces
- [ ] Volume mode creates VDB volumes
- [ ] Auto iso value detection works
- [ ] Time-series animation works for all modes
- [ ] Materials are applied correctly
- [ ] Progress reporting works with ETA
- [ ] Temporary VDB files are cleaned up
- [ ] Large datasets are handled with lazy loading
