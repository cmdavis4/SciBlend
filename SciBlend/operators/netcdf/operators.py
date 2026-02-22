import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, FloatProperty, EnumProperty, BoolProperty
import numpy as np
import os
import math
import time
from datetime import datetime, timedelta
from ..utils.scene import clear_scene, keyframe_visibility_single_frame, enforce_constant_interpolation
from ..utils.scene import get_import_target_collection

try:
    import xarray as xr
    import dask
    import dask.array as da
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import pyopenvdb as vdb
    PYOPENVDB_AVAILABLE = True
except ImportError:
    PYOPENVDB_AVAILABLE = False

class ImportNetCDFOperator(bpy.types.Operator, ImportHelper):
    """Import NetCDF files into Blender."""
    bl_idname = "import_netcdf.animation"
    bl_label = "Import NetCDF Animation"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".nc"
    filter_glob: StringProperty(default="*.nc;*.nc4", options={'HIDDEN'})

    variable_name: StringProperty(name="Variable Name", description="Name of the variable to visualize", default="")
    time_dimension: StringProperty(name="Time Dimension", description="Name of the time dimension", default="time")
    scale_factor: FloatProperty(name="Scale Factor", description="Scale factor for imported objects", default=1.0, min=0.0001, max=100.0)
    axis_forward: EnumProperty(name="Forward", items=[('X','X Forward',''),('Y','Y Forward',''),('Z','Z Forward',''),('-X','-X Forward',''),('-Y','-Y Forward',''),('-Z','-Z Forward','')], default='Y')
    axis_up: EnumProperty(name="Up", items=[('X','X Up',''),('Y','Y Up',''),('Z','Z Up',''),('-X','-X Up',''),('-Y','-Y Up',''),('-Z','-Z Up','')], default='Z')
    use_sphere: BoolProperty(name="Spherical Projection", description="Project data onto a sphere", default=False)
    sphere_radius: FloatProperty(name="Sphere Radius", default=1.0, min=0.01, max=100.0)
    height_scale: FloatProperty(name="Height Scale", default=0.01, min=0.0001, max=1.0, soft_min=0.001, soft_max=0.1)

    # 3D Visualization options
    visualization_mode: EnumProperty(
        name="Visualization Mode",
        items=[
            ('SURFACE_2D', '2D Surface', 'Create 2D height-mapped surface (current behavior)'),
            ('ISOSURFACE', 'Isosurface', 'Extract 3D isosurface at threshold value using marching cubes'),
            ('VOLUME', 'Volume', 'Create volume rendering with OpenVDB')
        ],
        default='SURFACE_2D'
    )

    # For 3D data dimension selection
    z_dimension: StringProperty(
        name="Z Dimension",
        description="Name of the vertical/depth dimension for 3D data",
        default="lev"
    )

    # For ISOSURFACE mode
    iso_value: FloatProperty(
        name="Iso Value",
        description="Threshold value for isosurface extraction",
        default=0.0
    )

    auto_iso_value: BoolProperty(
        name="Auto Iso Value",
        description="Use median of data as iso value",
        default=True
    )

    # For VOLUME mode
    vdb_resolution: EnumProperty(
        name="VDB Resolution",
        items=[
            ('64', '64³', 'Low resolution - faster but less detailed'),
            ('128', '128³', 'Medium resolution - balanced'),
            ('256', '256³', 'High resolution - detailed but slower'),
            ('512', '512³', 'Very high resolution - very detailed, slow')
        ],
        default='128'
    )

    def draw(self, context):
        """Draw the operator panel with conditional UI based on visualization mode."""
        layout = self.layout

        # Variable selection
        layout.prop(self, "variable_name")
        layout.prop(self, "time_dimension")

        # Visualization mode selection
        layout.separator()
        layout.label(text="Visualization Mode:")
        layout.prop(self, "visualization_mode", text="")

        layout.separator()

        # Mode-specific options
        if self.visualization_mode == 'SURFACE_2D':
            box = layout.box()
            box.label(text="2D Surface Options:")
            box.prop(self, "use_sphere")
            if self.use_sphere:
                box.prop(self, "sphere_radius")
                box.prop(self, "height_scale")

        elif self.visualization_mode == 'ISOSURFACE':
            box = layout.box()
            box.label(text="Isosurface Options:")
            box.prop(self, "z_dimension")
            box.prop(self, "auto_iso_value")
            if not self.auto_iso_value:
                box.prop(self, "iso_value")

        elif self.visualization_mode == 'VOLUME':
            box = layout.box()
            box.label(text="Volume Options:")
            box.prop(self, "z_dimension")
            box.prop(self, "vdb_resolution")

        # Common options
        layout.separator()
        layout.label(text="Common Options:")
        layout.prop(self, "scale_factor")
        layout.prop(self, "axis_forward")
        layout.prop(self, "axis_up")

    def execute(self, context):
        if not XARRAY_AVAILABLE:
            self.report({'ERROR'}, "xarray/dask is not available. Please install xarray and dask.")
            return {'CANCELLED'}

        # Check for mode-specific dependencies
        if self.visualization_mode == 'ISOSURFACE' and not SKIMAGE_AVAILABLE:
            self.report({'ERROR'}, "scikit-image is not available. Please install scikit-image for isosurface extraction.")
            return {'CANCELLED'}

        if self.visualization_mode == 'VOLUME' and not PYOPENVDB_AVAILABLE:
            self.report({'ERROR'}, "pyopenvdb is not available. Please install pyopenvdb for volume rendering.")
            return {'CANCELLED'}

        try:
            # Open dataset with lazy loading
            dataset = xr.open_dataset(self.filepath, chunks='auto')

            # Auto-detect variable if not specified
            if not self.variable_name:
                for var_name in dataset.data_vars:
                    var = dataset[var_name]
                    if len(var.shape) >= 2:
                        self.variable_name = var_name
                        break

            if self.variable_name not in dataset.data_vars:
                self.report({'ERROR'}, f"Variable '{self.variable_name}' not found")
                dataset.close()
                return {'CANCELLED'}

            variable = dataset[self.variable_name]

            # Validate dimensions based on mode
            min_dims = 3 if self.visualization_mode in ['ISOSURFACE', 'VOLUME'] else 2
            if len(variable.shape) < min_dims:
                self.report({'ERROR'}, f"Variable '{self.variable_name}' must have at least {min_dims} dimensions for {self.visualization_mode} mode")
                dataset.close()
                return {'CANCELLED'}

            spatial_dims = [dim for dim in variable.dims if dim != self.time_dimension]
            min_spatial = 3 if self.visualization_mode in ['ISOSURFACE', 'VOLUME'] else 2
            if len(spatial_dims) < min_spatial:
                self.report({'ERROR'}, f"Need at least {min_spatial} spatial dimensions for {self.visualization_mode} mode")
                dataset.close()
                return {'CANCELLED'}
            # Branch based on visualization mode
            if self.visualization_mode == 'SURFACE_2D':
                result = self.create_2d_surface(context, dataset, variable)
            elif self.visualization_mode == 'ISOSURFACE':
                result = self.create_isosurface(context, dataset, variable)
            elif self.visualization_mode == 'VOLUME':
                result = self.create_volume(context, dataset, variable)
            else:
                self.report({'ERROR'}, f"Unknown visualization mode: {self.visualization_mode}")
                result = {'CANCELLED'}

            dataset.close()
            return result
        except Exception as e:
            self.report({'ERROR'}, f"Error importing file: {str(e)}")
            return {'CANCELLED'}

    def _identify_spatial_dimensions(self, spatial_dims):
        """Identify X, Y, Z dimensions from common NetCDF naming conventions.

        Args:
            spatial_dims: List of spatial dimension names (excluding time)

        Returns:
            tuple: (x_dim, y_dim, z_dim) or (x_dim, y_dim, None) for 2D data
        """
        # Common naming patterns
        x_names = ['x', 'lon', 'longitude', 'xc', 'xi', 'nlon', 'west_east']
        y_names = ['y', 'lat', 'latitude', 'yc', 'yi', 'nlat', 'south_north']
        z_names = ['z', 'lev', 'level', 'depth', 'altitude', 'height', 'pressure', 'plev', 'bottom_top']

        # Convert to lowercase for matching
        dims_lower = {dim: dim.lower() for dim in spatial_dims}

        # Find X dimension
        x_dim = None
        for dim, dim_lower in dims_lower.items():
            if any(x_name in dim_lower for x_name in x_names):
                x_dim = dim
                break

        # Find Y dimension
        y_dim = None
        for dim, dim_lower in dims_lower.items():
            if any(y_name in dim_lower for y_name in y_names):
                y_dim = dim
                break

        # Find Z dimension (if needed for 3D)
        z_dim = None
        if len(spatial_dims) >= 3:
            # First try the user-specified z_dimension
            if self.z_dimension in spatial_dims:
                z_dim = self.z_dimension
            else:
                # Try to find from common names
                for dim, dim_lower in dims_lower.items():
                    if any(z_name in dim_lower for z_name in z_names):
                        z_dim = dim
                        break

        # Fallback: use last 2 or 3 dimensions if not found
        if x_dim is None or y_dim is None:
            if len(spatial_dims) >= 2:
                # Default to last two dimensions as y, x (typical NetCDF convention)
                if x_dim is None:
                    x_dim = spatial_dims[-1]
                if y_dim is None:
                    y_dim = spatial_dims[-2]

        if z_dim is None and len(spatial_dims) >= 3:
            # Use remaining dimension or first dimension
            remaining = [d for d in spatial_dims if d not in [x_dim, y_dim]]
            if remaining:
                z_dim = remaining[0]
            else:
                z_dim = spatial_dims[-3]

        return x_dim, y_dim, z_dim

    def create_2d_surface(self, context, dataset, variable):
        """Create 2D height-mapped surface from NetCDF data (original functionality)."""
        # Get coordinate arrays
        coords = {dim: variable.coords[dim].values for dim in variable.dims if dim in variable.coords}

        # Check for time dimension
        has_time = self.time_dimension in variable.dims and len(variable.coords[self.time_dimension]) > 0
        if has_time:
            time_steps = len(variable.coords[self.time_dimension])
        else:
            time_steps = 1

        loop_count = max(1, getattr(context.scene.x3d_import_settings, "loop_count", 1))
        if has_time:
            context.scene.frame_start = 1
            context.scene.frame_end = time_steps * loop_count

        if context.scene.x3d_import_settings.overwrite_scene:
            clear_scene(context)

        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        target_collection = get_import_target_collection(context, context.scene.x3d_import_settings.import_to_new_collection, base_name)

        # For material creation, we need to sample some data to get value range
        if has_time:
            sample_data = variable.isel({self.time_dimension: 0}).compute().values
        else:
            sample_data = variable.compute().values
        material = self.create_material([sample_data], self.variable_name)

        start_wall = time.time()
        print(f"[NetCDF] Starting 2D surface import of {time_steps} time step(s) at {datetime.now().strftime('%H:%M:%S')}")

        for frame in range(time_steps):
            per_item_start = time.time()

            # Lazy load only this time slice
            if has_time:
                data = variable.isel({self.time_dimension: frame}).compute().values
            else:
                data = variable.compute().values

            vertices = []
            faces = []

            # Get spatial dimensions using smart detection
            spatial_dims_list = [dim for dim in variable.dims if dim != self.time_dimension]
            dim_x, dim_y, _ = self._identify_spatial_dimensions(spatial_dims_list)

            rows = len(variable.coords[dim_y])
            cols = len(variable.coords[dim_x])
            x_coords = coords.get(dim_x, np.arange(cols))
            y_coords = coords.get(dim_y, np.arange(rows))

            # Build vertices
            for i in range(rows):
                for j in range(cols):
                    if self.use_sphere:
                        lon = float(x_coords[j])
                        lat = float(y_coords[i])
                        lon_rad = math.radians(lon)
                        lat_rad = math.radians(lat)
                        radius = self.sphere_radius * self.scale_factor
                        x = radius * math.cos(lat_rad) * math.cos(lon_rad)
                        y = radius * math.cos(lat_rad) * math.sin(lon_rad)
                        z = radius * math.sin(lat_rad)
                    else:
                        x = float(x_coords[j]) * self.scale_factor
                        y = float(y_coords[i]) * self.scale_factor
                        z = 0.0

                    try:
                        value = data[i, j]
                        if np.isscalar(value):
                            height = float(value) if not np.isnan(value) else 0.0
                        else:
                            if isinstance(value, np.ndarray) and value.size > 0 and not np.all(np.isnan(value)):
                                height = float(np.nanmean(value))
                            else:
                                height = 0.0
                    except IndexError:
                        height = 0.0

                    if self.use_sphere:
                        factor = 1.0 + (height * self.height_scale)
                        x *= factor
                        y *= factor
                        z *= factor
                    else:
                        z = height

                    vertices.append((x, y, z))

            # Build faces
            for i in range(rows - 1):
                for j in range(cols - 1):
                    v0 = i * cols + j
                    v1 = v0 + 1
                    v2 = (i + 1) * cols + j + 1
                    v3 = (i + 1) * cols + j
                    faces.append([v0, v1, v2, v3])

            # Create mesh
            mesh_name = f"Frame_{frame+1}" if has_time else f"NetCDF_{self.variable_name}"
            mesh = bpy.data.meshes.new(mesh_name)
            obj = bpy.data.objects.new(mesh_name, mesh)
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            # Add attributes
            self.add_attributes(mesh, data, coords, dim_x, dim_y)

            # Link to collection
            if target_collection is not None:
                target_collection.objects.link(obj)
            else:
                context.collection.objects.link(obj)

            obj.data.materials.append(material)

            # Setup animation
            if has_time:
                self.setup_animation(obj, frame, time_steps, loop_count)

            # Progress reporting
            duration = time.time() - per_item_start
            processed = frame + 1
            elapsed = time.time() - start_wall
            avg = (elapsed / processed) if processed > 0 else 0.0
            remaining = max(0, time_steps - processed)
            eta_dt = datetime.now() + timedelta(seconds=avg * remaining) if avg > 0 else datetime.now()
            print(f"[NetCDF] Imported time step {processed}/{time_steps} in {duration:.2f}s. ETA ~ {eta_dt.strftime('%H:%M:%S')}")

        self.report({'INFO'}, f"Imported {time_steps} 2D surface(s) successfully")
        return {'FINISHED'}

    def create_isosurface(self, context, dataset, variable):
        """Extract 3D isosurface using marching cubes algorithm."""
        # Identify spatial dimensions
        has_time = self.time_dimension in variable.dims
        spatial_dims = [dim for dim in variable.dims if dim != self.time_dimension]

        if len(spatial_dims) < 3:
            self.report({'ERROR'}, "Need 3 spatial dimensions for isosurface")
            return {'CANCELLED'}

        # Identify z, y, x dimensions using smart detection
        x_dim, y_dim, z_dim = self._identify_spatial_dimensions(spatial_dims)

        if z_dim is None:
            self.report({'ERROR'}, "Could not identify Z dimension for 3D data")
            return {'CANCELLED'}

        print(f"[NetCDF] Detected dimensions: X='{x_dim}', Y='{y_dim}', Z='{z_dim}'")

        # Get coordinate arrays
        z_coords = variable.coords[z_dim].values
        y_coords = variable.coords[y_dim].values
        x_coords = variable.coords[x_dim].values

        # Calculate spacing for proper vertex positioning
        # Spacing is in (z, y, x) order to match the data array dimensions
        spacing = (
            abs(np.mean(np.diff(z_coords))) if len(z_coords) > 1 else 1.0,
            abs(np.mean(np.diff(y_coords))) if len(y_coords) > 1 else 1.0,
            abs(np.mean(np.diff(x_coords))) if len(x_coords) > 1 else 1.0
        )

        time_steps = len(variable.coords[self.time_dimension]) if has_time else 1
        loop_count = max(1, getattr(context.scene.x3d_import_settings, "loop_count", 1))

        if has_time:
            context.scene.frame_start = 1
            context.scene.frame_end = time_steps * loop_count

        if context.scene.x3d_import_settings.overwrite_scene:
            clear_scene(context)

        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        target_collection = get_import_target_collection(context, context.scene.x3d_import_settings.import_to_new_collection, base_name)

        start_wall = time.time()
        print(f"[NetCDF] Starting isosurface extraction of {time_steps} time step(s) at {datetime.now().strftime('%H:%M:%S')}")

        for frame in range(time_steps):
            per_item_start = time.time()

            # Lazy load only this time slice
            if has_time:
                data_slice = variable.isel({self.time_dimension: frame}).compute()
            else:
                data_slice = variable.compute()

            # Transpose to (z, y, x) order
            data_3d = data_slice.transpose(z_dim, y_dim, x_dim).values

            # Determine iso value
            if self.auto_iso_value:
                valid_data = data_3d[~np.isnan(data_3d)]
                if len(valid_data) > 0:
                    iso_value = float(np.median(valid_data))
                else:
                    self.report({'WARNING'}, f"Frame {frame}: No valid data for auto iso value, skipping")
                    continue
            else:
                iso_value = self.iso_value

            print(f"[NetCDF] Frame {frame+1}: Using iso value = {iso_value}")

            # Run marching cubes
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    data_3d,
                    level=iso_value,
                    spacing=spacing,
                    allow_degenerate=False
                )
            except (ValueError, RuntimeError) as e:
                self.report({'WARNING'}, f"Frame {frame}: Marching cubes failed: {str(e)}")
                continue

            if len(verts) == 0:
                self.report({'WARNING'}, f"Frame {frame}: No vertices generated at iso value {iso_value}")
                continue

            # Reorder vertices from (z, y, x) to (x, y, z) for Blender's coordinate system
            verts = verts[:, [2, 1, 0]]

            # Scale vertices
            verts = verts * self.scale_factor

            # Create Blender mesh
            mesh_name = f"Isosurface_{frame+1}" if has_time else f"Isosurface_{self.variable_name}"
            mesh = bpy.data.meshes.new(mesh_name)
            obj = bpy.data.objects.new(mesh_name, mesh)

            mesh.from_pydata(verts.tolist(), [], faces.tolist())
            mesh.update()

            # Add vertex attribute for iso values
            if len(values) > 0:
                attr = mesh.attributes.new(name=self.variable_name, type='FLOAT', domain='POINT')
                for i, val in enumerate(values):
                    if i < len(attr.data):
                        attr.data[i].value = float(val)

            # Link to collection
            if target_collection is not None:
                target_collection.objects.link(obj)
            else:
                context.collection.objects.link(obj)

            # Apply material
            material = self.create_material([data_3d], self.variable_name)
            obj.data.materials.append(material)

            # Setup animation
            if has_time:
                self.setup_animation(obj, frame, time_steps, loop_count)

            # Progress reporting
            duration = time.time() - per_item_start
            processed = frame + 1
            elapsed = time.time() - start_wall
            avg = (elapsed / processed) if processed > 0 else 0.0
            remaining = max(0, time_steps - processed)
            eta_dt = datetime.now() + timedelta(seconds=avg * remaining) if avg > 0 else datetime.now()
            print(f"[NetCDF] Extracted isosurface {processed}/{time_steps} ({len(verts)} verts) in {duration:.2f}s. ETA ~ {eta_dt.strftime('%H:%M:%S')}")

        self.report({'INFO'}, f"Extracted {time_steps} isosurface(s) successfully")
        return {'FINISHED'}

    def create_volume(self, context, dataset, variable):
        """Create volume rendering using OpenVDB format."""
        import tempfile

        # Identify spatial dimensions
        has_time = self.time_dimension in variable.dims
        spatial_dims = [dim for dim in variable.dims if dim != self.time_dimension]

        if len(spatial_dims) < 3:
            self.report({'ERROR'}, "Need 3 spatial dimensions for volume")
            return {'CANCELLED'}

        # Identify z, y, x dimensions using smart detection
        x_dim, y_dim, z_dim = self._identify_spatial_dimensions(spatial_dims)

        if z_dim is None:
            self.report({'ERROR'}, "Could not identify Z dimension for 3D data")
            return {'CANCELLED'}

        print(f"[NetCDF] Detected dimensions: X='{x_dim}', Y='{y_dim}', Z='{z_dim}'")

        time_steps = len(variable.coords[self.time_dimension]) if has_time else 1
        loop_count = max(1, getattr(context.scene.x3d_import_settings, "loop_count", 1))

        if has_time:
            context.scene.frame_start = 1
            context.scene.frame_end = time_steps * loop_count

        if context.scene.x3d_import_settings.overwrite_scene:
            clear_scene(context)

        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        target_collection = get_import_target_collection(context, context.scene.x3d_import_settings.import_to_new_collection, base_name)

        resolution = int(self.vdb_resolution)

        # Setup viewport for volume rendering
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.clip_end = 100000.0

        start_wall = time.time()
        print(f"[NetCDF] Starting volume creation of {time_steps} time step(s) at {datetime.now().strftime('%H:%M:%S')}")
        print(f"[NetCDF] Target resolution: {resolution}³")

        vdb_files = []

        for frame in range(time_steps):
            per_item_start = time.time()

            # Lazy load only this time slice
            if has_time:
                data_slice = variable.isel({self.time_dimension: frame})
            else:
                data_slice = variable

            # Get original dimensions
            orig_z = len(data_slice.coords[z_dim])
            orig_y = len(data_slice.coords[y_dim])
            orig_x = len(data_slice.coords[x_dim])

            print(f"[NetCDF] Frame {frame+1}: Original size ({orig_z}, {orig_y}, {orig_x})")

            # Resample to target resolution if needed
            needs_resampling = (orig_z != resolution or orig_y != resolution or orig_x != resolution)

            if needs_resampling:
                print(f"[NetCDF] Frame {frame+1}: Resampling to ({resolution}, {resolution}, {resolution})...")
                # Use xarray's interpolation with dask support
                z_new = np.linspace(float(data_slice[z_dim].min()), float(data_slice[z_dim].max()), resolution)
                y_new = np.linspace(float(data_slice[y_dim].min()), float(data_slice[y_dim].max()), resolution)
                x_new = np.linspace(float(data_slice[x_dim].min()), float(data_slice[x_dim].max()), resolution)

                data_slice = data_slice.interp(
                    {z_dim: z_new, y_dim: y_new, x_dim: x_new},
                    method='linear'
                )

            # Now compute the data
            # Transpose to (x, y, z) order for Blender/VDB coordinate system
            data_3d = data_slice.transpose(x_dim, y_dim, z_dim).compute().values

            # Handle NaN values - replace with 0
            data_3d = np.nan_to_num(data_3d, nan=0.0)

            print(f"[NetCDF] Frame {frame+1}: Data range [{np.min(data_3d):.3f}, {np.max(data_3d):.3f}]")

            # Create VDB grid
            grid = vdb.FloatGrid()
            grid.name = self.variable_name

            # Copy data to grid (pyopenvdb expects C-order array in X, Y, Z order)
            grid.copyFromArray(data_3d.astype(np.float32))

            # Set grid transform for proper scaling
            # VDB uses index space, so we set voxel size based on scale factor
            voxel_size = self.scale_factor
            grid.transform = vdb.createLinearTransform([[voxel_size, 0, 0, 0],
                                                         [0, voxel_size, 0, 0],
                                                         [0, 0, voxel_size, 0],
                                                         [0, 0, 0, 1]])

            # Write to temporary VDB file
            vdb_file = tempfile.NamedTemporaryFile(suffix='.vdb', delete=False, dir=tempfile.gettempdir())
            vdb_path = vdb_file.name
            vdb_file.close()

            vdb.write(vdb_path, grids=[grid])
            vdb_files.append(vdb_path)

            print(f"[NetCDF] Frame {frame+1}: Wrote VDB to {vdb_path}")

            # Import VDB into Blender
            bpy.ops.object.volume_import(filepath=vdb_path, files=[{"name": os.path.basename(vdb_path)}], directory=os.path.dirname(vdb_path))

            # Get the imported volume object (should be the active object)
            vol_obj = context.active_object
            if vol_obj and vol_obj.type == 'VOLUME':
                # Rename
                vol_name = f"Volume_{frame+1}" if has_time else f"Volume_{self.variable_name}"
                vol_obj.name = vol_name

                # Move to target collection
                if target_collection is not None:
                    # Unlink from current collections
                    for coll in vol_obj.users_collection:
                        coll.objects.unlink(vol_obj)
                    # Link to target collection
                    target_collection.objects.link(vol_obj)

                # Setup animation
                if has_time:
                    self.setup_animation(vol_obj, frame, time_steps, loop_count)

                print(f"[NetCDF] Frame {frame+1}: Imported volume object '{vol_name}'")

            # Progress reporting
            duration = time.time() - per_item_start
            processed = frame + 1
            elapsed = time.time() - start_wall
            avg = (elapsed / processed) if processed > 0 else 0.0
            remaining = max(0, time_steps - processed)
            eta_dt = datetime.now() + timedelta(seconds=avg * remaining) if avg > 0 else datetime.now()
            print(f"[NetCDF] Created volume {processed}/{time_steps} in {duration:.2f}s. ETA ~ {eta_dt.strftime('%H:%M:%S')}")

        # Clean up temporary VDB files
        for vdb_path in vdb_files:
            try:
                os.unlink(vdb_path)
                print(f"[NetCDF] Cleaned up temporary file: {vdb_path}")
            except Exception as e:
                print(f"[NetCDF] Warning: Could not delete temporary file {vdb_path}: {e}")

        self.report({'INFO'}, f"Created {time_steps} volume(s) successfully")
        return {'FINISHED'}

    def add_attributes(self, mesh, data, coords, dim_x, dim_y):
        """Add per-point attributes from NetCDF arrays, matching vertex count."""
        rows, cols = int(data.shape[-2]), int(data.shape[-1])
        vertex_count = rows * cols

        attr = mesh.attributes.new(name=self.variable_name, type='FLOAT', domain='POINT')
        flat_values = np.ravel(data).astype(float)
        for idx in range(min(vertex_count, flat_values.size)):
            v = flat_values[idx]
            attr.data[idx].value = float(v) if not np.isnan(v) else 0.0

        if dim_x in coords:
            x_vals = np.asarray(coords[dim_x], dtype=float)
            attr_x = mesh.attributes.new(name=f"coord_{dim_x}", type='FLOAT', domain='POINT')
            k = 0
            for i in range(rows):
                for j in range(cols):
                    attr_x.data[k].value = float(x_vals[j]) if j < x_vals.size else 0.0
                    k += 1
        if dim_y in coords:
            y_vals = np.asarray(coords[dim_y], dtype=float)
            attr_y = mesh.attributes.new(name=f"coord_{dim_y}", type='FLOAT', domain='POINT')
            k = 0
            for i in range(rows):
                for j in range(cols):
                    attr_y.data[k].value = float(y_vals[i]) if i < y_vals.size else 0.0
                    k += 1

    def create_material(self, variable_data, variable_name):
        """Create a material that maps scalar values to colors."""
        material = bpy.data.materials.new(name=f"NetCDF_{variable_name}_Material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        attribute_node = nodes.new(type='ShaderNodeAttribute')
        attribute_node.attribute_name = variable_name
        map_range = nodes.new(type='ShaderNodeMapRange')
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        output = nodes.new(type='ShaderNodeOutputMaterial')
        try:
            data = np.asarray(variable_data)
            flat_data = data.ravel()
            valid_mask = np.logical_and(~np.isnan(flat_data), ~np.isinf(flat_data))
            if valid_mask.size > 0 and valid_mask.any():
                valid_data = flat_data[valid_mask]
                if valid_data.size > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                else:
                    min_val = 0.0
                    max_val = 1.0
            else:
                min_val = 0.0
                max_val = 1.0
        except Exception:
            min_val = 0.0
            max_val = 1.0
        map_range.inputs['From Min'].default_value = min_val
        map_range.inputs['From Max'].default_value = max_val
        map_range.inputs['To Min'].default_value = 0.0
        map_range.inputs['To Max'].default_value = 1.0
        links.new(attribute_node.outputs['Fac'], map_range.inputs['Value'])
        links.new(map_range.outputs['Result'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        return material

    def setup_animation(self, obj, frame, time_steps, loop_count):
        """Insert keyframes to reveal one frame per time step, repeated for loop_count."""
        base = frame + 1
        for k in range(loop_count):
            occurrence = base + (k * time_steps)
            keyframe_visibility_single_frame(obj, occurrence)
        enforce_constant_interpolation(obj)

__all__ = ["ImportNetCDFOperator"] 