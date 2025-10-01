# Simulation_Builder.py

import openmc
import numpy as np
import os
import sys
from tqdm import tqdm
import neutronics_material_maker as nmm
from openmc_plasma_source import tokamak_source, fusion_ring_source, fusion_point_source
from openmc_source_plotter import plot_source_energy, plot_source_position, plot_source_direction


def create_unit_geometry_source_points(n_points, z_coord, characteristic_length, model_type):
    """Generates random neutron source points within a hexagonal unit cross-section."""
    points = []
    s = characteristic_length
    z = z_coord

    # Create a bounding box large enough to fully contain the geometry.
    abs_x_max = s * 3
    abs_y_max = s * 3

    print(f"\nGenerating {n_points} source points on a unit cross-section at z={z_coord}...")

    while len(points) < n_points:
        # NOTE: This section may need modification to match the specific geometry.
        # Randomly sample within the bounding box.
        x = np.random.uniform(-abs_x_max, abs_x_max)
        y = np.random.uniform(-abs_y_max, abs_y_max)

        is_inside = False

        if model_type == 'LP':
            if (abs(x) <= s * np.sqrt(3.0) / 2) and \
               (abs(-x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s) and \
               (abs(x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s):
                is_inside = True
        elif model_type == 'HP':
            if (abs(y) <= s * np.sqrt(3.0) / 2) and \
               (abs(-x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s) and \
               (abs(x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s):
                is_inside = True
        else:
            raise ValueError("Invalid model_type. Must be 'LP' or 'HP'.")

        if is_inside:
            points.append((x, y, z))

    print(f"\nGenerated {len(points)} points for unit cross-section source.")
    return points


class NuclearFusion:
    """A class to build and run an OpenMC simulation for a nuclear fusion model."""

    def __init__(self, user_choices, config):
        try:
            # Store the config object as a class attribute.
            self.config = config

            # Selections passed from the GUI.
            self.model_type = user_choices['model_type']
            self.geometry_type = user_choices['geometry_type']
            self.source_choice = user_choices['source']
            cross_section_choice = user_choices['cross_section']

            print(f"\n--- Initializing Simulation with User Choices ---")
            print(f"Model Type: {self.model_type}")
            print(f"Geometry Type: {self.geometry_type}")
            print(f"Cross Section: {cross_section_choice.upper()}")
            print(f"-------------------------------------------------")

            self.model_params = self.config['models'][self.model_type]

            # Temporary storage for material configurations.
            self.material_configs = []
            # Dictionary to store all openmc.Material objects.
            self.materials = {}
            # Collection for all materials.
            self.all_materials_collection = None
            # Container for the simulation geometry.
            self.geometry = None

            # Basic tokamak geometry information [cm].
            self.tokamak_major_radius = self.config['geometry']['tokamak_major_radius']
            self.tokamak_minor_radius = self.config['geometry']['tokamak_minor_radius']
            # Distance from the center of the tokamak to the outboard wall.
            self.tokamak_radius = self.tokamak_major_radius + self.tokamak_minor_radius

            # Container for simulation settings.
            self.settings = None
            # List for post-processing tallies.
            self.tallies = []
            # For matching cell names to IDs in Mesh Tally settings.
            self.cell_name_to_id = {}

            # Official OpenMC libraries (ENDF/B-VII.1, JEFF 3.3, FENDL 3.2) are included in the Docker image.
            # https://openmc.org/official-data-libraries/
            # Set the path to the nuclear data library.
            if cross_section_choice == 'endf':
                self.cross_section_path = r'/app/data/endfb-vii.1-hdf5/cross_sections.xml'
            elif cross_section_choice == 'jeff':
                self.cross_section_path = r'/app/data/jeff-3.3-hdf5/cross_sections.xml'
            elif cross_section_choice == 'fendl':
                self.cross_section_path = r'/app/data/fendl-3.2-hdf5/cross_sections.xml'
            openmc.config['cross_sections'] = self.cross_section_path

        except Exception as e:
            print(f"\n\nError during NuclearFusion class initialization: {e}\n")
            raise

    def define_materials(self):
        """Defines the materials to be used in the simulation."""
        try:
            # Use predefined properties from the neutronics_material_maker (nmm) module.
            print("\n\n\nDefining materials using neutronics_material_maker and OpenMC built-in modules...")

            self.materials = {}
            openmc_materials_list = []
            # Temporary storage for creating breeder mixture.
            nmm_materials_temp = {}

            self.material_configs = [
                {'id': 100, 'nmm_name': 'eurofer', 'output_name': 'eurofer_base', 'kwargs': {
                    'temperature': self.config['materials']['eurofer']['temperature'],
                }},
                # Temperature/pressure definition is mandatory for fluids.
                {'id': 201, 'nmm_name': 'He', 'output_name': 'He_inner', 'kwargs': {
                    'temperature': self.config['materials']['he']['temperature'],
                    'pressure': self.config['materials']['he']['pressure']
                }},
                {'id': 202, 'nmm_name': 'He', 'output_name': 'He_outer', 'kwargs': {
                    'temperature': self.config['materials']['he']['temperature_purge'],
                    'pressure': self.config['materials']['he']['pressure_purge']
                }},
                {'id': 203, 'nmm_name': 'He', 'output_name': 'He_channel', 'kwargs': {
                    'temperature': self.config['materials']['he']['temperature'],
                    'pressure': self.config['materials']['he']['pressure']
                }},
                {'id': 301, 'nmm_name': 'tungsten', 'output_name': 'tungsten', 'kwargs': {}},
                # Li6 enrichment for the breeder can be defined.
                {'id': 402, 'nmm_name': 'Li4SiO4', 'output_name': 'Li4SiO4', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'temperature': self.config['materials']['breeder']['temperature'],
                }},
                {'id': 403, 'nmm_name': 'Li2TiO3', 'output_name': 'Li2TiO3', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'temperature': self.config['materials']['breeder']['temperature'],
                }},
            ]

            # Get the list of temporary materials.
            for mat_config in self.material_configs:
                mat_id, nmm_name, mat_kwargs = mat_config['id'], mat_config['nmm_name'], mat_config['kwargs']
                print(f"\nLoading base material: {nmm_name} with parameters {mat_kwargs}...")
                # Load material from the nmm module library.
                nmm_mat = nmm.Material.from_library(nmm_name, **mat_kwargs)
                # Convert nmm material properties to openmc material properties.
                openmc_mat = nmm_mat.openmc_material

                if nmm_name == 'eurofer':
                    print(f"Cloning material: {nmm_name} for different components...\n")
                    # For Pressure tube.
                    eurofer_pressure_tube_mat = openmc_mat.clone()
                    eurofer_pressure_tube_mat.id, eurofer_pressure_tube_mat.name = 101, 'eurofer_pressure_tube'
                    self.materials['eurofer_pressure_tube'] = eurofer_pressure_tube_mat
                    openmc_materials_list.append(eurofer_pressure_tube_mat)
                    print(f"  -> Created 'eurofer_pressure_tube' (ID: {eurofer_pressure_tube_mat.id})")
                    # For Pin.
                    eurofer_pin_mat = openmc_mat.clone()
                    eurofer_pin_mat.id, eurofer_pin_mat.name = 102, 'eurofer_pin'
                    self.materials['eurofer_pin'] = eurofer_pin_mat
                    openmc_materials_list.append(eurofer_pin_mat)
                    print(f"  -> Created 'eurofer_pin' (ID: {eurofer_pin_mat.id})")
                    # For First wall channel.
                    eurofer_first_wall_channel_mat = openmc_mat.clone()
                    eurofer_first_wall_channel_mat.id, eurofer_first_wall_channel_mat.name = 103, 'eurofer_first_wall_channel'
                    self.materials['eurofer_first_wall_channel'] = eurofer_first_wall_channel_mat
                    openmc_materials_list.append(eurofer_first_wall_channel_mat)
                    print(f"  -> Created 'eurofer_first_wall_channel' (ID: {eurofer_first_wall_channel_mat.id})")
                else:
                    output_name = mat_config['output_name']
                    openmc_mat.id, openmc_mat.name = mat_id, output_name
                    self.materials[output_name] = openmc_mat
                    nmm_materials_temp[output_name] = nmm_mat
                    openmc_materials_list.append(openmc_mat)
                    print(f"Material {output_name} loaded (OpenMC ID: {openmc_mat.id}).")

            # Separately configure the multiplier to add thermal scattering data.
            print(f"\nCreating material: Be12Ti...")
            Be12Ti_inner_mat = openmc.Material(name='Be12Ti_inner', material_id=501)
            Be12Ti_inner_mat.add_elements_from_formula('Be12Ti')
            Be12Ti_inner_mat.set_density('g/cm3', 2.27)
            Be12Ti_inner_mat.temperature = self.config['materials']['multiplier']['temperature']

            # Beryllium acts as a moderator, so thermal scattering data must be set.
            # Handle exception since FENDL library lacks thermal scattering data.
            if 'fendl' not in self.cross_section_path.lower():
                print(" -> Adding thermal scattering data for Be12Ti...")
                Be12Ti_inner_mat.add_s_alpha_beta('c_Be')  # Name is in cross_sections.xml.
            else:
                print(" -> FENDL library detected. Skipping thermal scattering data addition.")
            self.materials['Be12Ti_inner'] = Be12Ti_inner_mat
            openmc_materials_list.append(Be12Ti_inner_mat)

            print(f"\nCloning material: Be12Ti...")
            Be12Ti_outer_mat = Be12Ti_inner_mat.clone()
            Be12Ti_outer_mat.id, Be12Ti_outer_mat.name = 502, 'Be12Ti_outer'
            self.materials['Be12Ti_outer'] = Be12Ti_outer_mat
            openmc_materials_list.append(Be12Ti_outer_mat)

            # Create a mixture using the base materials generated above.
            print("\nCreating mixed material (Li4SiO4 + Li2TiO3)...")
            # Load nmm.Material objects to be mixed.
            mat1, mat2 = nmm_materials_temp['Li4SiO4'], nmm_materials_temp['Li2TiO3']
            # Mix 65% Li4SiO4 and 35% Li2TiO3 by volume fraction (vo).
            mixed_breeder_material = nmm.Material.from_mixture(
                materials=[mat1, mat2],
                fracs=[self.config['materials']['breeder']['mixture_Li4SiO4'],
                       self.config['materials']['breeder']['mixture_Li2TiO3']],
                percent_type='vo',
                packing_fraction=self.config['materials']['breeder']['packing_fraction'],
                temperature=self.config['materials']['breeder']['temperature'],
                name='breeder_pebble_mix'  # Specify the name of the new material.
            )
            # Add the newly created mixture to the material dictionary and list.
            mixed_openmc_material = mixed_breeder_material.openmc_material
            mixed_openmc_material.id, mixed_openmc_material.name = 401, 'breeder_pebble_mix'
            self.materials['breeder_pebble_mix'] = mixed_openmc_material
            openmc_materials_list.append(mixed_openmc_material)
            print(f"Mixed material breeder_pebble_mix created (OpenMC ID: {mixed_openmc_material.id}).")

            # Group all OpenMC Material objects into an openmc.Materials collection.
            self.all_materials_collection = openmc.Materials(openmc_materials_list)
            # Export the defined materials to an XML file required by OpenMC.
            print("\nExporting materials to materials.xml...\n")
            self.all_materials_collection.export_to_xml()
            print("materials.xml exported successfully.\n")

        except Exception as e:
            print(f"\n\nError in define_materials method: {e}\n")
            raise

    def define_geometry(self):
        """
        Defines the geometry for the OpenMC simulation.
        Ref: https://docs.openmc.org/en/v0.15.2/usersguide/geometry.html

        There are two main approaches to defining geometry in OpenMC:
        Option 1: Define geometry within OpenMC.
            - OpenMC provides basic shape functions like plane, cylinder, sphere, etc.
            - Geometry is defined using Constructive Solid Geometry (CSG).
            - No conversion is needed as it uses OpenMC's native functions.
            - It is difficult to represent complex geometries.
        Option 2: Define geometry outside of OpenMC.
            - Typically, geometry is created in CAD software and saved as a .step file.
            - Can also be created with modules like CadQuery.
            - .step or CadQuery files must be converted to a DAGMC (.h5m) file.
            - Modules like cad_to_dag or gmsh can convert .step to .h5m.
            - The .h5m file is then imported into OpenMC using DAGMC.
        """
        print(f"\n\n\nDefining geometry for {self.model_type} model using {self.geometry_type}...")

        if self.model_type == 'LP' and self.geometry_type == 'CSG':
            self._define_LP_CSG_geometry()
        elif self.model_type == 'HP' and self.geometry_type == 'CSG':
            self._define_HP_CSG_geometry()
        elif self.model_type == 'LP' and self.geometry_type == 'DAGMC':
            self._define_LP_DAGMC_geometry()
        elif self.model_type == 'HP' and self.geometry_type == 'DAGMC':
            self._define_HP_DAGMC_geometry()
        else:
            raise ValueError(f"Unsupported model/geometry combination: {self.model_type}/{self.geometry_type}")

        print("\nExporting geometry to geometry.xml...")
        self.geometry.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")

    def _define_LP_CSG_geometry(self):
        print("\n\n\nDefining geometry with CSG...")

        csg_params = self.model_params['csg_geometry']
        csg_plane = csg_params['plane']
        csg_cylinder = csg_params['cylinder']
        csg_rectangle = csg_params['rectangle']
        csg_hexagon = csg_params['hexagon']
        csg_cone = csg_params['cone']

        surfaces = {}

        # --- Planes ---
        surfaces['tokamak_major_radius'] = openmc.ZPlane(z0=self.tokamak_major_radius - 5.0,
                                                         boundary_type='vacuum')
        surfaces['first_wall'] = openmc.ZPlane(z0=self.tokamak_radius)
        surfaces['fw_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                               + csg_plane['first_wall_thickness'])
        surfaces['fw_channel_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                       + csg_plane['first_wall_thickness']
                                                       + csg_plane['fw_channel_thickness'])
        surfaces['impinging_plane'] = openmc.ZPlane(z0=self.tokamak_radius
                                                       + csg_plane['first_wall_thickness']
                                                       + csg_plane['fw_channel_thickness']
                                                       + csg_plane['fw_channel_to_impinging'])
        surfaces['multiplier_front'] = openmc.ZPlane(z0=self.tokamak_radius
                                                        + csg_plane['first_wall_thickness']
                                                        + csg_plane['fw_channel_thickness']
                                                        + csg_plane['fw_channel_to_multiplier'])
        surfaces['pin_front'] = openmc.ZPlane(z0=self.tokamak_radius
                                                 + csg_plane['first_wall_thickness']
                                                 + csg_plane['fw_channel_thickness']
                                                 + csg_plane['fw_channel_to_impinging']
                                                 + csg_plane['impinging_to_pin'])
        surfaces['pin_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                + csg_plane['first_wall_thickness']
                                                + csg_plane['fw_channel_thickness']
                                                + csg_plane['fw_channel_to_impinging']
                                                + csg_plane['impinging_to_pin']
                                                + csg_plane['pin_tip_thickness'])
        surfaces['pin_diagonal'] = openmc.ZPlane(z0=self.tokamak_radius
                                                    + csg_plane['first_wall_thickness']
                                                    + csg_plane['fw_channel_thickness']
                                                    + csg_plane['fw_channel_to_impinging']
                                                    + csg_plane['impinging_to_pin']
                                                    + csg_plane['pin_tip_thickness']
                                                    + csg_plane['cone_axial'])
        surfaces['multiplier_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                       + csg_plane['first_wall_thickness']
                                                       + csg_plane['fw_channel_thickness']
                                                       + csg_plane['fw_channel_to_multiplier']
                                                       + csg_plane['length_multiplier'])
        surfaces['axial_end'] = openmc.ZPlane(z0=self.tokamak_radius
                                                 + csg_plane['first_wall_thickness']
                                                 + csg_plane['fw_channel_thickness']
                                                 + csg_plane['fw_channel_to_impinging']
                                                 + csg_plane['impinging_to_pin']
                                                 + csg_plane['pin_tip_thickness']
                                                 + csg_plane['cone_axial']
                                                 + csg_plane['length_straight'],
                                              boundary_type='vacuum')

        # --- Cylinders ---
        surfaces['multiplier_inner'] = openmc.ZCylinder(r=csg_cylinder['multiplier_inner'])
        surfaces['tube_outer'] = openmc.ZCylinder(r=csg_cylinder['tube_outer'])
        surfaces['tube_inner'] = openmc.ZCylinder(r=csg_cylinder['tube_inner'])
        surfaces['pin_outer'] = openmc.ZCylinder(r=csg_cylinder['pin_outer'])
        surfaces['breeder_outer'] = openmc.ZCylinder(r=csg_cylinder['breeder_outer'])
        surfaces['breeder_inner'] = openmc.ZCylinder(r=csg_cylinder['breeder_inner'])
        surfaces['pin_inner'] = openmc.ZCylinder(r=csg_cylinder['pin_inner'])

        # --- Rectangles ---
        rectangles_z_coord = self.tokamak_radius + csg_plane['first_wall_thickness'] + csg_rectangle[
            'fw_back_to_channel_front'] + csg_rectangle['height'] / 2.0

        surfaces['fw_channel_center'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(csg_rectangle['pitch'] * 0, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_1'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(csg_rectangle['pitch'] * 1, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_2'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(csg_rectangle['pitch'] * 2, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_3'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(csg_rectangle['pitch'] * 3, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_4'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(csg_rectangle['pitch'] * 4, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_1'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(-csg_rectangle['pitch'] * 1, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_2'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(-csg_rectangle['pitch'] * 2, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_3'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(-csg_rectangle['pitch'] * 3, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_4'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='y',
            origin=(-csg_rectangle['pitch'] * 4, rectangles_z_coord)
        )

        # --- Hexagons ---
        surfaces['unit_hexagon'] = openmc.model.HexagonalPrism(
            edge_length=csg_hexagon['unit'],
            origin=(0.0, 0.0),
            orientation='y',
            boundary_type='periodic'
        )
        surfaces['multiplier_hexagon'] = openmc.model.HexagonalPrism(
            edge_length=csg_hexagon['multiplier'],
            origin=(0.0, 0.0),
            orientation='y'
        )

        # --- Cones ---
        surfaces['outer_pin_diagonal'] = openmc.model.ZConeOneSided(
            z0=self.tokamak_radius
               + csg_plane['first_wall_thickness']
               + csg_plane['fw_channel_thickness']
               + csg_plane['fw_channel_to_impinging']
               + csg_plane['impinging_to_pin']
               + csg_plane['pin_tip_thickness']
               + csg_plane['cone_axial']
               - (csg_cylinder['pin_outer'] / np.tan(csg_cone['diagonal_angle'])),
            x0=0.0,
            y0=0.0,
            r2=(np.tan(csg_cone['diagonal_angle'])) ** 2
        )
        surfaces['inner_pin_diagonal'] = openmc.model.ZConeOneSided(
            z0=self.tokamak_radius
               + csg_plane['first_wall_thickness']
               + csg_plane['fw_channel_thickness']
               + csg_plane['fw_channel_to_impinging']
               + csg_plane['impinging_to_pin']
               + csg_plane['pin_tip_thickness']
               + csg_plane['cone_axial']
               - (csg_cylinder['breeder_outer'] / np.tan(csg_cone['diagonal_angle'])),
            x0=0.0,
            y0=0.0,
            r2=(np.tan(csg_cone['diagonal_angle'])) ** 2
        )

        # regions
        regions = {}

        regions['simulation_volume'] = -surfaces['unit_hexagon'] & +surfaces['tokamak_major_radius'] & -surfaces[
            'axial_end']

        regions['first_wall'] = -surfaces['unit_hexagon'] & +surfaces['first_wall'] & -surfaces['fw_back']

        regions['he_channel_center'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_center']
        regions['he_channel_pos_1'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_1']
        regions['he_channel_pos_2'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_2']
        regions['he_channel_pos_3'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_3']
        regions['he_channel_pos_4'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_4']
        regions['he_channel_neg_1'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_1']
        regions['he_channel_neg_2'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_2']
        regions['he_channel_neg_3'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_3']
        regions['he_channel_neg_4'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_4']
        regions['he_channel'] = regions['he_channel_center'] | regions['he_channel_pos_1'] | regions[
            'he_channel_pos_2'] | regions['he_channel_pos_3'] | regions['he_channel_pos_4'] | regions[
                                    'he_channel_neg_1'] | regions['he_channel_neg_2'] | regions['he_channel_neg_3'] | \
                                regions['he_channel_neg_4']

        regions['first_wall_channel_wo_he'] = -surfaces['unit_hexagon'] & +surfaces['fw_back'] & -surfaces[
            'fw_channel_back']
        regions['first_wall_channel'] = regions['first_wall_channel_wo_he'] & ~(regions['he_channel'])

        regions['tube_axial'] = -surfaces['tube_outer'] & +surfaces['tube_inner'] & +surfaces['fw_channel_back'] & - \
        surfaces['axial_end']
        regions['tube_radial'] = -surfaces['tube_inner'] & +surfaces['fw_channel_back'] & -surfaces['impinging_plane']
        regions['tube'] = regions['tube_axial'] | regions['tube_radial']

        regions['breeder'] = -surfaces['breeder_outer'] & -surfaces['inner_pin_diagonal'] & +surfaces[
            'breeder_inner'] & +surfaces['pin_back'] & -surfaces['axial_end']

        regions['pin_outer'] = -surfaces['pin_outer'] & -surfaces['outer_pin_diagonal'] & +surfaces['pin_inner'] & + \
        surfaces['pin_front'] & -surfaces['axial_end']
        regions['pin'] = regions['pin_outer'] & ~regions['breeder']

        regions['he_main_outer'] = -surfaces['tube_inner'] & +surfaces['impinging_plane'] & -surfaces['axial_end']
        regions['he_main'] = regions['he_main_outer'] & ~(regions['pin'] | regions['breeder'])

        regions['multiplier'] = -surfaces['multiplier_hexagon'] & +surfaces['multiplier_inner'] & +surfaces[
            'multiplier_front'] & -surfaces['multiplier_back']

        regions['he_purge_outer'] = -surfaces['unit_hexagon'] & +surfaces['fw_channel_back'] & -surfaces['axial_end']
        regions['he_purge'] = regions['he_purge_outer'] & ~(
                    regions['multiplier'] | regions['tube'] | regions['he_main'] | regions['pin'] | regions['breeder'])

        filled_regions = [
            regions['first_wall'],
            regions['he_channel'],
            regions['first_wall_channel'],
            regions['tube'],
            regions['breeder'],
            regions['pin'],
            regions['he_main'],
            regions['multiplier'],
            regions['he_purge'],
        ]

        regions['void'] = regions['simulation_volume'] & ~openmc.Union(filled_regions)

        # cells
        cells = {}

        cells['first_wall'] = openmc.Cell(fill=self.materials['tungsten'],
                                          region=regions['first_wall'],
                                          name='first_wall')
        cells['he_channel'] = openmc.Cell(fill=self.materials['He_channel'],
                                          region=regions['he_channel'],
                                          name='he_channel')
        cells['first_wall_channel'] = openmc.Cell(fill=self.materials['eurofer_first_wall_channel'],
                                                  region=regions['first_wall_channel'],
                                                  name='first_wall_channel')
        cells['tube'] = openmc.Cell(fill=self.materials['eurofer_pressure_tube'],
                                    region=regions['tube'],
                                    name='tube')
        cells['breeder'] = openmc.Cell(fill=self.materials['breeder_pebble_mix'],
                                       region=regions['breeder'],
                                       name='breeder')
        cells['pin'] = openmc.Cell(fill=self.materials['eurofer_pin'],
                                   region=regions['pin'],
                                   name='pin')
        cells['he_main'] = openmc.Cell(fill=self.materials['He_inner'],
                                       region=regions['he_main'],
                                       name='he_main')
        cells['multiplier'] = openmc.Cell(fill=self.materials['Be12Ti_outer'],
                                          region=regions['multiplier'],
                                          name='multiplier')
        cells['he_purge'] = openmc.Cell(fill=self.materials['He_outer'],
                                        region=regions['he_purge'],
                                        name='he_purge')
        cells['void'] = openmc.Cell(fill=None,
                                    region=regions['void'],
                                    name='void')

        # universe
        root_universe = openmc.Universe(name='root_universe',
                                        cells=list(cells.values()))
        geometry_obj = openmc.Geometry(root_universe)

        print("\nExporting geometry to geometry.xml...")
        geometry_obj.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")
        self.geometry = geometry_obj


    def _define_HP_CSG_geometry(self):
        print("\n\n\nDefining geometry with CSG...")

        csg_params = self.model_params['csg_geometry']
        csg_plane = csg_params['plane']
        csg_cylinder = csg_params['cylinder']
        csg_rectangle = csg_params['rectangle']
        csg_hexagon = csg_params['hexagon']

        surfaces = {}

        # --- Planes ---
        surfaces['tokamak_major_radius'] = openmc.ZPlane(z0=self.tokamak_major_radius - 5.0,
                                                         boundary_type='vacuum')
        surfaces['first_wall'] = openmc.ZPlane(z0=self.tokamak_radius)
        surfaces['fw_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                               + csg_plane['first_wall_thickness'])
        surfaces['impinging_plane'] = openmc.ZPlane(z0=self.tokamak_radius
                                                       + csg_plane['first_wall_thickness']
                                                       + csg_plane['fw_channel_thickness'])
        surfaces['pin_front'] = openmc.ZPlane(z0=self.tokamak_radius
                                                 + csg_plane['first_wall_thickness']
                                                 + csg_plane['fw_channel_thickness']
                                                 + csg_plane['impinging_to_pin'])
        surfaces['pin_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                + csg_plane['first_wall_thickness']
                                                + csg_plane['fw_channel_thickness']
                                                + csg_plane['impinging_to_pin']
                                                + csg_plane['pin_tip_thickness'])
        surfaces['inner_multiplier_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                             + csg_plane['first_wall_thickness']
                                                             + csg_plane['fw_channel_thickness']
                                                             + csg_plane['impinging_to_pin']
                                                             + csg_plane['pin_tip_thickness']
                                                             + csg_plane['length_inner_multiplier'])
        surfaces['outer_multiplier_front'] = openmc.ZPlane(z0=self.tokamak_radius
                                                              + csg_plane['first_wall_thickness']
                                                              + csg_plane['fw_channel_thickness']
                                                              + csg_plane['impinging_to_outer_multiplier'])
        surfaces['outer_multiplier_back'] = openmc.ZPlane(z0=self.tokamak_radius
                                                             + csg_plane['first_wall_thickness']
                                                             + csg_plane['fw_channel_thickness']
                                                             + csg_plane['impinging_to_outer_multiplier']
                                                             + csg_plane['length_outer_multiplier'])
        surfaces['axial_end'] = openmc.ZPlane(z0=self.tokamak_radius
                                                 + csg_plane['first_wall_thickness']
                                                 + csg_plane['fw_channel_thickness']
                                                 + csg_plane['impinging_to_pin']
                                                 + csg_plane['pin_tip_thickness']
                                                 + csg_plane['length_inner_multiplier']
                                                 + csg_plane['inner_multiplier_back_to_axial_end'],
                                              boundary_type='vacuum')
        surfaces['outer_multiplier_1_6'] = openmc.XPlane(x0=-csg_plane['purge_gap'] / 2)
        surfaces['outer_multiplier_7_12'] = openmc.XPlane(x0=csg_plane['purge_gap'] / 2)
        surfaces['outer_multiplier_2_9'] = openmc.Plane(a=1.0, b=np.sqrt(3.0), c=0.0, d=csg_plane['purge_gap'])
        surfaces['outer_multiplier_3_8'] = openmc.Plane(a=1.0, b=np.sqrt(3.0), c=0.0, d=-csg_plane['purge_gap'])
        surfaces['outer_multiplier_4_11'] = openmc.Plane(a=-1.0, b=np.sqrt(3.0), c=0.0, d=csg_plane['purge_gap'])
        surfaces['outer_multiplier_5_10'] = openmc.Plane(a=-1.0, b=np.sqrt(3.0), c=0.0, d=-csg_plane['purge_gap'])

        # --- Cylinders ---
        surfaces['outer_multiplier'] = openmc.ZCylinder(r=csg_cylinder['outer_multiplier'])
        surfaces['tube_outer'] = openmc.ZCylinder(r=csg_cylinder['tube_outer'])
        surfaces['tube_inner'] = openmc.ZCylinder(r=csg_cylinder['tube_inner'])
        surfaces['pin_outer'] = openmc.ZCylinder(r=csg_cylinder['pin_outer'])
        surfaces['breeder_outer'] = openmc.ZCylinder(r=csg_cylinder['breeder_outer'])
        surfaces['breeder_inner'] = openmc.ZCylinder(r=csg_cylinder['breeder_inner'])
        surfaces['multiplier_inner'] = openmc.ZCylinder(r=csg_cylinder['multiplier_inner'])
        surfaces['pin_inner'] = openmc.ZCylinder(r=csg_cylinder['pin_inner'])

        # --- Rectangles ---
        rectangles_z_coord = self.tokamak_radius + csg_plane['first_wall_thickness'] + csg_rectangle[
            'fw_back_to_channel_front'] + csg_rectangle['height'] / 2.0

        surfaces['fw_channel_center'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(csg_rectangle['pitch'] * 0, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_1'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(csg_rectangle['pitch'] * 1, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_2'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(csg_rectangle['pitch'] * 2, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_3'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(csg_rectangle['pitch'] * 3, rectangles_z_coord)
        )
        surfaces['fw_channel_pos_4'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(csg_rectangle['pitch'] * 4, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_1'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(-csg_rectangle['pitch'] * 1, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_2'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(-csg_rectangle['pitch'] * 2, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_3'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(-csg_rectangle['pitch'] * 3, rectangles_z_coord)
        )
        surfaces['fw_channel_neg_4'] = openmc.model.RectangularPrism(
            width=csg_rectangle['width'],
            height=csg_rectangle['height'],
            axis='x',
            origin=(-csg_rectangle['pitch'] * 4, rectangles_z_coord)
        )

        # --- Hexagons ---
        surfaces['unit_hexagon'] = openmc.model.HexagonalPrism(
            edge_length=csg_hexagon['unit'],
            origin=(0.0, 0.0),
            orientation='x',
            boundary_type='periodic'
        )

        # regions
        regions = {}

        regions['simulation_volume'] = -surfaces['unit_hexagon'] & +surfaces['tokamak_major_radius'] & -surfaces[
            'axial_end']

        regions['first_wall'] = -surfaces['unit_hexagon'] & +surfaces['first_wall'] & -surfaces['fw_back']

        regions['he_channel_center'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_center']
        regions['he_channel_pos_1'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_1']
        regions['he_channel_pos_2'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_2']
        regions['he_channel_pos_3'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_3']
        regions['he_channel_pos_4'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_pos_4']
        regions['he_channel_neg_1'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_1']
        regions['he_channel_neg_2'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_2']
        regions['he_channel_neg_3'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_3']
        regions['he_channel_neg_4'] = -surfaces['unit_hexagon'] & -surfaces['fw_channel_neg_4']
        regions['he_channel'] = regions['he_channel_center'] | regions['he_channel_pos_1'] | regions[
            'he_channel_pos_2'] | regions['he_channel_pos_3'] | regions['he_channel_pos_4'] | regions[
                                    'he_channel_neg_1'] | regions['he_channel_neg_2'] | regions['he_channel_neg_3'] | \
                                regions['he_channel_neg_4']

        regions['first_wall_channel_wo_he'] = -surfaces['unit_hexagon'] & +surfaces['fw_back'] & -surfaces[
            'impinging_plane']
        regions['first_wall_channel'] = regions['first_wall_channel_wo_he'] & ~(regions['he_channel'])

        regions['tube'] = -surfaces['tube_outer'] & +surfaces['tube_inner'] & +surfaces['impinging_plane'] & -surfaces[
            'axial_end']

        regions['inner_multiplier'] = -surfaces['breeder_inner'] & +surfaces['multiplier_inner'] & +surfaces[
            'pin_back'] & -surfaces['inner_multiplier_back']

        regions['breeder_outer'] = -surfaces['breeder_outer'] & +surfaces['multiplier_inner'] & +surfaces[
            'pin_back'] & -surfaces['axial_end']
        regions['breeder'] = regions['breeder_outer'] & ~regions['inner_multiplier']

        regions['pin_outer'] = -surfaces['pin_outer'] & +surfaces['pin_inner'] & +surfaces['pin_front'] & -surfaces[
            'axial_end']
        regions['pin'] = regions['pin_outer'] & ~(regions['breeder'] | regions['inner_multiplier'])

        regions['he_main_outer'] = -surfaces['tube_inner'] & +surfaces['impinging_plane'] & -surfaces['axial_end']
        regions['he_main'] = regions['he_main_outer'] & ~(
                    regions['pin'] | regions['breeder'] | regions['inner_multiplier'])

        regions['outer_multiplier_a'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & -surfaces[
            'outer_multiplier_1_6'] & +surfaces['outer_multiplier_2_9'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier_b'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & -surfaces[
            'outer_multiplier_3_8'] & +surfaces['outer_multiplier_4_11'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier_c'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & -surfaces[
            'outer_multiplier_5_10'] & -surfaces['outer_multiplier_1_6'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier_d'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & +surfaces[
            'outer_multiplier_7_12'] & -surfaces['outer_multiplier_3_8'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier_e'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & +surfaces[
            'outer_multiplier_2_9'] & -surfaces['outer_multiplier_5_10'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier_f'] = -surfaces['unit_hexagon'] & +surfaces['outer_multiplier'] & +surfaces[
            'outer_multiplier_4_11'] & +surfaces['outer_multiplier_7_12'] & +surfaces['outer_multiplier_front'] & - \
                                        surfaces['outer_multiplier_back']
        regions['outer_multiplier'] = regions['outer_multiplier_a'] | regions['outer_multiplier_b'] | regions[
            'outer_multiplier_c'] | regions['outer_multiplier_d'] | regions['outer_multiplier_e'] | regions[
                                          'outer_multiplier_f']

        regions['he_purge_outer'] = -surfaces['unit_hexagon'] & +surfaces['tube_outer'] & + surfaces[
            'impinging_plane'] & -surfaces['axial_end']
        regions['he_purge'] = regions['he_purge_outer'] & ~regions['outer_multiplier']

        filled_regions = [
            regions['first_wall'],
            regions['he_channel'],
            regions['first_wall_channel'],
            regions['tube'],
            regions['inner_multiplier'],
            regions['breeder'],
            regions['pin'],
            regions['he_main'],
            regions['outer_multiplier'],
            regions['he_purge'],
        ]

        regions['void'] = regions['simulation_volume'] & ~openmc.Union(filled_regions)

        # cells
        cells = {}

        cells['first_wall'] = openmc.Cell(fill=self.materials['tungsten'],
                                          region=regions['first_wall'],
                                          name='first_wall')
        cells['he_channel'] = openmc.Cell(fill=self.materials['He_channel'],
                                          region=regions['he_channel'],
                                          name='he_channel')
        cells['first_wall_channel'] = openmc.Cell(fill=self.materials['eurofer_first_wall_channel'],
                                                  region=regions['first_wall_channel'],
                                                  name='first_wall_channel')
        cells['tube'] = openmc.Cell(fill=self.materials['eurofer_pressure_tube'],
                                    region=regions['tube'],
                                    name='tube')
        cells['inner_multiplier'] = openmc.Cell(fill=self.materials['Be12Ti_inner'],
                                                region=regions['inner_multiplier'],
                                                name='inner_multiplier')
        cells['breeder'] = openmc.Cell(fill=self.materials['breeder_pebble_mix'],
                                       region=regions['breeder'],
                                       name='breeder')
        cells['pin'] = openmc.Cell(fill=self.materials['eurofer_pin'],
                                   region=regions['pin'],
                                   name='pin')
        cells['he_main'] = openmc.Cell(fill=self.materials['He_inner'],
                                       region=regions['he_main'],
                                       name='he_main')
        cells['outer_multiplier'] = openmc.Cell(fill=self.materials['Be12Ti_outer'],
                                                region=regions['outer_multiplier'],
                                                name='outer_multiplier')
        cells['he_purge'] = openmc.Cell(fill=self.materials['He_outer'],
                                        region=regions['he_purge'],
                                        name='he_purge')
        cells['void'] = openmc.Cell(fill=None,
                                    region=regions['void'],
                                    name='void')

        # universe
        root_universe = openmc.Universe(name='root_universe',
                                        cells=list(cells.values()))
        geometry_obj = openmc.Geometry(root_universe)

        print("\nExporting geometry to geometry.xml...")
        geometry_obj.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")
        self.geometry = geometry_obj


    def _define_LP_DAGMC_geometry(self):
        print("\n\n\nDefining geometry with DAGMC from .h5m file...")

        # =============================================================================================================
        # One-sixth cell
        # triangle_1_plane = openmc.Plane(
        #     a=1.0 / np.sqrt(3),
        #     b=-1.0,
        #     c=0.0,
        #     d=0.0,
        #     name='triangle_1_plane',
        #     boundary_type='reflective'
        # )
        #
        # triangle_2_plane = openmc.Plane(
        #     a=1.0 / np.sqrt(3),
        #     b=1.0,
        #     c=0.0,
        #     d=0.0,
        #     name='triangle_2_plane',
        #     boundary_type='reflective'
        # )
        #
        # triangle_3_plane = openmc.XPlane(
        #     x0=self.model_params['characteristic_length'] * (np.sqrt(3.0) / 2.0),
        #     name='triangle_3_plane',
        #     boundary_type='reflective'
        # )
        # z_min_plane = openmc.ZPlane(z0=self.tokamak_major_radius - 5, boundary_type='vacuum')
        # z_max_plane = openmc.ZPlane(z0=self.model_params['axial_end'], boundary_type='vacuum')
        #
        # final_region = +triangle_1_plane & +triangle_2_plane & -triangle_3_plane & +z_min_plane & -z_max_plane
        # =============================================================================================================

        # =============================================================================================================
        # Unit cell
        # HexagonalPrism은 z축 axis만 지원
        hex_prism = openmc.model.HexagonalPrism(
            edge_length=(self.model_params['characteristic_length']),
            origin=(0.0, 0.0),
            orientation='y',
            boundary_type='periodic'
        )

        z_min_plane = openmc.ZPlane(z0=self.tokamak_major_radius - 5, boundary_type='vacuum')
        z_max_plane = openmc.ZPlane(z0=self.model_params['axial_end'], boundary_type='vacuum')

        final_region = -hex_prism & +z_min_plane & -z_max_plane
        # =============================================================================================================

        # =============================================================================================================
        # Wide cell
        # hex_prism = openmc.model.HexagonalPrism(
        #     edge_length=(self.config['geometry']['characteristic_length']),
        #     origin=(0.0, 0.0),
        #     orientation='x',
        #     boundary_type='periodic'
        # )
        #
        # z_min_plane = openmc.ZPlane(z0=self.tokamak_major_radius - 5, boundary_type='vacuum')
        # z_max_plane = openmc.ZPlane(z0=self.model_params['axial_end'], boundary_type='vacuum')
        #
        # final_region = -hex_prism & +z_min_plane & -z_max_plane
        # =============================================================================================================

        # DAGMC를 위해 형상 불러오기
        dagmc_filepath = self.model_params['dagmc_path']

        # auto_geom_ids를 활성화해야 OpenMC CSG ID랑 충돌하지 않는 것 같음.
        dag_universe = openmc.DAGMCUniverse(filename=dagmc_filepath, auto_geom_ids=True)

        root_cell = openmc.Cell(
            name='root_cell',
            region=final_region,
            fill=dag_universe,
            cell_id=999
        )

        # 최종 형상 조립
        root_universe = openmc.Universe(cells=[root_cell])
        geometry_obj = openmc.Geometry(root_universe)

        print("\nExporting geometry to geometry.xml...")
        geometry_obj.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")
        self.geometry = geometry_obj


    def _define_HP_DAGMC_geometry(self):
        print("\n\n\nDefining geometry with DAGMC from .h5m file...")

        hex_prism = openmc.model.HexagonalPrism(
            edge_length=(self.model_params['characteristic_length']),
            origin=(0.0, 0.0),
            orientation='x',
            boundary_type='periodic'
        )

        z_min_plane = openmc.ZPlane(z0=self.tokamak_major_radius - 5, boundary_type='vacuum')
        z_max_plane = openmc.ZPlane(z0=self.model_params['axial_end'], boundary_type='vacuum')

        final_region = -hex_prism & +z_min_plane & -z_max_plane

        # DAGMC를 위해 형상 불러오기
        dagmc_filepath = self.model_params['dagmc_path']

        # auto_geom_ids를 활성화해야 OpenMC CSG ID랑 충돌하지 않는 것 같음.
        dag_universe = openmc.DAGMCUniverse(filename=dagmc_filepath, auto_geom_ids=True)

        root_cell = openmc.Cell(
            name='root_cell',
            region=final_region,
            fill=dag_universe,
            cell_id=999
        )

        # 최종 형상 조립
        root_universe = openmc.Universe(cells=[root_cell])
        geometry_obj = openmc.Geometry(root_universe)

        print("\nExporting geometry to geometry.xml...")
        geometry_obj.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")
        self.geometry = geometry_obj


    def define_settings(self):
        """Configures the simulation settings."""
        try:
            print("\n\n\nDefining settings...\n")
            self.settings = openmc.Settings()

            # Set to 'fixed source' mode as the neutron source does not change in fusion simulations.
            self.settings.run_mode = 'fixed source'

            sim_config = self.config['simulation']
            self.settings.particles = sim_config['particles']
            # The number of iterations to run. Doubling batches reduces uncertainty by a factor of sqrt(2).
            self.settings.batches = sim_config['batches']
            # Maximum number of particle tracks to include in tracks.h5.
            self.settings.max_tracks = sim_config['max_tracks']

            # FENDL library lacks photon data for some isotopes, so disable photon transport.
            if 'fendl' in self.cross_section_path.lower():
                print("FENDL library detected. Disabling photon_transport.\n")
                self.settings.photon_transport = False
            else:  # ENDF/JEFF have complete photon libraries.
                print("ENDF/JEFF library detected. Enabling photon_transport.\n")
                self.settings.photon_transport = True

            plasma_source, source_name = None, ""

            # Set one of the four source types based on user selection.
            if isinstance(self.source_choice, int):
                # NOTE: Units may need to be checked.
                if self.source_choice == 1:
                    source_name = "Tokamak plasma source"
                    print(f"Selected source: {source_name}")
                    plasma_source = tokamak_source(
                        elongation=1.557, ion_density_centre=1.09e20,  # [m^-3]
                        ion_density_peaking_factor=1, ion_density_pedestal=1.09e20,  # [m^-3]
                        ion_density_separatrix=3e19,  # [m^-3]
                        ion_temperature_centre=45900,  # [eV]
                        ion_temperature_peaking_factor=8.06, ion_temperature_pedestal=6090,  # [eV]
                        ion_temperature_separatrix=100,  # [eV]
                        major_radius=self.tokamak_major_radius,  # [cm]
                        minor_radius=self.tokamak_minor_radius,  # [cm]
                        pedestal_radius=0.8 * self.tokamak_minor_radius,  # [cm]
                        mode="H", shafranov_factor=0.3 * self.tokamak_minor_radius,  # [cm]
                        triangularity=0.270, ion_temperature_beta=6,
                        sample_size=1000, angles=(0.0, 2 * np.pi),
                    )
                elif self.source_choice == 2:
                    source_name = "Simplified ring source"
                    print(f"Selected source: {source_name}")
                    plasma_source = fusion_ring_source(
                        fuel={"D": 0.09, "T": 0.91}, temperature=20000.0,  # [eV]
                        radius=self.tokamak_major_radius, angles=(0.0, 2 * np.pi),  # [cm]
                        z_placement=0.0
                    )
                elif self.source_choice == 3:
                    source_name = "Single point source"
                    print(f"Selected source: {source_name}")
                    plasma_source = fusion_point_source(
                        fuel={"D": 0.09, "T": 0.91}, temperature=20000.0,  # [eV]
                        coordinate=(self.tokamak_major_radius, 0.0, 0.0)  # [cm]
                    )
            elif isinstance(self.source_choice, tuple) and self.source_choice[0] == 4:
                source_name = "Custom Source"
                print(f"Selected source: {source_name}")
                options = self.source_choice[1]
                custom_source = openmc.IndependentSource()
                energy_options, angle_options, space_options = options["energy"], options["angle"], options["space"]
                energy_type, params = energy_options["type"], energy_options["params"]

                if energy_type == "Discrete": custom_source.energy = openmc.stats.Discrete([params["energy"]], [params["prob"]])
                elif energy_type == "Watt (fission) distribution": custom_source.energy = openmc.stats.Watt(a=params["a"], b=params["b"])
                elif energy_type == "Muir (normal) distribution": custom_source.energy = openmc.stats.muir(e0=params["e0"], m_rat=params["m_rat"], kt=params["kt"])
                elif energy_type == "Maxwell distribution": custom_source.energy = openmc.stats.Maxwell(theta=params["theta"])

                if angle_options["type"] == "Isotropic": custom_source.angle = openmc.stats.Isotropic()
                elif angle_options["type"] == "Monodirectional": custom_source.angle = openmc.stats.Monodirectional(reference_uvw=angle_options["uvw"])

                if space_options["type"] == "Point": custom_source.space = openmc.stats.Point(space_options["coords"])
                elif space_options["type"] == "Unit cross-section":
                    source_positions = create_unit_geometry_source_points(
                        n_points=sim_config['n_source_points'], z_coord=space_options["z_coord"],
                        characteristic_length=space_options["characteristic_length"], model_type=self.model_type
                    )
                    custom_source.space = openmc.stats.PointCloud(source_positions)

                plasma_source = custom_source

            self.settings.source = plasma_source
            print(f"\nSource '{source_name}' assigned to settings.")
            # Use nearest temperature data for fluids if the exact match is not in cross_sections.xml.
            self.settings.temperature = {'method': 'nearest', 'default': 600.0, 'tolerance': 1000.0}
            # Configure output files: summary.h5 and tallies.out in addition to statepoint.h5.
            self.settings.output = {'summary': True, 'tallies': False, 'path': 'results'}

            print("\nExporting settings to settings.xml...\n")
            self.settings.export_to_xml()
            print("settings.xml exported successfully.\n")
            print("========================================================================\n")

        except Exception as e:
            print(f"\n\nError in define_settings method: {e}\n")
            raise

    def define_tallies(self):
        """
        Defines what to measure in the simulation.
        Tallies are broadly divided into two types:
            1. Global tallies that measure average values over the entire simulation.
            2. Local tallies that use a 2D or 3D mesh to measure values at specific locations.
        """
        try:
            print("\n\n\nDefining tallies...")
            self.tallies = []

            # Cell 별로 지정된 재료를 filter로 사용
            eurofer_pressure_tube_object = self.materials['eurofer_pressure_tube']
            eurofer_pin_object = self.materials['eurofer_pin']
            eurofer_first_wall_channel_object = self.materials['eurofer_first_wall_channel']
            he_inner_object = self.materials['He_inner']
            he_outer_object = self.materials['He_outer']
            breeder_object = self.materials['breeder_pebble_mix']
            be12ti_inner_object = self.materials['Be12Ti_inner']
            be12ti_outer_object = self.materials['Be12Ti_outer']
            tungsten_object = self.materials['tungsten']

            solid_materials_to_tally = [
                eurofer_pressure_tube_object,
                eurofer_pin_object,
                eurofer_first_wall_channel_object,
                breeder_object,
                be12ti_inner_object,
                be12ti_outer_object,
                tungsten_object,
            ]
            solid_material_filter = openmc.MaterialFilter(solid_materials_to_tally, filter_id=11)
            breeder_filter = openmc.MaterialFilter([breeder_object], filter_id=21)
            be12ti_inner_filter = openmc.MaterialFilter([be12ti_inner_object], filter_id=31)
            be12ti_outer_filter = openmc.MaterialFilter([be12ti_outer_object], filter_id=32)
            be12ti_filter = openmc.MaterialFilter([be12ti_outer_object, be12ti_inner_object], filter_id=33)
            eurofer_filter = openmc.MaterialFilter([eurofer_pressure_tube_object, eurofer_pin_object, eurofer_first_wall_channel_object], filter_id=41)
            tungsten_filter = openmc.MaterialFilter([tungsten_object], filter_id=51)

            # Energy filter.
            energy_filter = openmc.EnergyFilter.from_group_structure('CCFE-709')
            energy_filter.filter_id = 61
            # energy_bins = np.logspace(-3, 7.18, 1001)  # 0.001 eV ~ 15.1 MeV 범위를 1000개로 쪼개기
            # energy_filter = openmc.EnergyFilter(energy_bins, filter_id=61)

            # Particle filter
            neutron_filter = openmc.ParticleFilter(['neutron'], filter_id=71)
            # photon_filter = openmc.ParticleFilter(['photon'], filter_id=72)
            particle_filter = openmc.ParticleFilter(['neutron', 'photon'], filter_id=73)


            '''Global Tallies Start Here'''
            # Define Scores: What to measure?
            # Ref: https://docs.openmc.org/en/stable/usersguide/tallies.html (OpenMC built-in tallies)
            # Ref: https://www.oecd-nea.org/dbdata/data/manual-endf/endf102_MT.pdf (ENDF MT list)
            # Define Filters: When/where/which particles to measure?

            # Set user-friendly names
            tally_tbr = openmc.Tally(name='tbr', tally_id=98)
            tally_tbr.scores = ['H3-production']  # Also '(n,Xt)' where X is a wildcard; MT number: 205.
            tally_tbr.nuclides = ['Li6', 'Li7']  # Calculate for Li isotopes.
            tally_tbr.filters = [breeder_filter]
            self.tallies.append(tally_tbr)

            tally_multiplying = openmc.Tally(name='multiplication', tally_id=99)
            tally_multiplying.scores = ['(n,2n)']
            tally_multiplying.nuclides = ['Be9']
            tally_multiplying.filters = [be12ti_filter]
            self.tallies.append(tally_multiplying)

            tally_global_structure = openmc.Tally(name='global_structure', tally_id=94)
            tally_global_structure.scores = ['flux', 'absorption', 'elastic', 'heating']
            tally_global_structure.filters = [eurofer_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_structure)

            tally_global_armor = openmc.Tally(name='global_armor', tally_id=95)
            tally_global_armor.scores = ['flux', 'absorption', 'elastic', 'heating']
            tally_global_armor.filters = [tungsten_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_armor)

            tally_global_multiplier = openmc.Tally(name='global_multiplier', tally_id=96)
            tally_global_multiplier.scores = ['flux', 'absorption', 'elastic', 'heating', '(n,2n)']
            tally_global_multiplier.filters = [be12ti_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_multiplier)

            tally_global_breeder = openmc.Tally(name='global_breeder', tally_id=97)
            tally_global_breeder.scores = ['flux', 'absorption', 'elastic', 'heating', 'H3-production']
            tally_global_breeder.filters = [breeder_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_breeder)

            '''Local Tallies Start Here'''
            # Create a mesh for calculating local tallies.
            # RegularMesh: Cartesian grid, CylindricalMesh: Cylindrical grid, SphericalMesh: Spherical grid.
            # CylindricalMesh currently only supports the z-axis as the axis of revolution.
            # Use OpenMC's structured mesh (for tallies where only axial distribution is of interest).

            # Create a cylindrical mesh that encloses the geometry.
            mesh_cylindrical_config = self.model_params['mesh_cylindrical']

            mesh_cylindrical = openmc.CylindricalMesh(name='cylindrical_mesh',
                                                      r_grid=np.linspace(
                                                          mesh_cylindrical_config['r_min'],
                                                          mesh_cylindrical_config['r_max'],
                                                          mesh_cylindrical_config['division_r']),
                                                      # phi_grid=np.linspace(
                                                      #     mesh_cylindrical_config['phi_min'],
                                                      #     mesh_cylindrical_config['phi_max'],
                                                      #     mesh_cylindrical_config['division_phi']),
                                                      z_grid=np.linspace(
                                                          mesh_cylindrical_config['z_min'],
                                                          mesh_cylindrical_config['z_max'],
                                                          mesh_cylindrical_config['division_z']),
                                                      origin=(0.0, 0.0, 0.0))

            mesh_cylindrical_filter = openmc.MeshFilter(mesh_cylindrical, filter_id=100)


            tally_local_heating_breeder = openmc.Tally(name='local_heating_breeder', tally_id=101)
            tally_local_heating_breeder.scores = ['heating']
            tally_local_heating_breeder.filters = [mesh_cylindrical_filter, breeder_filter, particle_filter]

            tally_local_heating_multiplier = openmc.Tally(name='local_heating_multiplier', tally_id=102)
            tally_local_heating_multiplier.scores = ['heating']
            tally_local_heating_multiplier.filters = [mesh_cylindrical_filter, be12ti_outer_filter, particle_filter]

            tally_local_heating_structure = openmc.Tally(name='local_heating_structure', tally_id=103)
            tally_local_heating_structure.scores = ['heating']
            tally_local_heating_structure.filters = [mesh_cylindrical_filter, eurofer_filter, particle_filter]

            tally_local_heating_inner_multiplier = openmc.Tally(name='local_heating_inner_multiplier', tally_id=104)
            tally_local_heating_inner_multiplier.scores = ['heating']
            tally_local_heating_inner_multiplier.filters = [mesh_cylindrical_filter, be12ti_inner_filter, particle_filter]

            tally_local_flux_breeder = openmc.Tally(name='local_flux_breeder', tally_id=201)
            tally_local_flux_breeder.scores = ['flux']
            tally_local_flux_breeder.filters = [mesh_cylindrical_filter, breeder_filter, neutron_filter]

            tally_local_flux_multiplier = openmc.Tally(name='local_flux_multiplier', tally_id=202)
            tally_local_flux_multiplier.scores = ['flux']
            tally_local_flux_multiplier.filters = [mesh_cylindrical_filter, be12ti_outer_filter, neutron_filter]

            tally_local_flux_structure = openmc.Tally(name='local_flux_structure', tally_id=203)
            tally_local_flux_structure.scores = ['flux']
            tally_local_flux_structure.filters = [mesh_cylindrical_filter, eurofer_filter, neutron_filter]

            tally_local_flux_inner_multiplier = openmc.Tally(name='local_flux_inner_multiplier', tally_id=204)
            tally_local_flux_inner_multiplier.scores = ['flux']
            tally_local_flux_inner_multiplier.filters = [mesh_cylindrical_filter, be12ti_inner_filter, neutron_filter]


            local_tallies_list = [
                tally_local_heating_breeder,
                tally_local_heating_multiplier,
                tally_local_heating_structure,
                tally_local_heating_inner_multiplier,
                tally_local_flux_breeder,
                tally_local_flux_multiplier,
                tally_local_flux_structure,
                tally_local_flux_inner_multiplier
            ]

            self.tallies.extend(local_tallies_list)

            # # Tally를 mesh 하나의 부피로 나눌 것인가?
            # for tally in local_tallies_list:
            #     tally.volume_normalization = True

            tallies_obj = openmc.Tallies(self.tallies)
            print("\nExporting tallies to tallies.xml...\n")
            tallies_obj.export_to_xml()
            print("tallies.xml exported successfully.\n")
            print("========================================================================\n")

        except Exception as e:
            print(f"\n\nError in define_tallies method: {e}\n")
            raise

    def generate_geometry_2D_plots(self, plots_folder='plots'):
        """Generates 2D visualizations of the geometry before running the simulation."""
        try:
            print("\n\n\n--- Generating 2D material-colored geometry plots with axes using openmc.Plot() ---")
            material_colors = {
                self.materials['eurofer_pressure_tube']: (128, 128, 128),  # Gray
                self.materials['eurofer_pin']: (128, 128, 128),  # Gray
                self.materials['eurofer_first_wall_channel']: (128, 128, 128),  # Gray
                self.materials['Be12Ti_inner']: (0, 255, 0),      # Green
                self.materials['Be12Ti_outer']: (0, 255, 0),      # Green
                self.materials['breeder_pebble_mix']: (255, 0, 0),    # Red
                self.materials['He_inner']: (0, 0, 255),        # Blue
                self.materials['He_outer']: (0, 0, 255),        # Blue
                self.materials['He_channel']: (0, 0, 255),      # Blue
                self.materials['tungsten']: (128, 0, 128),    # Purple
            }

            # Create Plot objects.
            plot_xy = openmc.Plot()
            plot_xy.filename = os.path.join(plots_folder, 'geometry_by_material_xy')
            plot_xy.width = (15.0, 15.0)
            plot_xy.pixels = (1500, 1500)
            plot_xy.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'] - 10)
            plot_xy.basis = 'xy'
            plot_xy.color_by = 'material'
            plot_xy.colors = material_colors

            plot_yz = openmc.Plot()
            plot_yz.filename = os.path.join(plots_folder, 'geometry_by_material_yz')
            plot_yz.width = (15.0, 60.0)
            plot_yz.pixels = (1500, 6000)
            plot_yz.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'])
            plot_yz.basis = 'yz'
            plot_yz.color_by = 'material'
            plot_yz.colors = material_colors

            plot_zx = openmc.Plot()
            plot_zx.filename = os.path.join(plots_folder, 'geometry_by_material_zx')
            plot_zx.width = (15.0, 60.0)
            plot_zx.pixels = (1500, 6000)
            plot_zx.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'])
            plot_zx.basis = 'xz'
            plot_zx.color_by = 'material'
            plot_zx.colors = material_colors

            # 플롯 생성
            if self.model_type == 'LP':
                plots = openmc.Plots([plot_xy, plot_zx])
            elif self.model_type == 'HP':
                plots = openmc.Plots([plot_xy, plot_yz, plot_zx])

            plots.export_to_xml()  # Create plots.xml.

            # Requires geometry.xml and materials.xml to be already generated.
            openmc.plot_geometry()

            print(f" -> Material-colored plots saved.\n")

        except Exception as e:
            print(f"\n\nError in generate_geometry_2d_plots method: {e}\n")
            raise

    def preview_source_distribution(self, plots_folder='plots', n_samples=500):
        """Visualizes the source distribution."""
        try:
            if self.settings is None or self.settings.source is None:
                print("Error: Settings or source is not defined yet.")
                return

            print("\n\n\n--- Previewing source distribution (before simulation) ---")
            # Create an openmc.Model object to pass to the visualization functions.
            # A model.xml file is created, which lacks tally info, so it must be deleted before the main run.
            model = openmc.Model(
                geometry=self.geometry, materials=self.all_materials_collection,
                settings=self.settings,
            )

            # Spatial Distribution (Position).
            print("Generating source position preview...")
            plot_pos = plot_source_position(this=model, n_samples=n_samples)
            plot_pos.update_layout(title='Source Particle Starting Position', autosize=True, showlegend=True)
            plot_pos.write_html(os.path.join(plots_folder, 'source_preview_position.html'))
            print("-> Source position preview saved.\n")

            # Energy Distribution.
            print("Generating source energy preview...")
            plot_en = plot_source_energy(this=model, n_samples=n_samples)
            plot_en.update_layout(title='Source Particle Energy Distribution', xaxis_title='Energy [MeV]', yaxis_title='Probability [-]', autosize=True, showlegend=True)
            plot_en.write_html(os.path.join(plots_folder, 'source_preview_energy.html'))
            print("-> Source energy preview saved.\n")

            # Directional Distribution.
            print("Generating source direction preview...")
            plot_dir = plot_source_direction(this=model, n_samples=n_samples)
            plot_dir.update_layout(title='Source Particle Moving Direction', autosize=True, showlegend=True)
            plot_dir.write_html(os.path.join(plots_folder, 'source_preview_direction.html'))
            print("-> Source direction preview saved.\n")
            print("\n--- Source preview generation finished. ---")

        except Exception as e:
            print(f"\n\nError in create_source_previews method: {e}\n")
            raise

    def run_setup_pipeline(self, status_window, tasks):
        """Executes all pre-simulation setup tasks."""
        try:
            pbar_tasks = tasks[3:-1]
            # ANSI color codes.
            green, reset = '\033[92m', '\033[0m'
            # The default tqdm format is {l_bar}{bar}{r_bar}.
            custom_bar_format = f"{green}{{l_bar}}{{bar}}|{{r_bar}}{reset}"

            print("\n--- Running pre-simulation setup pipeline ---")
            with tqdm(total=len(pbar_tasks), desc="Preparation", file=sys.stdout, bar_format=custom_bar_format) as pbar:

                # This helper function now updates the status to "Running..." before starting the task.
                def run_step(description, function, *args, **kwargs):
                    """Helper function to run a setup step and update progress."""
                    pbar.set_description(description.replace("\n", ""))
                    status_window.update_task_status(description, "Running...", "blue")

                    function(*args, **kwargs)  # Execute the long-running task.

                    status_window.update_task_status(description, "OK! ✓", "green")
                    pbar.update(1)

                run_step("Materials Definition", self.define_materials)
                run_step("Geometry Definition", self.define_geometry)
                run_step("Settings Definition", self.define_settings)
                run_step("Tallies Definition", self.define_tallies)
                run_step("Geometry Plots", self.generate_geometry_2D_plots, plots_folder='plots')
                run_step("Source Previews", self.preview_source_distribution, plots_folder='plots')

        except Exception as e:
            print(f"\n\nError in run_setup_pipeline method: {e}\n")
            raise

    def prompt_and_run_simulation(self, status_window):
        """Starts the simulation."""
        try:
            # Delete the temporary model.xml file created during the source preview.
            if os.path.exists('model.xml'):
                print("\n--- Removing temporary 'model.xml' before main run ---\n\n")
                os.remove('model.xml')

            status_window.update_task_status("Main OpenMC Simulation", "Running...", "blue")
            # Start the simulation!
            openmc.run(tracks=False, threads=self.config['simulation']['threads'])
            status_window.update_task_status("Main OpenMC Simulation", "OK! ✓", "green")
            status_window.complete("\nAll simulation tasks finished!\nRefer to /results folder.")

        except Exception as e:
            print(f"\n\nError in prompt_and_run_simulation method: {e}\n")
            raise