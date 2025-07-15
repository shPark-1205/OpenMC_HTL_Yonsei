# Simulation_Builder_DAGMC.py

import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from tqdm import tqdm
import neutronics_material_maker as nmm
from openmc_plasma_source import tokamak_source, fusion_ring_source, fusion_point_source
from openmc_source_plotter import plot_source_energy, plot_source_position, plot_source_direction
from dagmc_geometry_slice_plotter import plot_axis_slice

# Periodic 육각기둥 내부 단면에 무작위 위치의 중성자 소스 분포
def create_hexagonal_source_points(n_points, x_coord, pitch):

    points =[]

    # 단위 육각형 한 변의 길이
    s = pitch * (2.0 / 3.0)

    # 육각형을 포함하는 가장 작은 사각형 생성
    x = x_coord
    abs_y_max = s
    abs_z_max = s * (np.sqrt(3.0) / 2.0)

    print(f"\nGenerating {n_points} source points on a hexagonal face at x={x_coord}...")

    while len(points) < n_points:

        # 사각형 안에서 무작위 샘플링
        y = np.random.uniform(-abs_y_max, abs_y_max)
        z = np.random.uniform(-abs_z_max, abs_z_max)

        is_inside = (abs(y) <= s) and \
                    (abs(np.sqrt(3.0) * y + z) <= s * np.sqrt(3)) and \
                    (abs(-np.sqrt(3.0) * y + z) <= s * np.sqrt(3))

        if is_inside:
            points.append((x, y, z))

    print(f"\nGenerated {len(points)} points for hexagonal source.")
    return points


class NuclearFusion:
    def __init__(self, source_choice, config):
        try:

            # config 객체를 클래스 속성으로 저장
            self.config = config

            # 해석에 사용할 재료를 먼저 생성해야 함. 재료가 nmm에 정의되어 있는 이름대로 작성
            self.material_configs = [
                {'name': 'eurofer', 'kwargs': {}},
                {'name': 'He', 'kwargs': {
                    'temperature': self.config['materials']['he']['temperature'],
                    'pressure': self.config['materials']['he']['pressure']
                }},  # 유체는 온도/압력 정의 필수
                {'name': 'Li4SiO4', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'packing_fraction': self.config['materials']['breeder']['packing_fraction'],
                }},
                {'name': 'Li2TiO3', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'packing_fraction': self.config['materials']['breeder']['packing_fraction'],
                }},
                # 증식재의 Li6 enrichment와 packing factor 정의 가능
                # {'name': 'Be12Ti', 'kwargs': {}},
                {'name': 'tungsten', 'kwargs': {}},
            ]

            # 사용할 모든 재료를 openmc.Materia 객체로 저장할 딕셔너리
            self.materials = {}

            # 사용할 모든 재료를 하나로 묶어 놓는 바구니
            self.all_materials_collection = None

            # 해석에 사용할 형상을 묶어 놓는 바구니
            self.geometry = None

            # 토카막 기본적 형상 정보 [cm]
            self.tokamak_major_radius = self.config['geometry']['tokamak_major_radius']
            self.tokamak_minor_radius = self.config['geometry']['tokamak_minor_radius']

            # 토카막 중심부터 OB 벽까지의 거리
            self.tokamak_radius = self.tokamak_major_radius + self.tokamak_minor_radius

            # 해석에 사용할 설정을 묶어 놓는 바구니
            self.settings = None

            # TokamakSource, FusionRingSource, FusionPointSource 중 어떤 source를 선택할 지 저장
            self.source_choice = source_choice

            # 해석 후 후처리할 것들을 묶어 놓는 바구니
            self.tallies = []

            # Docker image 안에 OpenMC 공식 library(ENDF/B-VII.1)를 넣어 놨음.
            # https://openmc.org/official-data-libraries/
            openmc.config['cross_sections'] = r'/app/data/endfb-vii.1-hdf5/cross_sections.xml'

        except Exception as e:
            print(f"\n\nError during NuclearFusion class initialization: {e}\n")
            raise

    # 해석에 사용할 재료 정의
    def define_materials(self):
        try:
            # neutronics_material_maker(nmm) 모듈을 통해 미리 정의된 물성을 사용
            print("\n\n\nDefining materials using neutronics_material_maker...")

            self.materials = {}

            # 재료 담을 리스트
            openmc_materials_list = []

            # 증식재 혼합물 만들기 위해 임시로 저장하는 곳
            nmm_materials_temp = {}

            # 증배재만 Thermal scattering data 추가하기 위해 따로 설정
            Be12Ti_mat = openmc.Material(name='Be12Ti')
            Be12Ti_mat.add_elements_from_formula('Be12Ti')
            Be12Ti_mat.set_density('g/cm3', 2.27)
            Be12Ti_mat.temperature = 400

            # 베릴륨은 감속재로 작용하기 때문에 thermal scattering data 설정해야 함.
            Be12Ti_mat.add_s_alpha_beta('c_Be') # cross_sections.xml 파일에 이름 있음.
            self.materials['Be12Ti'] = Be12Ti_mat
            openmc_materials_list.append(Be12Ti_mat)

            # __init__ 메소드에서 만들었던 재료 목록 가져오기
            for config in self.material_configs:
                mat_name = config['name']
                mat_kwargs = config['kwargs']

                print(f"\nLoading base material: {mat_name} with parameters {mat_kwargs}...")

                # nmm 모듈에서 재료 목록 불러오기
                nmm_mat = nmm.Material.from_library(mat_name, **mat_kwargs)
                
                # nmm 물성을 openmc 물성으로 저장
                openmc_mat = nmm_mat.openmc_material

                # OpenMC Material 객체와 nmm Material 객체를 각각 저장
                self.materials[mat_name] = openmc_mat
                nmm_materials_temp[mat_name] = nmm_mat  # 혼합을 위해 nmm 객체 저장
                openmc_materials_list.append(openmc_mat)
                print(f"Material {mat_name} loaded (OpenMC ID: {openmc_mat.id}).")

            # 위에서 생성한 기본 재료들을 사용하여 혼합물 생성
            print("\nCreating mixed material (Li4SiO4 + Li2TiO3)...")

            # 혼합할 nmm.Material 객체 불러오기
            mat1 = nmm_materials_temp['Li4SiO4']
            mat2 = nmm_materials_temp['Li2TiO3']

            # 65% Li4SiO4와 35% Be12Ti를 부피비(vo)로 혼합
            mixed_breeder_material = nmm.Material.from_mixture(
                materials=[mat1, mat2],
                fracs=[self.config['materials']['breeder']['mixture_Li4SiO4'], self.config['materials']['breeder']['mixture_Li2TiO3']],
                percent_type='vo',
                name='breeder_pebble_mix'  # 새로운 재료의 이름 지정
            )

            # 새로 만든 혼합물을 재료 딕셔너리와 리스트에 추가
            mixed_openmc_material = mixed_breeder_material.openmc_material
            self.materials['breeder_pebble_mix'] = mixed_openmc_material
            openmc_materials_list.append(mixed_openmc_material)

            print(f"Mixed material 'breeder_pebble_mix' created (OpenMC ID: {mixed_openmc_material.id}).")

            # 모든 OpenMC Material 객체들을 openmc.Materials 컬렉션으로 묶음
            self.all_materials_collection = openmc.Materials(openmc_materials_list)

            # 정의한 재료를 OpenMC가 필요로 하는 xml 파일로 저장
            print("\nExporting materials to materials.xml...\n")
            self.all_materials_collection.export_to_xml()
            print("materials.xml exported successfully.\n")
            # print("========================================================================\n")
            time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_materials method: {e}\n")
            raise

    # 해석에 사용할 형상 정의 (surface, region, cell 헬퍼 함수를 참조)
    def define_geometry(self):
        # https://docs.openmc.org/en/v0.15.2/usersguide/geometry.html
        """
        OpenMC 해석을 하기 위해서는 아래와 같은 흐름으로 진행
        Option 1. OpenMC 안에서 형상 정의 (https://docs.openmc.org/en/stable/pythonapi/base.html#building-geometry)
            -> OpenMC 자체에서 plane, cylinder, sphere, cone, torus 등등 기본적인 형상 함수 제공
            -> CSG를 활용해서 형상을 정의하는 과정
            -> OpenMC 자체의 기능을 사용하기 때문에 정의한 형상을 OpenMC 용으로 변환할 필요 X
            -> 다만 복잡한 형상을 반영하기 어려움
        Option 2. OpenMC 외부에서 형상 정의
            -> 흔히 .step 파일로 CATIA 등에서 만든 해석 형상을 저장
            -> CadQuery와 같은 모듈을 통해서 만드는 방법도 있음.
            -> .step 파일이나 CadQuery 파일을 DAGMC(.h5m) 파일로 변환해야 함.
            -> cad_to_dag, gmsh와 같은 모듈로 .step 파일을 .h5m 파일로 변환 가능
            -> 이후 DAGMC를 활용해서 OpenMC로 .h5m 파일 전송
        """

        print("\n\n\nDefining geometry with DAGMC from .h5m file...")

        # CAD 형상을 충분히 감쌀 수 있는 크기의 경계 설정
        z_height = self.config['bounding']['z_height']
        y_width = self.config['bounding']['y_width']
        x_min = self.config['bounding']['x_min']
        x_max = self.config['bounding']['x_max']

        # 최외곽 경계 사각기둥 생성
        prism = openmc.model.RectangularPrism(
            width=y_width,
            height=z_height,
            axis='x',
            origin=(0.0, 0.0),
            boundary_type='periodic'
        )

        # 사각기둥 끝을 닫을 x 평면
        x_min_plane = openmc.XPlane(x0=x_min, boundary_type='vacuum')
        x_max_plane = openmc.XPlane(x0=x_max, boundary_type='vacuum')

        # 최외곽 닫힌 육각 기둥
        final_region = -prism & +x_min_plane & -x_max_plane

        # DAGMC를 위해 형상 불러오기
        h5m_path = self.config['geometry']['h5m_path']
        dag_universe = openmc.DAGMCUniverse(filename=h5m_path, auto_geom_ids=True)

        root_cell = openmc.Cell(
            name='root_cell',
            region=final_region,
            fill=dag_universe
        )

        # 최종 형상 조립
        root_universe = openmc.Universe(cells=[root_cell])
        geometry_obj = openmc.Geometry(root_universe)

        print("\nExporting geometry to geometry.xml...")
        geometry_obj.export_to_xml()
        print("\ngeometry.xml exported successfully.\n")
        self.geometry = geometry_obj

    # 해석 시작 전에 가능한 시각화
    def generate_geometry_2D_plots(self, plots_folder='plots'):

        try:
            print("\n\n\n--- Generating 2D geometry plots with axes using geometry.plot() ---")

            # Plot에 사용할 색상을 재료에 부여
            material_color_map = {
                self.materials['eurofer']: (128, 128, 128),  # 회색
                self.materials['Be12Ti']: (0, 255, 0),  # 초록색
                self.materials['breeder_pebble_mix']: (255, 0, 0),  # 빨간색
                self.materials['He']: (0, 0, 255),  # 파란색
                self.materials['tungsten']: (128, 0, 128) # 보라색
            }

            print("1")
            plot_xy = plot_axis_slice(
                dagmc_file_or_trimesh_object=self.config['geometry']['h5m_path'],
                view_direction='-z',
                plane_origin=[self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord']],
            )

            plot_xy.savefig(os.path.join(plots_folder, 'geometry_2D_DAGMC_xy.png'), dpi=600)
            print("XY geometry plot with axes saved.\n")

            print("2")
            plot_yz = plot_axis_slice(
                dagmc_file_or_trimesh_object=self.config['geometry']['h5m_path'],
                view_direction='x',
                plane_origin=[self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord']],
            )

            plot_xy.savefig(os.path.join(plots_folder, 'geometry_2D_DAGMC_yz.png'), dpi=600)
            print("YZ geometry plot with axes saved.\n")


        except Exception as e:
            print(f"\n\nError in generate_geometry_2d_plots method: {e}\n")
            raise

    # 해석 설정
    def define_settings(self):
        try:
            print("\n\n\nDefining settings...\n")

            # 매번 openmc.Settings 쓰기 귀찮아서 settings로 사용
            self.settings = openmc.Settings()

            # Fusion 해석은 중성자 source가 변하지 않기 때문에 fixed source로 설정
            self.settings.run_mode = 'fixed source'

            sim_config = self.config['simulation']

            # 1회 batch 당 몇 개의 particle이 생성?
            # Fixed source에서는 의미 없음
            # self.settings.generations_per_batch = 10

            # Particles per generation.
            self.settings.particles = sim_config['particles']

            # 해석을 진행할 iteration 수와 비슷한 개념. Batch 수가 2배가 되면 해석 uncertainty는 sqrt(2)배 감소
            self.settings.batches = sim_config['batches']

            # 작은 해석 : 50~100개, 큰 해석 : 수 백개 정도면 충분
            self.settings.inactive = sim_config['inactive']

            # statepoint.*.h5 파일에 source 정보 추가
            self.settings.sourcepoint_write = True

            # Photon transport 해석 추가
            self.settings.photon_transport = True

            # Tracking할 particle 지정
            # self.settings.track = [
            #     (20, 1, 1), # (Batch, Generation, Particle #)
            #     (50, 1, 50),
            #     (80, 1, 100)
            # ]

            # 사용자 선택을 받기 위한 도구
            plasma_source = None
            source_name = ""

            # 사용자의 source 선택에 따라 4가지 중 하나 설정
            if isinstance(self.source_choice, int):

                # 단위 주의해야 할 듯. 점검 필요
                if self.source_choice == 1:
                    source_name = "Tokamak plasma source"
                    print(f"Selected source: {source_name}")
                    plasma_source = tokamak_source(
                        elongation=1.557,
                        ion_density_centre=1.09e20,  # [m^-3]
                        ion_density_peaking_factor=1,
                        ion_density_pedestal=1.09e20,  # [m^-3]
                        ion_density_separatrix=3e19,  # [m^-3]
                        ion_temperature_centre=45900,  # [eV]
                        ion_temperature_peaking_factor=8.06,
                        ion_temperature_pedestal=6090,  # [eV]
                        ion_temperature_separatrix=100,  # [eV]
                        major_radius=self.tokamak_major_radius,  # [cm]
                        minor_radius=self.tokamak_minor_radius,  # [cm]
                        pedestal_radius=0.8 * self.tokamak_minor_radius,  # [cm]
                        mode="H",
                        # shafranov_factor=0.44789,  # [cm]
                        shafranov_factor=0.3 * self.tokamak_minor_radius,  # [cm]
                        triangularity=0.270,
                        ion_temperature_beta=6,
                        sample_size=1000,  # 기본 샘플 수 : 1,000
                        angles=(0.0, 2 * np.pi),  # 한 바퀴 전체
                    )

                elif self.source_choice == 2:
                    source_name = "Simplified ring source"
                    print(f"Selected source: {source_name}")
                    plasma_source = fusion_ring_source(
                        fuel={"D": 0.09, "T": 0.91},  # 핵융합 연료비
                        temperature=20000.0,  # 플라즈마 온도 [eV]
                        radius=self.tokamak_major_radius,  # Major radius에 위치한 링 소스 [cm]
                        angles=(0.0, 2 * np.pi),
                        z_placement=0.0
                    )

                elif self.source_choice == 3:
                    source_name = "Single point source"
                    print(f"Selected source: {source_name}")
                    plasma_source = fusion_point_source(
                        fuel={"D": 0.09, "T": 0.91},  # 핵융합 연료비
                        temperature=20000.0,  # 플라즈마 온도 [eV]
                        coordinate=(self.tokamak_major_radius, 0.0, 0.0)  # 토러스 중심의 한 점 [cm]
                    )

            # Custom source
            elif isinstance(self.source_choice, tuple) and self.source_choice[0] == 4:
                source_name = "Custom Source"
                print(f"Selected source: {source_name}")
                options = self.source_choice[1]

                # IndependentSource 객체 생성
                custom_source = openmc.IndependentSource()

                energy_options = options["energy"]
                energy_type = energy_options["type"]
                params = energy_options["params"]

                # 4개의 옵션 제공
                if energy_type == "Discrete":
                    custom_source.energy = openmc.stats.Discrete([params["energy"]], [params["prob"]])
                elif energy_type == "Watt (fission) distribution":
                    custom_source.energy = openmc.stats.Watt(a=params["a"], b=params["b"])
                elif energy_type == "Muir (normal) distribution":
                    custom_source.energy = openmc.stats.muir(e0=params["e0"], m_rat=params["m_rat"], kt=params["kt"])
                elif energy_type == "Maxwell distribution":
                    custom_source.energy = openmc.stats.Maxwell(theta=params["theta"])

                # 각도 분포는 Isotropic으로 고정
                # custom_source.angle = openmc.stats.Isotropic()
                custom_source.angle = openmc.stats.Monodirectional(reference_uvw=[1.0, 0.0, 0.0])

                space_options = options["space"]

                # Point 소스는 하나밖에 없으니 Point 함수
                if space_options["type"] == "Point":
                    custom_source.space = openmc.stats.Point(space_options["coords"])

                # Hexagonal Face 소스는 여러 개의 point 소스가 있으니 PointCloud 함수
                elif space_options["type"] == "Hexagonal Face":
                    source_positions = create_hexagonal_source_points(
                        n_points=10000,
                        x_coord=space_options["x_coord"],
                        pitch=space_options["pitch"],
                    )
                    custom_source.space = openmc.stats.PointCloud(source_positions)

                # 완성된 custom source를 plasma_source 변수에 할당
                plasma_source = custom_source

            self.settings.source = plasma_source

            print(f"\n\nSource '{source_name}' assigned to settings.")

            # cross_section.xml에 유체의 모든 온도 데이터가 없으므로, 설정한 온도와 가장 가까운 곳의 온도 데이터 사용
            self.settings.temperature = {
                'method': 'interpolation',
                'range': [250.0, 2500.0],
                'default': 600.0,
                'tolerance': 200.0
            }

            # 해석 후 결과 파일로 statepoint.h5 파일 외에 summary.h5 파일과 tallies.out 파일도 같이 저장
            self.settings.output = {
                'summary': True,  # 요약 파일 저장
                'tallies': False,  # 후처리 파일 저장
                'path': 'results',  # /results 폴더에 저장
            }

            print("\nExporting settings to settings.xml...\n")
            self.settings.export_to_xml()
            print("settings.xml exported successfully.\n")
            print("========================================================================\n")
            time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_settings method: {e}\n")
            raise

    # Source 시각화
    def preview_source_distribution(self, plots_folder='plots', n_samples=1000):
        try:
            if self.settings is None or self.settings.source is None:
                print("Error: Settings or source is not defined yet.")
                return

            print("\n\n\n--- Previewing source distribution (before simulation) ---")

            # 시각화 함수에 전달할 openmc.Model 객체 생성
            # model.xml 파일이 생성되는데 여기에는 tally 정보가 없으니 해석 시작 전 model.xml 파일 삭제 필요
            model = openmc.Model(
                geometry=self.geometry,
                materials=self.all_materials_collection,
                settings=self.settings,
            )

            # 공간 분포 (Position)
            print("Generating source position preview...")
            plot_pos = plot_source_position(this=model, n_samples=n_samples)
            plot_pos.update_layout(
                title='Source Particle Staring Position',
                xaxis_title='X [cm]',
                yaxis_title='Y [cm]',
                autosize=True,
                legend=dict(),
                showlegend=True,
            )
            plot_pos.write_html(os.path.join(plots_folder, 'source_preview_position.html'))
            print("-> Source position preview saved.\n")

            # 에너지 분포 (Energy)
            print("Generating source energy preview...")
            plot_en = plot_source_energy(this=model, n_samples=n_samples)
            plot_en.write_html(os.path.join(plots_folder, 'source_preview_energy.html'))
            print("-> Source energy preview saved.\n")

            # 방향 분포 (Direction)
            print("Generating source direction preview...")
            plot_dir = plot_source_direction(this=model, n_samples=n_samples)
            plot_dir.write_html(os.path.join(plots_folder, 'source_preview_direction.html'))
            print("-> Source direction preview saved.\n")

            print("\n--- Source preview generation finished. ---")

        except Exception as e:
            print(f"\n\nError in create_source_previews method: {e}\n")
            raise

    # 해석을 통해 무엇을 볼지?
    def define_tallies(self):

        """
        Tally는 크게 두 개로 나뉨.
            1. Mesh를 이용하지 않고 해석의 평균 값을 보는 것
            2. 2D or 3D Mesh를 이용해서 평균 값이 아닌 local 값을 보는 것
        """

        try:
            print("\n\n\nDefining tallies...")

            self.tallies = []

            # Cubit에서 정의한 볼륨 이름 저장
            target_pressure_tube = ["cell_Pressure_Tube"]
            target_pin = ["cell_Pin"]
            target_breeder = ["cell_Breeder"]
            target_outer_multiplier = ["cell_Outer_Multiplier"]
            target_structure = ["cell_Pressure_Tube", "cell_Pin"]
            target_material = ["cell_Pressure_Tube", "cell_Pin", "cell_Breeder", "cell_Outer_Multiplier"]

            # Cubit 볼륨 이름으로 cell filter 정의
            pressure_tube_cell_filter = openmc.CellFilter(target_pressure_tube)
            pin_cell_filter = openmc.CellFilter(target_pin)
            breeder_cell_filter = openmc.CellFilter(target_breeder)
            outer_multiplier_cell_filter = openmc.CellFilter(target_outer_multiplier)
            structure_cells_filter = openmc.CellFilter(target_structure)
            material_cells_filter = openmc.CellFilter(target_material)

            # 에너지 filter
            energy_bins = np.logspace(-2, 7.3, 501)  # 0.01 eV ~ 20 MeV 범위를 500개로 쪼개기
            energy_filter = openmc.EnergyFilter(energy_bins)


            '''여기부터는 평균 Tally'''

            # 사용자가 알기 쉬운 이름 설정
            tally_tbr = openmc.Tally(name='tbr')
            tally_multiplying = openmc.Tally(name='multiplication')

            # Score 정의 : 무엇을 측정?
            # https://docs.openmc.org/en/stable/usersguide/tallies.html        : OpenMC 내장 Tally
            # https://www.oecd-nea.org/dbdata/data/manual-endf/endf102_MT.pdf  : ENDF MT list
            tally_tbr.scores = ['(n,Xt)']  # 'H3-production' 으로 써도 됨. (X: wildcard) MT number: 205
            tally_tbr.nuclides = ['Li6', 'Li7']  # Li 원자에 대해서 계산
            tally_multiplying.scores = ['(n,2n)']
            tally_multiplying.nuclides = ['Be9']

            # Filter 정의 : Score를 언제/어디서/어떤 입자를 측정?
            tally_tbr.filters = [breeder_cell_filter]
            tally_multiplying.filters = [outer_multiplier_cell_filter]


            # tallies에 모두 추가
            self.tallies.append(tally_tbr)
            self.tallies.append(tally_multiplying)


            '''여기부터는 local Tally'''

            # Local Tally 계산을 위한 mesh 생성
            # RegularMesh : 직육면체 격자, CylindricalMesh : 원통형 격자, SphericalMesh : 구형 격자
            # CylindricalMesh는 현재 z축만 회전축으로 지원하는데, openmc-plasma-source 모듈은 z축을 토카막의 축으로 지정
            # -> 그래서 CylindricalMesh는 의미가 없을 듯

            mesh = openmc.RegularMesh(name='focused_mesh')

            mesh_config = self.config['mesh']

            mesh_x_min = self.tokamak_radius  # OB FW 위치부터
            mesh_x_max = self.tokamak_radius + mesh_config['length_x'] # OB Pin 맨 끝까지

            # 2D or 3D mesh 설정해야 함.
            mesh_z_center = mesh_config['z_center']
            mesh_z_thickness = mesh_config['z_thickness']

            mesh_y_min = -mesh_config['length_y'] / 2.0  # 핀 반경보다 조금 더 크게
            mesh_z_min = mesh_z_center - (mesh_z_thickness / 2.0)

            mesh_y_max = +mesh_config['length_y'] / 2.0
            mesh_z_max = mesh_z_center + (mesh_z_thickness / 2.0)

            # 두 꼭짓점 사이만 격자를 생성
            mesh.lower_left = (mesh_x_min, mesh_y_min, mesh_z_min)  # lower_left 꼭짓점
            mesh.upper_right = (mesh_x_max, mesh_y_max, mesh_z_max)  # upper_right 꼭짓점

            # 격자를 몇개로 나누지? (x, y, z)
            mesh.dimension = (mesh_config['division_x'], mesh_config['division_y'], mesh_config['division_z'])

            mesh_filter = openmc.MeshFilter(mesh)

            tally_local_flux = openmc.Tally(name='local_flux')
            tally_local_tbr = openmc.Tally(name='local_tbr')

            tally_local_heating_neutron = openmc.Tally(name='local_heating_neutron')
            tally_local_heating_photon = openmc.Tally(name='local_heating_photon')

            tally_local_heating_tube_neutron = openmc.Tally(name='local_heating_tube_neutron')
            tally_local_heating_tube_photon = openmc.Tally(name='local_heating_tube_photon')
            tally_local_heating_pin_neutron = openmc.Tally(name='local_heating_pin_neutron')
            tally_local_heating_pin_photon = openmc.Tally(name='local_heating_pin_photon')
            tally_local_heating_breeder_neutron = openmc.Tally(name='local_heating_breeder_neutron')
            tally_local_heating_breeder_photon = openmc.Tally(name='local_heating_breeder_photon')
            tally_local_heating_outer_multiplier_neutron = openmc.Tally(name='local_heating_outer_multiplier_neutron')
            tally_local_heating_outer_multiplier_photon = openmc.Tally(name='local_heating_outer_multiplier_photon')

            tally_local_multiplication = openmc.Tally(name='local_multiplication')

            tally_local_flux.scores = ['flux']
            tally_local_tbr.scores = ['(n,Xt)']

            tally_local_heating_neutron.scores = ['heating']
            tally_local_heating_photon.scores = ['heating']

            tally_local_heating_tube_neutron.scores = ['heating']
            tally_local_heating_tube_photon.scores = ['heating']
            tally_local_heating_pin_neutron.scores = ['heating']
            tally_local_heating_pin_photon.scores = ['heating']
            tally_local_heating_breeder_neutron.scores = ['heating']
            tally_local_heating_breeder_photon.scores = ['heating']
            tally_local_heating_outer_multiplier_neutron.scores = ['heating']
            tally_local_heating_outer_multiplier_photon.scores = ['heating']

            tally_local_multiplication.scores = ['(n,2n)']

            tally_local_tbr.nuclides = ['Li6', 'Li7']
            tally_local_multiplication.nuclides = ['Be9']

            tally_local_flux.filters = [mesh_filter, material_cells_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_tbr.filters = [mesh_filter, breeder_cell_filter]

            tally_local_heating_neutron.filters = [mesh_filter, material_cells_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_heating_photon.filters = [mesh_filter, material_cells_filter, openmc.ParticleFilter(['photon'])]

            tally_local_heating_tube_neutron.filters = [mesh_filter, pressure_tube_cell_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_heating_tube_photon.filters = [mesh_filter, pressure_tube_cell_filter, openmc.ParticleFilter(['photon'])]
            tally_local_heating_pin_neutron.filters = [mesh_filter, pin_cell_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_heating_pin_photon.filters = [mesh_filter, pin_cell_filter, openmc.ParticleFilter(['photon'])]
            tally_local_heating_breeder_neutron.filters = [mesh_filter, breeder_cell_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_heating_breeder_photon.filters = [mesh_filter, breeder_cell_filter, openmc.ParticleFilter(['photon'])]
            tally_local_heating_outer_multiplier_neutron.filters = [mesh_filter, outer_multiplier_cell_filter, openmc.ParticleFilter(['neutron'])]
            tally_local_heating_outer_multiplier_photon.filters = [mesh_filter, outer_multiplier_cell_filter, openmc.ParticleFilter(['photon'])]

            tally_local_multiplication.filters = [mesh_filter, outer_multiplier_cell_filter]

            local_tallies_list = [
                tally_local_flux,
                tally_local_tbr,
                tally_local_heating_neutron,
                tally_local_heating_photon,
                tally_local_heating_tube_neutron,
                tally_local_heating_tube_photon,
                tally_local_heating_pin_neutron,
                tally_local_heating_pin_photon,
                tally_local_heating_breeder_neutron,
                tally_local_heating_breeder_photon,
                tally_local_heating_outer_multiplier_neutron,
                tally_local_heating_outer_multiplier_photon,
                tally_local_multiplication,
            ]

            # Tally를 mesh 하나의 부피로 나눌 것인가?
            for tally in local_tallies_list:
                tally.volume_normalization = True

            self.tallies.extend(local_tallies_list)


            tallies_obj = openmc.Tallies(self.tallies)
            print("\nExporting tallies to tallies.xml...\n")
            tallies_obj.export_to_xml()
            print("tallies.xml exported successfully.\n")
            print("========================================================================\n")
            time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_tallies method: {e}\n")
            raise

    # 해석에 필요한 사전 작업 모두 실시
    def run_setup_pipeline(self, status_window, tasks):
        try:
            pbar_tasks = tasks[3:-1]

            # ANSI 색상 코드 정의
            green = '\033[92m'  # 초록색 시작
            reset = '\033[0m'  # 색상 초기화

            # {l_bar}{bar}{r_bar}는 tqdm의 기본 형식
            # tqdm의 출력 전에 초록색으로 설정하고 출력하면 색상 초기화
            custom_bar_format = f"{green}{{l_bar}}{{bar}}|{{r_bar}}{reset}"

            print("\n--- Running pre-simulation setup pipeline ---")

            with tqdm(
                    total=len(pbar_tasks),
                    desc="Preparation",
                    file=sys.stdout,
                    bar_format=custom_bar_format
            ) as pbar:

                pbar.set_description("Defining Materials")
                self.define_materials()
                status_window.update_task_status("Materials Definition", "OK! ✓", "green")
                pbar.update(1)

                pbar.set_description("Defining Geometry")
                self.define_geometry()
                status_window.update_task_status("Geometry Definition", "OK! ✓", "green")
                pbar.update(1)

                pbar.set_description("\nDefining Settings")
                self.define_settings()
                status_window.update_task_status("Settings Definition", "OK! ✓", "green")
                pbar.update(1)

                pbar.set_description("\nDefining Tallies")
                self.define_tallies()
                status_window.update_task_status("Tallies Definition", "OK! ✓", "green")
                pbar.update(1)

                pbar.set_description("\nGenerating Geometry Plots")
                status_window.update_task_status("Geometry Plots", "Running...", "blue")
                self.generate_geometry_2D_plots(plots_folder='plots')
                status_window.update_task_status("Geometry Plots", "OK! ✓", "green")
                pbar.update(1)

                pbar.set_description("\nGenerating Source Previews")
                status_window.update_task_status("Source Previews", "Running...", "blue")
                self.preview_source_distribution(plots_folder='plots')
                status_window.update_task_status("Source Previews", "OK! ✓", "green")
                pbar.update(1)

        except Exception as e:
            print(f"\n\nError in run_setup_pipeline method: {e}\n")
            raise

    # 해석 시작
    def prompt_and_run_simulation(self, status_window):
        try:

            # Source preview에서 임시로 생성된 model.xml 파일 삭제
            if os.path.exists('model.xml'):
                print("\n--- Removing temporary 'model.xml' before main run ---\n\n")
                os.remove('model.xml')

            status_window.update_task_status("Main OpenMC Simulation", "Running...", "blue")

            # 해석 시작!!
            openmc.run(tracks=False, threads=self.config['simulation']['threads'])

            status_window.update_task_status("Main OpenMC Simulation", "OK! ✓", "green")
            status_window.complete("\nAll simulation tasks finished!\nRefer to /results folder.")

        except Exception as e:
            print(f"\n\nError in run_setup_pipeline method: {e}\n")
            raise