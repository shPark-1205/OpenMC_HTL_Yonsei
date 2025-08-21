# Simulation_Builder_DAGMC.py

import openmc
import numpy as np
import os
import sys
from tqdm import tqdm
import neutronics_material_maker as nmm
from openmc_plasma_source import tokamak_source, fusion_ring_source, fusion_point_source
from openmc_source_plotter import plot_source_energy, plot_source_position, plot_source_direction

# unit cross-section 내부 무작위 위치의 중성자 소스 분포
def create_unit_geometry_source_points(n_points, z_coord, characteristic_length):

    points = []

    # Unit cross-section의 대표 길이 (한 변의 길이)
    # s = pitch * (2.0 / 3.0)
    s = characteristic_length

    # 해석 형상을 충분히 감싸는 큰 사각형 생성
    z = z_coord
    abs_x_max = s * (np.sqrt(3.0) / 2.0)
    abs_y_max = s

    print(f"\nGenerating {n_points} source points on a unit cross-section at z={z_coord}...")

    while len(points) < n_points:
        """해석하고자 하는 형상에 맞게 수정 필요"""
        # 사각형 안에서 무작위 샘플링
        x = np.random.uniform(-abs_x_max, abs_x_max)
        y = np.random.uniform(-abs_y_max, abs_y_max)

        is_inside = (abs(-x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s) and \
                    (abs(x + (y * np.sqrt(3.0))) <= np.sqrt(3.0) * s)

        if is_inside:
            points.append((x, y, z))

    print(f"\nGenerated {len(points)} points for unit cross-section source.")
    return points


class NuclearFusion:
    def __init__(self, source_choice, cross_section_path, config):
        try:

            # config 객체를 클래스 속성으로 저장
            self.config = config

            # 해석에 사용할 재료를 임시로 저장
            self.material_configs = []

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
            
            # Mesh Tally 설정 시 Cell 이름과 ID 매칭 용도
            self.cell_name_to_id = {}
            
            # Docker image 안에 OpenMC 공식 library(ENDF/B-VII.1, JEFF 3.3, FENDL 3.2)를 넣어 놨음.
            # https://openmc.org/official-data-libraries/
            openmc.config['cross_sections'] = cross_section_path
            self.cross_section_path = cross_section_path # define_settings 메소드에서 photon_transport 제어하기 위해 정의

        except Exception as e:
            print(f"\n\nError during NuclearFusion class initialization: {e}\n")
            raise

    # 해석에 사용할 재료 정의
    def define_materials(self):
        try:
            # neutronics_material_maker(nmm) 모듈을 통해 미리 정의된 물성을 사용
            print("\n\n\nDefining materials using neutronics_material_maker and OpenMC built-in modules...")

            self.materials = {}

            # 재료 담을 리스트
            openmc_materials_list = []

            # 증식재 혼합물 만들기 위해 임시로 저장하는 곳
            nmm_materials_temp = {}

            self.material_configs = [
                {'id': 100, 'nmm_name': 'eurofer', 'output_name': 'eurofer_base', 'kwargs': {
                    'temperature': self.config['materials']['eurofer']['temperature'],
                }},

                # 유체는 온도/압력 정의 필수
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
                {'id': 301, 'nmm_name': 'tungsten', 'output_name': 'tungsten','kwargs': {}},

                # 증식재의 Li6 enrichment정의 가능
                {'id': 402, 'nmm_name': 'Li4SiO4', 'output_name': 'Li4SiO4', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'temperature': self.config['materials']['breeder']['temperature'],
                }},
                {'id': 403, 'nmm_name': 'Li2TiO3', 'output_name': 'Li2TiO3', 'kwargs': {
                    'enrichment': self.config['materials']['breeder']['li_enrichment'],
                    'temperature': self.config['materials']['breeder']['temperature'],
                }},
                # {'id': 500, 'name': 'Be12Ti', 'kwargs': {}},
            ]

            # 임시 재료 목록 가져오기
            for mat_config in self.material_configs:
                mat_id = mat_config['id']
                nmm_name = mat_config['nmm_name']
                mat_kwargs = mat_config['kwargs']

                print(f"\nLoading base material: {nmm_name} with parameters {mat_kwargs}...")

                # nmm 모듈에서 재료 목록 불러오기
                nmm_mat = nmm.Material.from_library(nmm_name, **mat_kwargs)
                
                # nmm 물성을 openmc 물성으로 저장
                openmc_mat = nmm_mat.openmc_material

                if nmm_name == 'eurofer':
                    print(f"Cloning material: {nmm_name} for different components...\n")

                    # Pressure tube 용
                    eurofer_pressure_tube_mat = openmc_mat.clone()
                    eurofer_pressure_tube_mat.id = 101
                    eurofer_pressure_tube_mat.name = 'eurofer_pressure_tube'
                    self.materials['eurofer_pressure_tube'] = eurofer_pressure_tube_mat
                    openmc_materials_list.append(eurofer_pressure_tube_mat)
                    print(f"  -> Created 'eurofer_pressure_tube' (ID: {eurofer_pressure_tube_mat.id})")

                    # Pin 용
                    eurofer_pin_mat = openmc_mat.clone()
                    eurofer_pin_mat.id = 102
                    eurofer_pin_mat.name = 'eurofer_pin'
                    self.materials['eurofer_pin'] = eurofer_pin_mat
                    openmc_materials_list.append(eurofer_pin_mat)
                    print(f"  -> Created 'eurofer_pin' (ID: {eurofer_pin_mat.id})")

                    # First wall channel 용
                    eurofer_first_wall_channel_mat = openmc_mat.clone()
                    eurofer_first_wall_channel_mat.id = 103
                    eurofer_first_wall_channel_mat.name = 'eurofer_first_wall_channel'
                    self.materials['eurofer_first_wall_channel'] = eurofer_first_wall_channel_mat
                    openmc_materials_list.append(eurofer_first_wall_channel_mat)
                    print(f"  -> Created 'eurofer_first_wall_channel' (ID: {eurofer_first_wall_channel_mat.id})")

                # elif mat_name == 'He':
                #     print(f"Cloning material: {mat_name} for different components...\n")
                #
                #     # Inner He 용
                #     he_inner_mat = openmc_mat.clone()
                #     he_inner_mat.id = 201
                #     he_inner_mat.name = 'He_inner'
                #     self.materials['He_inner'] = he_inner_mat
                #     openmc_materials_list.append(he_inner_mat)
                #     print(f"  -> Created 'He_inner' (ID: {he_inner_mat.id})")
                #
                #     # Outer He 용
                #     he_outer_mat = openmc_mat.clone()
                #     he_outer_mat.id = 202
                #     he_outer_mat.name = 'He_outer'
                #     self.materials['He_outer'] = he_outer_mat
                #     openmc_materials_list.append(he_outer_mat)
                #     print(f"  -> Created 'He_outer' (ID: {he_outer_mat.id})")

                # elif mat_name == 'Be12Ti':
                #     print(f"Cloning material: {mat_name} for different components...\n")
                #
                #     # Inner Be12Ti 용
                #     be12ti_inner_mat = openmc_mat.clone()
                #     be12ti_inner_mat.id = 501
                #     be12ti_inner_mat.name = 'Be12Ti_inner'
                #     self.materials['Be12Ti_inner'] = be12ti_inner_mat
                #     openmc_materials_list.append(be12ti_inner_mat)
                #     print(f"  -> Created 'Be12Ti_inner' (ID: {be12ti_inner_mat.id})")
                #
                #     # Outer Be12Ti 용
                #     be12ti_outer_mat = openmc_mat.clone()
                #     be12ti_outer_mat.id = 502
                #     be12ti_outer_mat.name = 'Be12Ti_outer'
                #     self.materials['Be12Ti_outer'] = be12ti_outer_mat
                #     openmc_materials_list.append(be12ti_outer_mat)
                #     print(f"  -> Created 'Be12Ti_outer' (ID: {be12ti_outer_mat.id})")

                else:
                    output_name = mat_config['output_name']
                    openmc_mat.id = mat_id
                    openmc_mat.name = output_name
                    self.materials[output_name] = openmc_mat
                    nmm_materials_temp[output_name] = nmm_mat
                    openmc_materials_list.append(openmc_mat)
                    print(f"Material {output_name} loaded (OpenMC ID: {openmc_mat.id}).")

            # 증배재만 Thermal scattering data 추가하기 위해 따로 설정
            print(f"\nCreating material: Be12Ti...")
            Be12Ti_inner_mat = openmc.Material(name='Be12Ti_inner', material_id=501)
            Be12Ti_inner_mat.add_elements_from_formula('Be12Ti')
            Be12Ti_inner_mat.set_density('g/cm3', 2.27)
            Be12Ti_inner_mat.temperature = self.config['materials']['multiplier']['temperature']

            # 베릴륨은 감속재로 작용하기 때문에 thermal scattering data 설정해야 함.
            # FENDL 라이브러리에는 thermal scattering data가 없기 때문에 예외 처리
            if 'fendl' not in self.cross_section_path.lower():
                print(" -> Adding thermal scattering data for Be12Ti...")
                Be12Ti_inner_mat.add_s_alpha_beta('c_Be')  # cross_sections.xml 파일에 이름 있음.
            else:
                print(" -> FENDL library detected. Skipping thermal scattering data addition.")

            self.materials['Be12Ti_inner'] = Be12Ti_inner_mat
            openmc_materials_list.append(Be12Ti_inner_mat)

            print(f"\nCloning material: Be12Ti...")
            Be12Ti_outer_mat = Be12Ti_inner_mat.clone()
            Be12Ti_outer_mat.id = 502
            Be12Ti_outer_mat.name = 'Be12Ti_outer'
            self.materials['Be12Ti_outer'] = Be12Ti_outer_mat
            openmc_materials_list.append(Be12Ti_outer_mat)

            # 위에서 생성한 기본 재료들을 사용하여 혼합물 생성
            print("\nCreating mixed material (Li4SiO4 + Li2TiO3)...")

            # 혼합할 nmm.Material 객체 불러오기
            mat1 = nmm_materials_temp['Li4SiO4']
            mat2 = nmm_materials_temp['Li2TiO3']

            # 65% Li4SiO4와 35% Li2TiO3를 부피비(vo)로 혼합
            mixed_breeder_material = nmm.Material.from_mixture(
                materials=[mat1, mat2],
                fracs=[self.config['materials']['breeder']['mixture_Li4SiO4'], self.config['materials']['breeder']['mixture_Li2TiO3']],
                percent_type='vo',
                packing_fraction=self.config['materials']['breeder']['packing_fraction'],
                temperature=self.config['materials']['breeder']['temperature'],
                name='breeder_pebble_mix'  # 새로운 재료의 이름 지정
            )

            # 새로 만든 혼합물을 재료 딕셔너리와 리스트에 추가
            mixed_openmc_material = mixed_breeder_material.openmc_material
            mixed_openmc_material.id = 401
            mixed_openmc_material.name = 'breeder_pebble_mix'
            self.materials['breeder_pebble_mix'] = mixed_openmc_material
            openmc_materials_list.append(mixed_openmc_material)

            print(f"Mixed material breeder_pebble_mix created (OpenMC ID: {mixed_openmc_material.id}).")

            # 모든 OpenMC Material 객체들을 openmc.Materials 컬렉션으로 묶음
            self.all_materials_collection = openmc.Materials(openmc_materials_list)

            # 정의한 재료를 OpenMC가 필요로 하는 xml 파일로 저장
            print("\nExporting materials to materials.xml...\n")
            self.all_materials_collection.export_to_xml()
            print("materials.xml exported successfully.\n")
            # print("========================================================================\n")
            # time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_materials method: {e}\n")
            raise

    # 해석에 사용할 형상 정의
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

        """
        지금 해석 형상은 육각기둥 모양의 unit cell이기 때문에
        DAGMC보다 아주 약간 큰 육각기둥을 만들고 periodic 경계 조건을 부여함.
        만약 해석하고자 하는 형상이 육각기둥이 아니라면,
        final_region을 본인 형상에 맞게 수정해야 함.
        """
        
        # HexagonalPrism은 z축 axis만 지원
        hex_prism = openmc.model.HexagonalPrism(
            edge_length=(self.config['geometry']['characteristic_length']),
            origin=(0.0, 0.0),
            orientation='y',
            boundary_type='periodic'
        )

        z_min_plane = openmc.ZPlane(z0=self.config['bounding']['z_min'], boundary_type='vacuum')
        z_max_plane = openmc.ZPlane(z0=self.config['bounding']['z_max'], boundary_type='vacuum')

        final_region = -hex_prism & +z_min_plane & -z_max_plane

        # DAGMC를 위해 형상 불러오기
        h5m_path = self.config['geometry']['h5m_path']

        # auto_geom_ids를 활성화해야 OpenMC CSG ID랑 충돌하지 않는 것 같음.
        dag_universe = openmc.DAGMCUniverse(filename=h5m_path, auto_geom_ids=True)

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

            # Fixed source 해석에서는 의미 없음.
            # self.settings.inactive = sim_config['inactive']

            # 중간 batch에서 해석 결과 저장
            # self.settings.statepoint = {'batches': range(5, sim_config['batches'] + 5, 5)}
            
            # Traks.h5 파일에 포함할 최대 particle 수
            self.settings.max_tracks = sim_config['max_tracks']

            # FENDL library에는 동위원소에 대한 photon 데이터가 없으므로 비활성화
            if 'fendl' in self.cross_section_path.lower():
                print("FENDL library detected. Disabling photon_transport.\n")
                self.settings.photon_transport = False
            else: # ENDF/JEFF는 동위원소에 대한 photon library가 모두 있으니 활성화
                print("ENDF/JEFF library detected. Enabling photon_transport.\n")
                self.settings.photon_transport = True

            # Tracking할 particle 지정
            # self.settings.track = [
            #     (20, 1, 1), # (Batch, Generation, Particle #)
            #     (50, 1, 100),
            #     (80, 1, 10000)
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

                # Source direction 설정
                angle_options = options["angle"]
                angle_type = angle_options["type"]

                if angle_type == "Isotropic":
                    custom_source.angle = openmc.stats.Isotropic()
                elif angle_type == "Monodirectional":
                    custom_source.angle = openmc.stats.Monodirectional(reference_uvw=angle_options["uvw"])

                space_options = options["space"]

                # Point 소스는 하나밖에 없으니 Point 함수
                if space_options["type"] == "Point":
                    custom_source.space = openmc.stats.Point(space_options["coords"])

                # Unit cross-section 소스는 여러 개의 point 소스가 있으니 PointCloud 함수
                elif space_options["type"] == "Unit cross-section":
                    source_positions = create_unit_geometry_source_points(
                        # mesh tally에 사용한 mesh와 1:1 matching될 정도의 충분한 수 필요
                        n_points=sim_config['n_source_points'],
                        z_coord=space_options["z_coord"],
                        characteristic_length=space_options["characteristic_length"],
                    )
                    custom_source.space = openmc.stats.PointCloud(source_positions)

                # 완성된 custom source를 plasma_source 변수에 할당
                plasma_source = custom_source

            self.settings.source = plasma_source

            print(f"\nSource '{source_name}' assigned to settings.")

            # cross_section.xml에 유체의 모든 온도 데이터가 없으므로, 설정한 온도와 가장 가까운 곳의 온도 데이터 사용
            self.settings.temperature = {
                'method': 'nearest',
                'default': 600.0,
                'tolerance': 1000.0
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
            # time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_settings method: {e}\n")
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
            eurofer_filter = openmc.MaterialFilter([eurofer_pressure_tube_object, eurofer_pin_object, eurofer_first_wall_channel_object], filter_id=41)
            tungsten_filter = openmc.MaterialFilter([tungsten_object], filter_id=51)

            # Energy filter
            # energy_bins = np.logspace(-2, 7.3, 501)  # 0.01 eV ~ 20 MeV 범위를 500개로 쪼개기
            # energy_filter = openmc.EnergyFilter(energy_bins, filter_id=61)
            energy_filter = openmc.EnergyFilter.from_group_structure('UKAEA-1102') # 미리 정의된 에너지 bin 사용

            # Particle filter
            neutron_filter = openmc.ParticleFilter(['neutron'], filter_id=71)
            # photon_filter = openmc.ParticleFilter(['photon'], filter_id=72)
            particle_filter = openmc.ParticleFilter(['neutron', 'photon'], filter_id=73)


            '''여기부터는 평균 Tally'''
            # Score 정의 : 무엇을 측정?
            # https://docs.openmc.org/en/stable/usersguide/tallies.html        : OpenMC 내장 Tally
            # https://www.oecd-nea.org/dbdata/data/manual-endf/endf102_MT.pdf  : ENDF MT list
            # Filter 정의 : Score를 언제/어디서/어떤 입자를 측정?

            # 사용자가 알기 쉬운 이름 설정
            tally_tbr = openmc.Tally(name='tbr', tally_id=98)
            tally_tbr.scores = ['H3-production']  # '(n,Xt)' 으로 써도 됨. (X: wildcard) MT number: 205
            tally_tbr.nuclides = ['Li6', 'Li7']  # Li 원자에 대해서 계산
            tally_tbr.filters = [breeder_filter]
            self.tallies.append(tally_tbr)

            tally_multiplying = openmc.Tally(name='multiplication', tally_id=99)
            tally_multiplying.scores = ['(n,2n)']
            tally_multiplying.nuclides = ['Be9']
            tally_multiplying.filters = [be12ti_outer_filter]
            self.tallies.append(tally_multiplying)

            tally_global_structure = openmc.Tally(name='global_structure', tally_id=94)
            tally_global_structure.scores = ['flux', 'absorption', 'elastic']
            tally_global_structure.filters = [eurofer_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_structure)

            tally_global_armor = openmc.Tally(name='global_armor', tally_id=95)
            tally_global_armor.scores = ['flux', 'absorption', 'elastic']
            tally_global_armor.filters = [tungsten_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_armor)

            tally_global_multiplier = openmc.Tally(name='global_multiplier', tally_id=96)
            tally_global_multiplier.scores = ['flux', 'absorption', 'elastic']
            tally_global_multiplier.filters = [be12ti_outer_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_multiplier)

            tally_global_breeder = openmc.Tally(name='global_breeder', tally_id=97)
            tally_global_breeder.scores = ['flux', 'absorption', 'elastic']
            tally_global_breeder.filters = [breeder_filter, neutron_filter, energy_filter]
            self.tallies.append(tally_global_breeder)

            '''여기부터는 local Tally'''

            # Local Tally 계산을 위한 mesh 생성
            # RegularMesh : 직육면체 격자, CylindricalMesh : 원통형 격자, SphericalMesh : 구형 격자
            # CylindricalMesh는 현재 z축만 회전 축으로 지원
            # 3D mesh로 데이터를 뽑고 싶으면 CylindricalMesh 사용하면 될 듯

            """
            # OpenMC의 정렬격자 사용
            mesh = openmc.RegularMesh(name='focused_mesh')

            mesh_config = self.config['mesh']

            mesh_z_min = self.tokamak_radius  # OB FW 위치부터
            mesh_z_max = self.tokamak_radius + mesh_config['length_z'] # OB Pin 맨 끝까지

            # 2D or 3D mesh 설정해야 함.
            # 지금은 yz 평면의 pseudo-2D mesh
            mesh_y_center = mesh_config['y_center']
            mesh_y_thickness = mesh_config['y_thickness']

            mesh_y_min = mesh_y_center - (mesh_y_thickness / 2.0)
            mesh_y_max = mesh_y_center + (mesh_y_thickness / 2.0)

            mesh_x_min = 0.0
            mesh_x_max = mesh_config['length_x']

            # 두 꼭짓점 사이만 격자를 생성
            mesh.lower_left = (mesh_x_min, mesh_y_min, mesh_z_min)  # lower_left 꼭짓점
            mesh.upper_right = (mesh_x_max, mesh_y_max, mesh_z_max)  # upper_right 꼭짓점

            # 격자를 몇개로 나누지? (x, y, z)
            mesh.dimension = (mesh_config['division_x'], mesh_config['division_y'], mesh_config['division_z'])

            mesh_filter = openmc.MeshFilter(mesh, filter_id=99)
            """

            # 해석 형상을 감싸는 cylindrical mesh 생성
            mesh_cylindrical_config = self.config['mesh_cylindrical']

            mesh_cylindrical = openmc.CylindricalMesh(name='cylindrical_mesh',
                                                      r_grid=np.linspace(
                                                          mesh_cylindrical_config['r_min'],
                                                          mesh_cylindrical_config['r_max'],
                                                          mesh_cylindrical_config['division_r']),
                                                      phi_grid=np.linspace(
                                                          mesh_cylindrical_config['phi_min'],
                                                          mesh_cylindrical_config['phi_max'],
                                                          mesh_cylindrical_config['division_phi']),
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


            # multiplier 표면의 current 계산을 위한 surface mesh 생성 (r 방향)
            mesh_outer_multiplier_config = self.config['mesh_outer_multiplier']
            mesh_outer_multiplier = openmc.CylindricalMesh(name='outer_multiplier_r_mesh',
                                                           r_grid=(mesh_outer_multiplier_config['r_min'],
                                                                   mesh_outer_multiplier_config['r_max']),
                                                           # multiplier의 안쪽 면부터 바깥쪽 면까지
                                                           phi_grid=(mesh_outer_multiplier_config['phi_min'],
                                                                     mesh_outer_multiplier_config['phi_max']),
                                                           # 0 deg ~ 30 deg (-30 deg는 반영 X)
                                                           z_grid=(mesh_outer_multiplier_config['z_min'],
                                                                   mesh_outer_multiplier_config['z_max']),
                                                           # multiplier의 축 방향 좌표
                                                           origin=(0.0, 0.0, 0.0)) # z_grid의 기준점

            mesh_outer_multiplier_r_surface_filter = openmc.MeshSurfaceFilter(mesh_outer_multiplier, filter_id=201)

            tally_current_multiplier = openmc.Tally(name='multiplier_r_current', tally_id=301)
            tally_current_multiplier.scores = ['current']
            tally_current_multiplier.filters = [mesh_outer_multiplier_r_surface_filter, energy_filter, neutron_filter]

            local_tallies_list = [
                tally_local_heating_breeder,
                tally_local_heating_multiplier,
                tally_local_heating_structure,
                tally_current_multiplier,
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
            # time.sleep(1)

        except Exception as e:
            print(f"\n\nError in define_tallies method: {e}\n")
            raise

    # 해석 시작 전에 가능한 시각화
    def generate_geometry_2D_plots(self, plots_folder='plots'):
        try:
            print("\n\n\n--- Generating 2D material-colored geometry plots with axes using openmc.Plot() ---")

            material_colors = {
                self.materials['eurofer_pressure_tube']: (128, 128, 128),  # 회색
                self.materials['eurofer_pin']: (128, 128, 128),  # 회색
                self.materials['eurofer_first_wall_channel']: (128, 128, 128),  # 회색
                self.materials['Be12Ti_inner']: (0, 255, 0),  # 초록색
                self.materials['Be12Ti_outer']: (0, 255, 0),  # 초록색
                self.materials['breeder_pebble_mix']: (255, 0, 0),  # 빨간색
                self.materials['He_inner']: (0, 0, 255),  # 파란색
                self.materials['He_outer']: (0, 0, 255),  # 파란색
                self.materials['tungsten']: (128, 0, 128),  # 보라색
            }

            # Plot 객체 생성
            plot_xy = openmc.Plot()
            plot_xy.filename = os.path.join(plots_folder, 'geometry_by_material_xy')
            plot_xy.width = (15.0, 15.0)
            plot_xy.pixels = (1500, 1500)
            plot_xy.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'])
            plot_xy.basis = 'xy'
            plot_xy.color_by = 'material'
            plot_xy.colors = material_colors

            # plot_yz = openmc.Plot()
            # plot_yz.filename = os.path.join(plots_folder, 'geometry_by_material_yz')
            # plot_yz.width = (30.0, 60.0)
            # plot_yz.pixels = (1200, 2400)
            # plot_yz.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'])
            # plot_yz.basis = 'yz'
            # plot_yz.color_by = 'material'
            # plot_yz.colors = material_colors

            plot_zx = openmc.Plot()
            plot_zx.filename = os.path.join(plots_folder, 'geometry_by_material_zx')
            plot_zx.width = (15.0, 60.0)
            plot_zx.pixels = (1500, 6000)
            plot_zx.origin = (self.config['2D_plot']['x_coord'], self.config['2D_plot']['y_coord'], self.config['2D_plot']['z_coord'])
            plot_zx.basis = 'xz'
            plot_zx.color_by = 'material'
            plot_zx.colors = material_colors

            # 플롯 생성
            plots = openmc.Plots([plot_xy, plot_zx])
            plots.export_to_xml()  # plots.xml 생성

            # geometry.xml과 materials.xml이 이미 생성된 상태여야 함
            openmc.plot_geometry()

            print(f" -> Material-colored plots saved.\n")

        except Exception as e:
            print(f"\n\nError in generate_geometry_2d_plots method: {e}\n")
            raise

    # Source 시각화
    def preview_source_distribution(self, plots_folder='plots', n_samples=500):
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
                autosize=True,
                legend=dict(),
                showlegend=True,
            )
            plot_pos.write_html(os.path.join(plots_folder, 'source_preview_position.html'))
            print("-> Source position preview saved.\n")

            # 에너지 분포 (Energy)
            print("Generating source energy preview...")
            plot_en = plot_source_energy(this=model, n_samples=n_samples)
            plot_en.update_layout(
                title='Source Particle Energy Distribution',
                xaxis_title='Energy [MeV]',
                yaxis_title='Probability [-]',
                autosize=True,
                legend=dict(),
                showlegend=True,
            )
            plot_en.write_html(os.path.join(plots_folder, 'source_preview_energy.html'))
            print("-> Source energy preview saved.\n")

            # 방향 분포 (Direction)
            print("Generating source direction preview...")
            plot_dir = plot_source_direction(this=model, n_samples=n_samples)
            plot_dir.update_layout(
                title='Source Particle Moving Direction',
                autosize=True,
                legend=dict(),
                showlegend=True,
            )
            plot_dir.write_html(os.path.join(plots_folder, 'source_preview_direction.html'))
            print("-> Source direction preview saved.\n")

            print("\n--- Source preview generation finished. ---")

        except Exception as e:
            print(f"\n\nError in create_source_previews method: {e}\n")
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
            openmc.run(tracks=True, threads=self.config['simulation']['threads'])

            status_window.update_task_status("Main OpenMC Simulation", "OK! ✓", "green")
            status_window.complete("\nAll simulation tasks finished!\nRefer to /results folder.")

        except Exception as e:
            print(f"\n\nError in run_setup_pipeline method: {e}\n")
            raise