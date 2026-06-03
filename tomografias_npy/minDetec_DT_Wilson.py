#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from opengate.tests import utility
import math
from scipy.spatial.transform import Rotation
import argparse
from pathlib import Path
import opengate as gate
from opengate.geometry.materials import (
    read_voxel_materials,
)


if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("inicio", type=int, nargs="?",
                        default=0, help="valor inicial (padrao=0)")
    parser.add_argument("--image", default="data/0801.mhd",
                        help="caminho do arquivo .mhd de entrada")
    parser.add_argument("--output-dir", default="output",
                        help="pasta de saida da simulacao")
    parser.add_argument("--tag", default=None,
                        help="tag para nomes de arquivos de saida")
    parser.add_argument(
        "--materials-db",
        default=str(default_data_dir / "ICRP_HUmaterials.db"),
        help="arquivo .db de materiais",
    )
    parser.add_argument(
        "--hu2mat",
        default=str(default_data_dir / "hu2mat.txt"),
        help="arquivo de mapeamento HU para material",
    )
    args = parser.parse_args()

    print(f"Inicio = {args.inicio}")

    inicio = args.inicio
    tag = args.tag if args.tag is not None else str(args.inicio)

    n = 1

    # units
    m = gate.g4_units.m
    cm = gate.g4_units.cm
    nm = gate.g4_units.nm
    cm3 = gate.g4_units.cm3
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    gcm3 = gate.g4_units.g / cm3
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # Simulation
    sim = gate.Simulation()

    sim.visu = False
    sim.visu_type = "qt"
    sim.random_engine = "MersenneTwister"
    sim.random_seed = "auto"

    sim.g4_verbose_level = 1
    sim.store_json_archive = False
    # sim.json_archive_filename = "simulation.json"

    # add a material database
    sim.volume_manager.add_material_database(args.materials_db)

    sim.number_of_threads = 1
    sim.output_dir = args.output_dir

    sim.progress_bar = True
    sim.check_volumes_overlap = False
    sim.g4_verbose = True

    # Misc
    yellow = [1, 1, 0, 1]

    # world
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "Vacuum"

    # image
    patient = sim.add_volume("Image", "patient")
    patient.image = args.image
    patient.mother = "world"
    patient.material = "Vacuum"  # material used by default
    # or alternatively, from a file (like in Gate)
    vm = read_voxel_materials(args.hu2mat)
    # vm[0][0] = 0
    # Note: the voxel_materials are turned into a structured array when setting the user info
    # Therefore, we store the previous one, assign the new one, and only then compare them!
    # assert patient.voxel_materials == vm
    patient.voxel_materials = vm
    # Rotação inicial para vista de frente (AP)
    patient.rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    # write the image of labels (None by default)
    patient.dump_label_image = f"{args.output_dir}/FDK_label.mhd"

    # Rotation
    tr, rot = gate.geometry.utility.volume_orbiting_transform(
        "z", inicio, inicio+n, n, patient.translation, patient.rotation)
    patient.add_dynamic_parametrisation(translation=tr, rotation=rot)

    '''


    def add_spiral_insert(sim, mother, name, rmin=0 * mm, rmax=3 * mm, dz=20 * cm, material="Aluminium", translation=[0 * mm, 0 * mm, 0 * mm], color=yellow):
        spiral_insert = sim.add_volume("Tubs", name=name)
        spiral_insert.mother =  mother.name
        spiral_insert.rmin = rmin
        spiral_insert.rmax = rmax
        spiral_insert.dz = dz
        spiral_insert.material = material
        spiral_insert.translation = translation
        spiral_insert.color = color
    
    def add_spiral(sim):
        # Mother of all
        spiral = sim.add_volume("Tubs", name="Spiral")
        spiral.rmin = 0 * cm
        spiral.rmax = 10 * cm
        spiral.dz = 20 * cm
        spiral.material = "Water"
        spiral.color = yellow


        # Spiral inserts
        sradius = 4
        radius = list(range(0, 100 - sradius // 2, sradius))
        sangle = 139
        angles = [math.radians(a) for a in range(0, sangle * len(radius), sangle)]
        posx = [radius[i] * math.cos(angles[i]) for i in range(len(radius))]
        posy = [radius[i] * math.sin(angles[i]) for i in range(len(radius))]
           
        for i in range(len(radius)):
            add_spiral_insert(sim, spiral, f"SpiralInsert{i:02d}", translation=[posx[i] * mm, posy[i] * mm, 0])

        # Spiral rotation
        tr, rot = gate.geometry.utility.volume_orbiting_transform("z", 0, 360, n, spiral.translation, spiral.rotation)
        spiral.add_dynamic_parametrisation(translation=tr, rotation=rot)


    add_spiral(sim) 

    
    # waterbox
    waterbox = sim.add_volume("Box", "waterbox")
    waterbox.size = [20 * cm, 20 * cm, 20 * cm]
    waterbox.translation = [0 * cm, 0 * cm, 15 * cm]
    waterbox.material = "G4_WATER" '''

    # CBCT gantry source
    gantry = sim.add_volume("Box", "CBCT_gantry")
    gantry.size = [0.2 * m, 0.2 * m, 0.2 * m]
    gantry.translation = [1000 * mm, 0, 0]
    gantry.material = "Vacuum"
    gantry.color = [0, 1, 1, 1]

    # CBCT detector plane
    detector_plane = sim.add_volume("Box", "CBCT_detector_plane")
    detector_plane.size = [1 * mm, 700 * mm, 700 * mm]
    detector_plane.translation = [-536 * mm, 0, 0]
    detector_plane.material = "CdTe"
    detector_plane.color = [1, 0, 0, 1]

    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.set_production_cut("world", "all", 10 * mm)
    # cut

    # for fluency actor
    # detector_actor = sim.add_actor("FluenceActor", "detector_actor")
    # detector_actor.attached_to = detector_plane
    # detector_actor.output_filename = f"Slide-{tag}.mhd"
    # detector_actor.spacing = [1 * mm, 2 * mm, 2 * mm]
    # detector_actor.size = [1, 350, 350]
    # detector_actor.output_coordinate_system = "local"
    # detector_actor.keep_data_per_run = True

    # for dosimetry
    dose_actor = sim.add_actor("DoseActor", "dose_actor")
    dose_actor.attached_to = "patient"
    dose_actor.edep_uncertainty.active = True
    dose_actor.size = [256, 256, 256]
    dose_actor.spacing = [1.5 * mm, 1.5 * mm, 1.5 * mm]
    dose_actor.output_coordinate_system = "attached_to_image"
    dose_actor.translation = [0 * mm, 0 * mm, -0 * mm]
    dose_actor.hit_type = "random"
    dose_actor.keep_data_per_run = False

    # Saídas
    # energia depositada, normalmente já ativa
    dose_actor.edep.active = True
    dose_actor.dose.active = True             # dose absorvida
    dose_actor.dose_uncertainty.active = False  # incerteza estatística da dose

    # Arquivos de saída
    dose_actor.edep.output_filename = f"Edep-{tag}.mhd"
    dose_actor.dose.output_filename = f"Dose-{tag}.mhd"
    dose_actor.dose_uncertainty.output_filename = f"uDose-{tag}.mhd"
    dose_actor.edep_uncertainty.output_filename = f"uEdep-{tag}.mhd"

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = f"stat-{tag}.txt"

    # source
    source = sim.add_source("GenericSource", "mysource")
    source.attached_to = gantry.name
    source.particle = "gamma"
    source.energy.mono = 140 * keV
    source.position.type = "box"
    source.position.size = [1 * nm, 2 * 15 * mm, 2 * 15 * mm]
    source.direction.type = "focused"
    source.activity = 1E2 * Bq
    # FIXME warning in world coord ! Should be in mother coord system
    source.direction.focus_point = [gantry.translation[0] - 60 * mm, 0, 0]

    # print info
    # sim.volume_manager.print_volumes()

    # verbose
    # sim.g4_commands_after_init.append("/tracking/verbose 0")

    # statistics
    # sim.run_timing_intervals = [
    #     (0, 0.3 * sec),
    #     (0.3 * sec, 0.6 * sec),
    #     (0.6 * sec, 1.0 * sec),
    # ]
    # sim.run_timing_intervals = [(0, 1.0 * sec)]

    sim.run_timing_intervals = [
        [(i+inicio) * sec, (i+1+inicio) * sec] for i in range(n)]
    print(sim.run_timing_intervals)

    # sim.run_timing_intervals = [[i * sec, (i + 1) * sec] for i in range(n)]

    sim.run(start_new_process=False)

    # print results at the end
    # print(stats)
    # print(detector_actor)


'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-



if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, "gate_test009_voxels", "test009")

    # create the simulation
    sim = gate.Simulation()

    # main options
    sim.output_dir = paths.output / __file__.rstrip(".py")
    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.visu = False
    sim.output_dir = paths.output
    sim.store_json_archive = True
    sim.json_archive_filename = "simulation_test009_voxels.json"

    # add a material database
    sim.volume_manager.add_material_database(paths.data / "GateMaterials.db")

    # units
    m = gate.g4_units.m
    cm = gate.g4_units.cm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    mm = gate.g4_units.mm

    #  change world size
    world = sim.world
    world.size = [1 * m, 1 * m, 1 * m]

    # add a simple fake volume to test hierarchy
    # translation and rotation like in the Gate macro
    fake = sim.add_volume("Box", "fake")
    fake.size = [40 * cm, 40 * cm, 40 * cm]
    fake.material = "G4_AIR"
    fake.color = [1, 0, 1, 1]
    fake.rotation = Rotation.from_euler("x", 20, degrees=True).as_matrix()

    # image
    patient = sim.add_volume("Image", "patient")
    patient.image = paths.data / "patient-4mm.mhd"
    patient.mother = "fake"
    patient.material = "G4_AIR"  # material used by default
    patient.voxel_materials = [
        [-2000, -900, "G4_AIR"],
        [-900, -100, "Lung"],
        [-100, 0, "G4_ADIPOSE_TISSUE_ICRP"],
        [0, 300, "G4_TISSUE_SOFT_ICRP"],
        [300, 800, "G4_B-100_BONE"],
        [800, 6000, "G4_BONE_COMPACT_ICRU"],
    ]
    # or alternatively, from a file (like in Gate)
    vm = read_voxel_materials(paths.gate_data / "patient-HU2mat-v1.txt")
    vm[0][0] = -2000
    # Note: the voxel_materials are turned into a structured array when setting the user info
    # Therefore, we store the previous one, assign the new one, and only then compare them!
    assert patient.voxel_materials == vm
    patient.voxel_materials = vm
    # write the image of labels (None by default)
    patient.dump_label_image = paths.output / "test009_label.mhd"

    # default source for tests
    source = sim.add_source("GenericSource", "mysource")
    source.energy.mono = 130 * MeV
    source.particle = "proton"
    source.position.type = "sphere"
    source.position.radius = 10 * mm
    source.position.translation = [0, 0, -14 * cm]
    source.activity = 10000 * Bq
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]

    # cuts
    patient.set_production_cut(
        particle_name="electron",
        value=3 * mm,
    )

    # add dose actor
    dose_actor = sim.add_actor("DoseActor", "dose_actor")
    dose_actor.attached_to = "patient"
    dose_actor.edep_uncertainty.active = True
    dose_actor.size = [99, 99, 99]
    dose_actor.spacing = [2 * mm, 2 * mm, 2 * mm]
    dose_actor.output_coordinate_system = "attached_to_image"
    dose_actor.translation = [2 * mm, 3 * mm, -2 * mm]
    dose_actor.hit_type = "random"

    # add stat actor
    stats_actor = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats_actor.track_types_flag = True

    # print info
    print(sim.volume_manager.dump_volumes())

    # verbose
    sim.g4_commands_after_init.append("/tracking/verbose 0")

    # start simulation
    sim.run(start_new_process=True)

    # print results at the end
    print(stats_actor)
    print(dose_actor)

    # tests
    stats_ref = utility.read_stat_file(paths.gate_output / "stat.txt")
    is_ok = utility.assert_stats(stats_actor, stats_ref, 0.15)
    is_ok = is_ok and utility.assert_images(
        paths.gate_output / "output-Edep.mhd",
        dose_actor.edep.get_output_path(),
        stats_actor,
        tolerance=35,
        ignore_value_data2=0,
        apply_ignore_mask_to_sum_check=False,  # reproduce legacy behavior of assert_images
    )

    utility.test_ok(is_ok)'''
