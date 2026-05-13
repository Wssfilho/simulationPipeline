#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from opengate.geometry.materials import read_voxel_materials
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("inicio", type=int, nargs="?",
                        default=0, help="valor inicial (padrao=0)")
    parser.add_argument("--ct", type=str, required=True,
                        help="caminho para a imagem CT (.mhd)")
    parser.add_argument("--hu2mat", type=str, default=None,
                        help="arquivo HU->material [opcional]")
    parser.add_argument("--mat_db", type=str, default=None,
                        help="banco de materiais .db [opcional]")
    args = parser.parse_args()

    print(f"Inicio = {args.inicio}")
    print(f"CT    = {args.ct}")

    inicio = args.inicio
    n = 1

    # units
    m = gate.g4_units.m
    cm = gate.g4_units.cm
    nm = gate.g4_units.nm
    mm = gate.g4_units.mm
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # Simulation
    sim = gate.Simulation()
    sim.visu = False
    sim.random_engine = "MersenneTwister"
    sim.random_seed = "auto"
    sim.g4_verbose_level = 0
    sim.store_json_archive = False
    sim.number_of_threads = 4      # aumenta se tiver mais cores
    sim.output_dir = "output"
    sim.progress_bar = True
    sim.check_volumes_overlap = False
    sim.g4_verbose = False

    # Banco de materiais
    if args.mat_db:
        sim.volume_manager.add_material_database(args.mat_db)
    else:
        sim.volume_manager.add_material_database(
            "dataMASH_tumor/ICRP_HUmaterials.db")

    # World
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "Vacuum"

    # -------------------------------------------------------
    # Volume CT
    # -------------------------------------------------------
    patient = sim.add_volume("Image", "patient")
    patient.image = args.ct
    patient.mother = "world"
    patient.material = "G4_AIR"

    if args.hu2mat:
        vm = read_voxel_materials(args.hu2mat)
    else:
        vm = [
            [-1600, -950, "G4_AIR"],
            [-950, -700, "G4_LUNG_ICRP"],
            [-700, -200, "G4_ADIPOSE_TISSUE_ICRP"],
            [-200,   80, "G4_TISSUE_SOFT_ICRP"],
            [80,  400, "G4_B-100_BONE"],
            [400, 1600, "G4_BONE_COMPACT_ICRU"],
        ]

    patient.voxel_materials = vm
    patient.dump_label_image = "output/CT_label.mhd"
    # -------------------------------------------------------

    # Rotação dinâmica (orbiting do gantry)
    tr, rot = gate.geometry.utility.volume_orbiting_transform(
        "z", inicio, inicio + n, n,
        patient.translation, patient.rotation
    )
    patient.add_dynamic_parametrisation(translation=tr, rotation=rot)

    # CBCT gantry (fonte de raios-X)
    gantry = sim.add_volume("Box", "CBCT_gantry")
    gantry.size = [0.2 * m, 0.2 * m, 0.2 * m]
    gantry.translation = [1000 * mm, 0, 0]
    gantry.material = "Vacuum"
    gantry.color = [0, 1, 1, 1]

    # CBCT detector
    detector_plane = sim.add_volume("Box", "CBCT_detector_plane")
    detector_plane.size = [10 * mm, 1409.6 * mm, 1409.6 * mm]
    detector_plane.translation = [-536 * mm, 0, 0]
    detector_plane.material = "Air"
    detector_plane.color = [1, 0, 0, 1]

    # Physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.set_production_cut("world", "all", 10 * mm)

    # Detector actor
    detector_actor = sim.add_actor("FluenceActor", "detector_actor")
    detector_actor.attached_to = detector_plane
    detector_actor.output_filename = f"Slide-{inicio}.mhd"
    detector_actor.spacing = [10 * mm, 2.56 * mm, 2.56 * mm]  # ~1px/mm
    detector_actor.size = [1, 200, 200]                     # mais resolução
    detector_actor.output_coordinate_system = "local"
    detector_actor.keep_data_per_run = True

    # Stats actor
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = f"stat-{inicio}.txt"

    # -------------------------------------------------------
    # Source — CORREÇÕES PRINCIPAIS
    # -------------------------------------------------------
    source = sim.add_source("GenericSource", "mysource")
    source.attached_to = gantry.name
    source.particle = "gamma"
    source.energy.mono = 70 * keV
    source.position.type = "box"
    source.position.size = [1 * nm, 300 * mm,
                            300 * mm]  # cobre o paciente inteiro
    source.direction.type = "focused"
    source.activity = 1e7 * Bq
    # aponta pro centro do paciente
    source.direction.focus_point = [0, 0, 0]
    # -------------------------------------------------------

    # Timing
    sim.run_timing_intervals = [
        [(i + inicio) * sec, (i + 1 + inicio) * sec] for i in range(n)
    ]
    print(sim.run_timing_intervals)

    sim.run(start_new_process=True)
