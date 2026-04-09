from visuals.visual1_turbulent_memory import generate_visual1
from visuals.visual2_radial_escape import generate_visual2
from visuals.visual3_gravitational_bias import generate_visual3
from visuals.visual4_noise_geometry import generate_visual4
from visuals.visual5_vortex_constellation import generate_visual5
from visuals.visual6_wavefront_interference import generate_visual6
from visuals.visual7_elastic_lattice import generate_visual7
from visuals.visual8_cellular_bloom import generate_visual8


def main() -> None:
    generators = (
        generate_visual1, # turbulent memory
        #generate_visual2, # radial escape
        #generate_visual3, # gravitational bias
        #generate_visual4, # noise geometry
        generate_visual5, # vortex constellation
        generate_visual6, # wavefront interference
        #generate_visual7, # elastic lattice
        generate_visual8, # cellular bloom
    )

    for generator in generators:
        output_path = generator()
        print(f"Generated {output_path.name}")


if __name__ == "__main__":
    main()
