"""Configure and launch a full-population NeuroSeq recording analysis."""
import time
from orchestrate_micro_meso import RunConfig, run_pipeline


def main() -> None:
    """Run the same pipeline used by the GUI, with paths edited below.

    Change ``data_folder`` and ``output_root`` for each recording. Ensure
    ``total_duration`` matches the intended analyzed recording interval.
    """
    cfg = RunConfig(
        data_folder="C:/Users/owang/OneDrive/Bureau/Interval_Test/MMPP_20sec_challenging_States_10_replicates/Replicate_010/Spike_Trains_SU_Format",
        output_root="C:/Users/owang/OneDrive/Bureau/New_Test/10/Replicate_001",
        start_time=0.0,
        total_duration=60.0,
        T=0.5,

        ### Gamma sweep for Micro Louvain Epochs
        auto_gamma_min=1.07,
        auto_gamma_max=1.15,
        auto_gamma_step=0.001,
        seeds_per_gamma=3,

        ### For critical points
        persistence_tolerance_bins=1,
        consensus_cp_band_tolerance_bins=0,
        seed_for_states=42,

        ### Meso seed 
        meso_seed=42,
        do_graph=True,
    )
    t0 = time.perf_counter()
    print("NEUROSEQ IS STARTING", flush=True)
    run_pipeline(cfg)
    print(f"NEUROSEQ COMPLETED in {time.perf_counter()-t0:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
