import os
import glob
import numpy as np
import pandas as pd
import pyBigWig
from pybedtools import BedTool
from multiprocessing import Pool
from tqdm import tqdm



os.environ["TMPDIR"] = "/s/chromatin/c/nobackup/Saira/BERTModel/tmp_pybedtools/"

thresholds_500bps = {
     "DNA-binding protein": 0.9,
    "histone related": 1.0, #1.1
    "OCR": 0.7, #
    "histone": 1.3, #
    "Other": 0.8, 
    "DNA methylation": 0.9, # 0.9
}

# ================= FACTOR TYPE THRESHOLDS =================
THRESHOLD_MAP = {
     "DNA-binding protein": 0.9, #0.9 for 200 bins - 0.8 for introns
    "histone related": 1.2,
    "OCR": 0.8, #0.8 for 200 bins, 0.6 for introns
    "histone": 1.4,
    "Other": 0.9, #0.9 for 200 bins - 0.8 for introns
    "DNA methylation": 1.0
}

THRESHOLD_MAP_1000bps = {
     "DNA-binding protein": 0.9,
    "histone related": 1.1, #1.1
    "OCR": 0.6, #
    "histone": 1.3, #
    "Other": 0.8, 
    "DNA methylation": 0.9, # 0.9
}

thresholds_500bpsIR = {
     "DNA-binding protein": 0.85,
    "histone related": 1.0, #1.1
    "OCR": 0.55, #
    "histone": 1.4, #
    "Other": 0.7, 
    "DNA methylation": 0.9, # 0.9
}
# ================= CONFIGURATION =================
# Paths to your input folders
BW_FOLDER = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Bigwig/"       
BED_FOLDER = "/s/chromatin/b/nobackup/deepplant/Data/arabidopsis_thaliana/Chiphub_bedfile/"  
BINS_FILE =  "athal_500bp_bins.bed"
BINS_FILE = "enhancers_2500.bed"
BINS_FILE = "../idiffir_py3x/ir200_introns.bed"
BINS_FILE = "enhancers_600.bed"
BINS_FILE = "../idiffir_py3x/introns_5p3p_150_tested.bed"
BINS_FILE = "gene_TSS_600_3w.bed"

BINS_FILE = "../enhancers_600_full.bed"
BINS_FILE = "../DATA/phenotype/600_clutivar_932.bed"
#"/s/chromatin/a/nobackup/Saira/SATORI_ENHANCEMENTS/satoriv2/enhancers_500.bed" # enhancers file for 500bp bins
#"/s/chromatin/a/nobackup/Saira/SATORI_ENHANCEMENTS/satoriv2/enhancers_50.bed"
#"../idiffir_py3x/introns_5p3p_200bp_tested.bed" # "athal_200bp_bins.bed" # The master 200bp bins file created earlier
METADATA_PATH = '/s/chromatin/c/nobackup/Saira/BERTModel/merged_metadata_AT_OCR.csv'
# Output filenames
OUT_CONT_MATRIX = "../DATA/phenotype/600_clutivar_932.parquet"
OUT_DISC_MATRIX = "../DATA/phenotype/600_clutivar_932_disc.parquet"
bin_size = 600
overlap_threshold = 100 / bin_size  # 0.2 for 500bp bins, 0.5 for 200bp bins


print("Configuration:")
print(f"BW Folder: {BW_FOLDER}")
print(f"BED Folder: {BED_FOLDER}")
print(f"Bins File: {BINS_FILE}")
print(f"Metadata Path: {METADATA_PATH}")
print(f"Output Continuous Matrix: {OUT_CONT_MATRIX}")
print(f"Output Discretized Matrix: {OUT_DISC_MATRIX}")
print(f"Overlap Threshold: {overlap_threshold:.2f}")
# Number of cores to use (Leave 1 or 2 free for system)
NUM_CORES = 10

# ================= HELPER FUNCTIONS =================

def load_master_bins(bins_file):
    """
    Loads the master BED file to define the index for our matrix.
    """
    #df = pd.read_csv(bins_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'name']) #, 'score', 'strand'])
    df = pd.read_csv(bins_file, sep='\t')

    # remove any rows with start < 0 
    df = df[df['start'] >= 0].copy()

    if df['name'].isna().any():
        df['index_id'] = df['chrom'] + ":" + df['start'].astype(str) + "-" + df['end'].astype(str)
    else:
        df['index_id'] = df['chrom'] + ":" + df['start'].astype(str) + "-" + df['end'].astype(str) + "-" + df['name'].astype(str)
    return df

def process_experiment(args):
    """
    Worker function to process a single experiment (1 BW + 1 BED).
    Returns the continuous column and the discretized column.
    """
    bw_path, bed_path, bin_df, bin_size, factor_type = args
    experiment_name = os.path.basename(bw_path).replace('.bw', '')
    
    bw = pyBigWig.open(bw_path)
    signals = []
    
    for _, row in bin_df.iterrows():
        try:
            val = bw.stats(row['chrom'], row['start'], row['end'], type="mean")[0]
            signals.append(val if val is not None else 0.0)
        except RuntimeError:
            signals.append(0.0)
    bw.close()
    
    signals = np.array(signals, dtype=np.float32)

    bins_bt = BedTool.from_dataframe(bin_df[['chrom', 'start', 'end']])
    peaks_bt = BedTool(bed_path)

    #intersect_a = bins_bt.intersect(peaks_bt, c=True) # any overlap
    intersect = bins_bt.intersect(peaks_bt, f=overlap_threshold, F=1.0, e=True, c=True)

    # 1. Identify all bins with peak overlap
    overlap_mask = np.array([int(feat[-1]) > 0 for feat in intersect])

    # 2. Get signals specifically for those peak-present bins
    peak_signals = signals[overlap_mask]

    # 3. Define the 'Strong' threshold using the Median (50th percentile) of the peak signals
    strong_threshold = np.median(peak_signals) 

    # 4. Initialize labels
    discrete_labels = np.ones(len(signals), dtype=np.int8)

    # 5. Apply labels using BOTH the overlap mask AND the signal value
    # Label 4: Must have a peak AND signal > threshold
    discrete_labels[overlap_mask & (signals > strong_threshold)] = 4

    # Label 3: Must have a peak AND signal <= threshold
    discrete_labels[overlap_mask & (signals <= strong_threshold)] = 3
    
    background_mask = ~overlap_mask

    ### define threshold for neutral vs negative using the distribuiton of signals in the background bins (those without peaks)

    cutoff = thresholds_500bps.get(factor_type, 1.0)
    # Label 2 (Slightly Negative): Background signal above the specific cutoff
    neutral_mask = background_mask & (signals >= cutoff)
    discrete_labels[neutral_mask] = 2
        
    return experiment_name, signals, discrete_labels

# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    print("Loading master bins and metadata...")
    phenotype_dir = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/phenotype/"
    all_beds = glob.glob(os.path.join(phenotype_dir, "*.bed"))

    for bed_ph in all_beds:
        if bed_ph.endswith("600_clutivar_430.bed") or bed_ph.endswith("600_clutivar_932.bed"):
            continue
        bed_file_name_ph = os.path.basename(bed_ph).split(".bed")[0]


        print(f"Found BED file: {bed_ph}")
        master_bins = load_master_bins(bed_ph)

        df = pd.read_csv(METADATA_PATH)
        #df = meta.parse(sheet_name=meta.sheet_names[0])
        tasks = []
        missing_files_count = 0
        print("Constructing file paths...")
        missing_files = []
        for _, row in df.iterrows():
            proj = str(row['BioProject ID'])
            samp = str(row['Sample accession'])
            bed_name = str(row['Experiment Name']) + ".target.all.bed"
            factor_type = str(row['Factor_type'])
            
            # Build paths according to your structure
            bw_path = os.path.join(BW_FOLDER, proj, f"{samp}.bw")
            bed_path = os.path.join(BED_FOLDER, proj, "hammock", bed_name)
            
            # Validation: Only add to tasks if BOTH files exist
            if os.path.exists(bw_path) and os.path.exists(bed_path):
                tasks.append((bw_path, bed_path, master_bins, samp, factor_type))
            else:
                #check other bw path
                alt_bw_path = os.path.join(BW_FOLDER, proj, f"{samp}.rpgc.bw")
                if os.path.exists(alt_bw_path) and os.path.exists(bed_path):
                    tasks.append((alt_bw_path, bed_path, master_bins, samp, factor_type))
                else:
                    if not os.path.exists(bw_path):
                        print(f"Missing BW: {bw_path}")
                    if not os.path.exists(bed_path):
                        print(f"Missing BED: {bed_path}")
                        missing_files_count += 1
                    missing_files.append(samp)

        print(f"Ready to process {len(tasks)} matched experiments. Missing files: {missing_files_count}")
        with open("missing_files.txt", "w") as f:
            for item in missing_files:
                f.write("%s\n" % item)

        # Run Parallel Pool
        cont_data = {}
        disc_data = {}
        
        with Pool(NUM_CORES) as pool:
            results = list(tqdm(pool.imap(process_experiment, tasks), total=len(tasks)))
        
        print("Assembling DataFrames...")
        for name, sig_arr, disc_arr in results:
            
            cont_data[name] = sig_arr
            disc_data[name] = disc_arr
            
        df_cont = pd.DataFrame(cont_data, index=master_bins['index_id'])
        df_disc = pd.DataFrame(disc_data, index=master_bins['index_id'])
        OUT_CONT_MATRIX = f"{phenotype_dir}/{bed_file_name_ph}.parquet"
        OUT_DISC_MATRIX = f"{phenotype_dir}/{bed_file_name_ph}_disc.parquet"
        print("Saving results..." , OUT_CONT_MATRIX, OUT_DISC_MATRIX)
        df_cont.to_parquet(OUT_CONT_MATRIX)
        df_disc.to_parquet(OUT_DISC_MATRIX)
        
        print(f"Success! Final Matrix size: {df_disc.shape}")
        
        print("Done!")
        print(f"Continuous Matrix Shape: {df_cont.shape}")
        print(f"Discretized Matrix Shape: {df_disc.shape}")