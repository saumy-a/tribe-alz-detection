# OASIS-3 Download Scripts

Shell scripts for downloading OASIS-3 data from [NITRC Image Repository](https://nitrc.org/ir).  
Sourced from the official [NrgXnat/oasis-scripts](https://github.com/NrgXnat/oasis-scripts) repository.

## Prerequisites

1. Register at [nitrc.org/ir](https://nitrc.org/ir)
2. Join the **OASIS3** project and sign the Data Use Agreement (DUA)
3. Wait 24–48 h for access to propagate

## Usage

### Download BOLD scans (BIDS format)

```bash
# Fix Windows line endings first if needed
tr -d '\r' < sessions.csv > sessions_unix.csv

bash download_oasis_scans_bids.sh \
    sessions_unix.csv \
    ../../data/scans/ \
    YOUR_NITRC_USERNAME \
    bold
```

### Download T1w scans

```bash
bash download_oasis_scans_bids.sh \
    sessions_unix.csv \
    ../../data/scans/ \
    YOUR_NITRC_USERNAME \
    T1w
```

### Download FreeSurfer outputs

```bash
bash download_oasis_freesurfer.sh \
    freesurfer_ids.csv \
    ../../data/freesurfer/ \
    YOUR_NITRC_USERNAME
```

## Files

| File | Purpose |
|------|---------|
| `download_oasis_scans_bids.sh` | Download BOLD/T1w in BIDS format |
| `download_oasis_scans.sh` | Download scans (non-BIDS) |
| `download_oasis_freesurfer.sh` | Download FreeSurfer outputs |
| `download_oasis_pup.sh` | Download PET Unified Pipeline outputs |
| `filter_clinical.py` | Filter clinical CSVs to build session ID lists |
| `oasis_scan_download_example.csv` | Example session list format |
| `fs_download_example.csv` | Example FreeSurfer ID list format |

## Notes

- BIDS format uses spec v1.0.1; BOLD task name = `rest`
- CSV files must use Unix line endings (`\n`, not `\r\n`)
- Downloaded data goes in `data/scans/` — **never commit this directory**
