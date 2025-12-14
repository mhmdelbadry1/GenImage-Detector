#!/bin/bash

################################################################################
# GenImage Dataset Download and Setup Script
# 
# This script downloads specific folders from the GenImage dataset, extracts
# split zip archives, and organizes the data for use in the project.
#
# Usage: ./setup_dataset.sh [--folders folder1,folder2,...]
################################################################################

set -e  # Exit on error

#------------------------------------------------------------------------------
# CONFIGURATION - Edit this section to customize
#------------------------------------------------------------------------------

# Default folders to download (edit this list as needed)
DEFAULT_FOLDERS=(
    "ADM"
    "BigGAN"
    # "glide"
    "Midjourney"
    # "VQDM"
    # "stable_diffusion_v_1_4"
    "stable_diffusion_v_1_5"
    # "wukong"
)

# Rclone remote name (must be configured first)
REMOTE_NAME="mohamed"

# Source folder in Google Drive
DRIVE_FOLDER="genimage"

# Temporary download directory
TEMP_DIR="downloaded_dataset"

# Final data directory
DATA_DIR="data"

#------------------------------------------------------------------------------
# Parse command line arguments
#------------------------------------------------------------------------------

FOLDERS=("${DEFAULT_FOLDERS[@]}")

while [[ $# -gt 0 ]]; do
    case $1 in
        --folders)
            IFS=',' read -ra FOLDERS <<< "$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--folders folder1,folder2,...]"
            echo ""
            echo "Downloads and extracts specified folders from the GenImage dataset."
            echo ""
            echo "Options:"
            echo "  --folders    Comma-separated list of folders to download"
            echo "               (default: ADM,BigGAN,glide,Midjourney)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    local missing_deps=()
    
    # Check rclone
    if ! command -v rclone &> /dev/null; then
        missing_deps+=("rclone")
    fi
    
    # Check extraction tools
    if ! command -v 7z &> /dev/null && ! command -v 7za &> /dev/null; then
        missing_deps+=("p7zip-full or p7zip")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "ERROR: Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Please install missing dependencies:"
        echo "  Ubuntu/Debian: sudo apt install rclone p7zip-full"
        echo "  macOS: brew install rclone p7zip"
        exit 1
    fi
    
    echo "✓ All dependencies found"
}

check_rclone_config() {
    print_step "Checking rclone configuration..."
    
    if ! rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
        echo "ERROR: Rclone remote '${REMOTE_NAME}' not found."
        echo ""
        echo "Please configure rclone first:"
        echo "  1. Run: rclone config"
        echo "  2. Choose 'n' for new remote"
        echo "  3. Name it: ${REMOTE_NAME}"
        echo "  4. Choose 'drive' for Google Drive"
        echo "  5. Follow the authentication prompts"
        echo ""
        echo "See DATASET_SETUP.md for detailed instructions."
        exit 1
    fi
    
    echo "✓ Rclone remote '${REMOTE_NAME}' configured"
}

check_drive_folder() {
    print_step "Checking for GenImage folder in your Drive..."
    
    if ! rclone lsf "${REMOTE_NAME}:" --max-depth 1 --dirs-only | grep -q "^${DRIVE_FOLDER}/"; then
        echo "ERROR: Folder '${DRIVE_FOLDER}' not found in your Google Drive."
        echo ""
        echo "Please add the shared folder to your Drive:"
        echo "  1. Visit: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
        echo "  2. Right-click the folder → 'Add shortcut to Drive'"
        echo "  3. Choose 'My Drive' as the location"
        echo ""
        exit 1
    fi
    
    echo "✓ GenImage folder found"
}

download_folder() {
    local folder=$1
    local source="${REMOTE_NAME}:${DRIVE_FOLDER}/${folder}"
    local dest="${TEMP_DIR}/${folder}"
    
    echo "Downloading: $folder"
    echo "  From: $source"
    echo "  To: $dest"
    echo ""
    
    mkdir -p "$dest"
    
    rclone copy "$source" "$dest" \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --drive-chunk-size 64M \
        --stats-one-line \
        -v
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Download complete: $folder"
        return 0
    else
        echo ""
        echo "✗ Download failed: $folder"
        return 1
    fi
}

extract_archives() {
    local folder=$1
    local source_dir="${TEMP_DIR}/${folder}"
    local dest_dir="${DATA_DIR}/${folder}"
    
    print_step "Extracting archives in: $folder"
    
    mkdir -p "$dest_dir"
    
    # Find all .zip files (these are the main archives for split zips)
    local zip_files=($(find "$source_dir" -name "*.zip" -type f))
    
    if [ ${#zip_files[@]} -eq 0 ]; then
        echo "No zip files found in $folder"
        return 0
    fi
    
    for zip_file in "${zip_files[@]}"; do
        local basename=$(basename "$zip_file")
        echo "Extracting: $basename"
        
        # Use 7z to extract (handles split archives automatically)
        if command -v 7z &> /dev/null; then
            7z x "$zip_file" -o"$dest_dir" -y
        elif command -v 7za &> /dev/null; then
            7za x "$zip_file" -o"$dest_dir" -y
        else
            echo "ERROR: No extraction tool found (7z/7za)"
            return 1
        fi
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Extracted: $basename"
        else
            echo "  ✗ Failed to extract: $basename"
            return 1
        fi
    done
    
    echo ""
    echo "✓ Extraction complete: $folder"
    return 0
}

cleanup_archives() {
    local folder=$1
    local source_dir="${TEMP_DIR}/${folder}"
    
    print_step "Cleaning up archives in: $folder"
    
    # Calculate space before cleanup
    local size_before=$(du -sh "$source_dir" 2>/dev/null | cut -f1)
    
    # Remove all archive files
    find "$source_dir" -type f \( -name "*.zip" -o -name "*.z[0-9][0-9]" \) -delete
    
    # Calculate space after cleanup
    local size_after=$(du -sh "$source_dir" 2>/dev/null | cut -f1)
    
    echo "Space in $folder: $size_before → $size_after"
    echo "✓ Cleanup complete: $folder"
}

#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------

print_header "GenImage Dataset Setup"

echo "Folders to download:"
for folder in "${FOLDERS[@]}"; do
    echo "  - $folder"
done

# Pre-flight checks
check_dependencies
check_rclone_config
check_drive_folder

# Create directories
mkdir -p "$TEMP_DIR"
mkdir -p "$DATA_DIR"

# Process each folder
SUCCESSFUL=()
FAILED=()

for folder in "${FOLDERS[@]}"; do
    print_header "Processing: $folder"
    
    # Download
    if download_folder "$folder"; then
        # Extract
        if extract_archives "$folder"; then
            # Cleanup
            cleanup_archives "$folder"
            SUCCESSFUL+=("$folder")
        else
            echo "ERROR: Extraction failed for $folder"
            FAILED+=("$folder (extraction)")
        fi
    else
        echo "ERROR: Download failed for $folder"
        FAILED+=("$folder (download)")
    fi
done

# Final summary
print_header "Setup Complete"

echo "Successful: ${#SUCCESSFUL[@]}"
for folder in "${SUCCESSFUL[@]}"; do
    echo "  ✓ $folder"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}"
    for folder in "${FAILED[@]}"; do
        echo "  ✗ $folder"
    done
fi

echo ""
echo "Data directory: $DATA_DIR"
echo "Temp directory: $TEMP_DIR (you can delete this to save space)"
echo ""
echo "To completely remove downloaded archives:"
echo "  rm -rf $TEMP_DIR"
