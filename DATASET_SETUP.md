# GenImage Dataset Setup Instructions for Collaborators

This guide helps team members set up and download the GenImage dataset for this project.

## Prerequisites

### Required Software

1. **rclone** - for downloading from Google Drive
   ```bash
   # Ubuntu/Debian
   sudo apt install rclone
   
   # macOS
   brew install rclone
   ```

2. **p7zip** - for extracting split archives
   ```bash
   # Ubuntu/Debian
   sudo apt install p7zip-full
   
   # macOS
   brew install p7zip
   ```

## Setup Steps

### 1. Add GenImage Folder to Your Google Drive

Before downloading, you need to add the shared folder as a shortcut to your Drive:

1. Visit the shared folder: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
2. **Right-click** on the folder → Select **"Add shortcut to Drive"**
3. Choose **"My Drive"** as the location
4. Click **"Add"**

### 2. Configure Rclone

Run the rclone configuration wizard:

```bash
rclone config
```

Follow these steps in the wizard:

1. Type `n` for **new remote**
2. Name: `mohamed` (must match the script configuration)
3. Storage type: Enter the number for **"Google Drive"** (usually `drive`)
4. Client ID: (leave blank, press Enter)
5. Client Secret: (leave blank, press Enter)
6. Scope: Enter `1` for **"Full access"**
7. Root folder ID: (leave blank, press Enter)
8. Service Account File: (leave blank, press Enter)
9. Advanced config: Enter `n` for **No**
10. Auto config: Enter `y` for **Yes** (this will open a browser)
11. **Authenticate** in the browser window that opens
12. After successful authentication, select `q` to **quit**

### 3. Run the Setup Script

Download and extract the dataset:

```bash
./setup_dataset.sh
```

This script will:
- Download the configured folders (ADM, BigGAN, glide, Midjourney by default)
- Extract all split zip archives
- Organize data into the `data/` directory
- Clean up archive files to save disk space

### 4. (Optional) Customize Folders

To download different folders, edit `setup_dataset.sh` and modify the `DEFAULT_FOLDERS` array:

```bash
DEFAULT_FOLDERS=(
    "ADM"
    "BigGAN"
    "glide"
    "Midjourney"
    # Uncomment any of these to download them:
    # "VQDM"
    # "stable_diffusion_v_1_4"
    # "stable_diffusion_v_1_5"
    # "wukong"
)
```

Or use the command line:

```bash
./setup_dataset.sh --folders ADM,BigGAN,glide
```

## Expected Directory Structure

After setup completes, you'll have:

```
project/
├── data/                    # Ready-to-use dataset
│   ├── ADM/
│   ├── BigGAN/
│   ├── glide/
│   └── Midjourney/
├── downloaded_dataset/      # Temporary (can be deleted)
│   └── ...
├── setup_dataset.sh         # Setup script
└── DATASET_SETUP.md        # This file
```

## Disk Space

- The complete dataset is **~608 GB**
- Individual folders vary from ~20 GB to ~150 GB each
- Archives are deleted after extraction to save space
- You can safely delete `downloaded_dataset/` folder after setup completes

## Troubleshooting

### "Remote 'mohamed' not found"
- Run `rclone config` and set up the remote (see Step 2)

### "Folder 'genimage' not found in your Google Drive"
- Make sure you added the shortcut to your Drive (see Step 1)
- Check that the shortcut is in "My Drive", not in "Shared with me"

### "No extraction tool found"
- Install p7zip: `sudo apt install p7zip-full` (Ubuntu) or `brew install p7zip` (macOS)

### Download interrupted
- Just re-run `./setup_dataset.sh` - rclone will resume and skip already downloaded files

## Support

If you encounter issues, check:
1. You have enough disk space (~100-200 GB depending on folders)
2. Your internet connection is stable
3. You've authenticated rclone with the correct Google account
4. The genimage folder is accessible in your Google Drive
