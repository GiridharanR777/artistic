"""
WikiArt on-demand dataset downloader and cache manager.
Uses kagglehub to download targeted artistic styles without downloading the entire 80,000-image archive.
"""

import os
import zipfile
import pandas as pd
import kagglehub
from typing import List, Dict


def get_wikiart_manifest() -> pd.DataFrame:
    """Loads and extracts the classes.csv metadata manifest from the kagglehub cache."""
    cache_base = os.path.join(
        os.path.expanduser("~"),
        ".cache", "kagglehub", "datasets", "steubk", "wikiart", "versions", "1"
    )
    manifest_path = os.path.join(cache_base, "classes.csv")

    # If not yet downloaded, download classes.csv
    if not os.path.exists(manifest_path):
        manifest_path = kagglehub.dataset_download("steubk/wikiart", path="classes.csv")

    if zipfile.is_zipfile(manifest_path):
        with zipfile.ZipFile(manifest_path) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(manifest_path)

    df["style_category"] = df["filename"].apply(lambda x: str(x).split("/")[0])
    return df


def get_available_styles() -> Dict[str, int]:
    """Returns a dictionary mapping available style categories to their total image counts."""
    df = get_wikiart_manifest()
    return df["style_category"].value_counts().to_dict()


def download_style_subset(
    selected_styles: List[str],
    samples_per_category: int = 50,
    output_dir: str = "c:/art/wikiart_cache"
) -> Dict[str, List[str]]:
    """
    Downloads a curated subset of images for specified style categories.
    Skips files that are already cached locally.
    Returns a dict mapping style category name to list of local image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = get_wikiart_manifest()
    downloaded_files = {}

    for style in selected_styles:
        style_dir = os.path.join(output_dir, style)
        os.makedirs(style_dir, exist_ok=True)
        style_df = df[df["style_category"] == style]

        if style_df.empty:
            print(f"[Warning] Style '{style}' not found in WikiArt manifest.")
            continue

        selected_records = style_df.head(samples_per_category)
        file_paths = []

        print(f"[WikiArt] Preparing style '{style}' ({len(selected_records)} images)...")
        for _, row in selected_records.iterrows():
            rel_path = row["filename"]  # e.g., 'Impressionism/claude-monet_water-lilies.jpg'
            file_name = os.path.basename(rel_path)
            target_local = os.path.join(style_dir, file_name)

            if os.path.exists(target_local) and os.path.getsize(target_local) > 0:
                file_paths.append(target_local)
                continue

            try:
                # Targeted download of single file via kagglehub
                downloaded_loc = kagglehub.dataset_download("steubk/wikiart", path=rel_path)
                # If downloaded to kagglehub cache, copy or record
                if os.path.exists(downloaded_loc):
                    # Copy to local project cache directory
                    import shutil
                    shutil.copy2(downloaded_loc, target_local)
                    file_paths.append(target_local)
            except Exception as e:
                print(f"[Warning] Failed to download {rel_path}: {e}")

        downloaded_files[style] = file_paths
        print(f"[WikiArt] '{style}': {len(file_paths)} images ready.")

    return downloaded_files


if __name__ == "__main__":
    styles = get_available_styles()
    print(f"Total available styles: {len(styles)}")
    print("Top 5 styles:", list(styles.items())[:5])
