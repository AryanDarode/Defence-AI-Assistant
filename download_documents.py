import os
import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

CSV_PATH = "data/raw/DRDO.csv"
DOWNLOAD_DIR = "data/raw/india/DRDO"


print("============================================")
print("       DRDO PDF DOWNLOADER")
print("============================================")


# ============================================================
# CHECK CSV
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}"
    )


# ============================================================
# CREATE DOWNLOAD DIRECTORY
# ============================================================

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8",
    low_memory=False,
    on_bad_lines="skip"
)

print(f"Documents in CSV: {len(df)}")


# ============================================================
# DOWNLOAD FILES
# ============================================================

success = 0
failed = 0
already_exists = 0


for index, row in df.iterrows():

    url = str(row.get("url", "")).strip()
    local_path = str(row.get("local_path", "")).strip()

    if not url or url == "nan":
        print(f"\n[{index + 1}] No URL found")
        failed += 1
        continue


    # --------------------------------------------------------
    # Get filename
    # --------------------------------------------------------

    if local_path and local_path != "nan":

        filename = os.path.basename(
            local_path.replace("\\", "/")
        )

    else:

        filename = f"drdo_document_{index}.pdf"


    output_path = os.path.join(
        DOWNLOAD_DIR,
        filename
    )


    # --------------------------------------------------------
    # Skip existing
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print(
            f"[{index + 1}/{len(df)}] "
            f"Already exists: {filename}"
        )

        already_exists += 1
        continue


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"Downloading: {filename}"
    )

    try:

        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()


        # ----------------------------------------------------
        # Check content
        # ----------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()


        if (
            "pdf" not in content_type
            and not response.content.startswith(b"%PDF")
        ):

            print(
                "WARNING: Response does not appear "
                "to be a PDF."
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(response.content)


        size_kb = len(response.content) / 1024

        print(
            f"Downloaded successfully "
            f"({size_kb:.1f} KB)"
        )

        success += 1


    except Exception as e:

        print(
            f"FAILED: {filename}"
        )

        print(
            f"Reason: {e}"
        )

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("          DOWNLOAD COMPLETED")
print("============================================")

print(f"Successfully downloaded : {success}")
print(f"Already existed         : {already_exists}")
print(f"Failed                  : {failed}")

print(
    f"\nPDF folder:\n{DOWNLOAD_DIR}"
)

print("============================================")