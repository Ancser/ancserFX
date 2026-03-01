#!/usr/bin/env bash
#
# ancserFX - Unzip Test Data (macOS / Linux)
#

set -e

echo ""
echo "  ============================================="
echo "   ancserFX - Unzip Test Data"
echo "  ============================================="
echo ""

# Check test-data.zip exists
if [ ! -f "test-data.zip" ]; then
    echo "  [ERROR] test-data.zip not found."
    echo "          Make sure you are in the ancserFX root directory."
    exit 1
fi

# Check if already extracted
if [ -f "data/parquet/es/5min/data.parquet" ]; then
    echo "  [SKIP] data/parquet/ already has data."
    echo "         Delete data/parquet/ first if you want to re-extract."
    exit 0
fi

# Unzip using Python
echo "  Extracting test-data.zip to data/parquet/ ..."
python3 -c "
import zipfile, pathlib
pathlib.Path('data/parquet').mkdir(parents=True, exist_ok=True)
zipfile.ZipFile('test-data.zip').extractall('data/parquet')
print('  [OK] Done.')
"

echo ""
echo "  ============================================="
echo "   Data ready! Start the dashboard:"
echo "     streamlit run dashboard.py"
echo "  ============================================="
echo ""
