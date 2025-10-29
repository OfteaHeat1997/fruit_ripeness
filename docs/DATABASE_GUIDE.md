# Database Setup Guide

## Overview

This guide shows you how to view and analyze the predictions database (`predictions.db`) using different tools.

## Database Structure

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    label VARCHAR NOT NULL,
    score FLOAT,
    meta VARCHAR,
    timestamp DATETIME
)
```

---

## Method 1: Using Python Script (Easiest)

### View All Predictions:
```bash
python scripts/view_database.py
```

### View Recent Predictions:
```bash
python scripts/view_database.py --recent 10
```

### Search by Label:
```bash
python scripts/view_database.py --search banana
```

---

## Method 2: PyCharm Database Tool

### Step 1: Open Database Tab
1. In PyCharm, click **View** → **Tool Windows** → **Database**
2. Or press **Alt+1** (Windows/Linux) or **⌘+1** (Mac)

### Step 2: Add Data Source
1. Click **+** (New) button
2. Select **Data Source** → **SQLite**
3. In the dialog:
   - **File**: Browse to `predictions.db` in your project
   - Click **Test Connection**
   - Click **OK**

### Step 3: View Data
1. Expand the database in the Database tab
2. Expand **schemas** → **main** → **tables**
3. Right-click **predictions** → **Jump to Console**
4. Run queries like:
   ```sql
   SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;
   ```

---

## Method 3: SQLite Command Line

```bash
sqlite3 predictions.db
```

Then run SQL commands:
```sql
-- View all predictions
SELECT * FROM predictions;

-- View recent predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;

-- Count by label
SELECT label, COUNT(*) as count FROM predictions GROUP BY label;

-- Average confidence by label
SELECT label, AVG(score) as avg_confidence FROM predictions GROUP BY label;
```

To exit: `.quit`

---

## Method 4: Python Code

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('predictions.db')

# Read into pandas DataFrame
df = pd.read_sql_query("SELECT * FROM predictions", conn)

# View data
print(df.head())

# Statistics
print(df['label'].value_counts())
print(df.groupby('label')['score'].mean())

# Close connection
conn.close()
```

---

## Backup Database

```bash
# Create backup
cp predictions.db predictions_backup_$(date +%Y%m%d_%H%M%S).db

# Or using sqlite3
sqlite3 predictions.db ".backup predictions_backup.db"
```

---

## Export to CSV

Using the Python script:
```bash
python scripts/view_database.py
# Then press 'y' when prompted to export
```

Or using SQL:
```bash
sqlite3 -header -csv predictions.db "SELECT * FROM predictions;" > predictions.csv
```
