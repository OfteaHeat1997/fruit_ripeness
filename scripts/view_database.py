#!/usr/bin/env python
"""
Database Viewer - View predictions from the SQLite database
"""

import sqlite3
import pandas as pd
from datetime import datetime

def view_database():
    """Display all predictions from the database"""

    # Connect to database
    conn = sqlite3.connect('predictions.db')

    try:
        # Read all predictions into a DataFrame
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)

        print("=" * 80)
        print("PREDICTIONS DATABASE VIEWER")
        print("=" * 80)
        print(f"\nTotal predictions: {len(df)}")
        print("\n" + "-" * 80)

        if len(df) == 0:
            print("No predictions found in database.")
        else:
            # Display all predictions
            for idx, row in df.iterrows():
                print(f"\nPrediction #{row['id']}")
                print(f"  Label:      {row['label']}")
                print(f"  Confidence: {row['score']:.2%}")
                print(f"  Metadata:   {row['meta']}")
                print(f"  Timestamp:  {row['timestamp']}")
                print("-" * 80)

        # Statistics by label
        print("\n" + "=" * 80)
        print("STATISTICS BY LABEL")
        print("=" * 80)

        stats = df.groupby('label').agg({
            'id': 'count',
            'score': ['mean', 'min', 'max']
        }).round(4)

        if not stats.empty:
            print("\nLabel Statistics:")
            print(stats)

        # Export to CSV option
        print("\n" + "=" * 80)
        export = input("\nExport to CSV? (y/n): ").lower()
        if export == 'y':
            filename = f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"Exported to: {filename}")

    finally:
        conn.close()

def view_recent(limit=10):
    """Display recent predictions"""
    conn = sqlite3.connect('predictions.db')

    try:
        df = pd.read_sql_query(
            f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )

        print(f"\n{len(df)} Most Recent Predictions:")
        print(df.to_string())

    finally:
        conn.close()

def search_by_label(label):
    """Search predictions by label"""
    conn = sqlite3.connect('predictions.db')

    try:
        df = pd.read_sql_query(
            f"SELECT * FROM predictions WHERE label LIKE '%{label}%' ORDER BY timestamp DESC",
            conn
        )

        print(f"\nPredictions containing '{label}':")
        print(df.to_string())

    finally:
        conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--recent':
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            view_recent(limit)
        elif sys.argv[1] == '--search':
            if len(sys.argv) > 2:
                search_by_label(sys.argv[2])
            else:
                print("Usage: python view_database.py --search <label>")
        else:
            print("Usage:")
            print("  python view_database.py              # View all predictions")
            print("  python view_database.py --recent 10  # View 10 recent predictions")
            print("  python view_database.py --search banana  # Search for 'banana'")
    else:
        view_database()
