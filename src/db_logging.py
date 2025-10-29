
"""
Database Logging Module for Fruit Ripeness Predictions

This module provides database functionality for storing and retrieving
fruit ripeness predictions. It uses SQLite for persistent storage and
SQLAlchemy ORM for database operations.

Key Features:
- SQLite database: File-based, no server required
- Automatic table creation: Sets up schema on first import
- Prediction logging: Stores label, confidence, metadata, and timestamp
- Statistics queries: Count predictions by fruit/ripeness class
- History retrieval: Get recent predictions for analysis

Database Schema:
Table: predictions
- id: Primary key, auto-increment
- label: Predicted class (e.g., "freshapples", "rottenbanana")
- score: Confidence score (0.0 to 1.0)
- meta: JSON string with additional data (filename, etc.)
- timestamp: When prediction was made (UTC)

Why SQLite:
- No separate database server needed
- File-based storage (predictions.db)
- Perfect for small to medium applications
- Built into Python standard library
- Cross-platform compatible
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ===== Database Configuration =====

# Get project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to SQLite database file
# The database will be created in the project root directory
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

# SQLAlchemy database URL
# Format: sqlite:///path/to/database.db
# The three slashes (///) indicate an absolute path
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ===== SQLAlchemy Setup =====

# Create database engine
# echo=False: Don't print SQL statements to console (set True for debugging)
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
# Sessions are used to interact with the database
# autocommit=False: Transactions must be explicitly committed
# autoflush=False: Changes aren't automatically sent to database
# bind=engine: Connect sessions to our database engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
# All database models will inherit from this
Base = declarative_base()


# ===== Database Model =====

class Prediction(Base):
    """
    SQLAlchemy model for the predictions table.

    This class defines the schema for storing predictions in the database.
    Each instance represents one prediction record.

    Table Structure:
        id: Unique identifier for each prediction (auto-generated)
        label: The predicted fruit/ripeness class
        score: Confidence score from the model (0.0 to 1.0)
        meta: Additional metadata as JSON (e.g., {"filename": "apple.jpg"})
        timestamp: When the prediction was made (automatically set)

    Indexes:
        - Primary key on 'id' for fast lookups
        - Index on 'label' for fast filtering by class
    """
    # Table name in the database
    __tablename__ = "predictions"

    # Primary key column - unique ID for each prediction
    # Auto-increments with each new record
    # index=True creates an index for faster lookups
    id = Column(Integer, primary_key=True, index=True)

    # The predicted class label
    # Examples: "freshapples", "rottenbanana", "unripe orange"
    # nullable=False: This field is required
    # index=True: Allows fast queries filtering by label
    label = Column(String, nullable=False, index=True)

    # Confidence score from the model
    # Range: 0.0 (no confidence) to 1.0 (100% confident)
    # nullable=True: Optional field (some predictions might not have scores)
    score = Column(Float, nullable=True)

    # Metadata as JSON string
    # Can store additional information like filename, user_id, etc.
    # Example: '{"filename": "apple.jpg", "source": "camera"}'
    # nullable=True: Optional field
    meta = Column(String, nullable=True)

    # Timestamp of when prediction was made
    # default=datetime.utcnow: Automatically set to current UTC time
    # UTC is used to avoid timezone issues
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        """
        String representation of a Prediction object.
        Useful for debugging and logging.

        Returns:
            str: Human-readable representation
        """
        return f"<Prediction(id={self.id}, label='{self.label}', score={self.score})>"


# ===== Initialize Database =====

# Create all tables defined in our models
# If tables already exist, this does nothing
# If they don't exist, creates them with the defined schema
Base.metadata.create_all(bind=engine)

# Print confirmation message
print(f"Database initialized at: {DB_PATH}")


# ===== Database Functions =====

def log_prediction(label: str, score: float = None, meta: dict = None):
    """
    Log a prediction to the database.

    This function saves a new prediction record with the given label,
    confidence score, and optional metadata. The timestamp is automatically
    added.

    Args:
        label (str): The predicted class name
            Examples: "freshapples", "rottenbanana", "unripe orange"
            Required field
        score (float, optional): Confidence score from 0.0 to 1.0
            Higher values indicate more confidence
            Default: None
        meta (dict, optional): Additional metadata
            Will be converted to JSON string
            Examples: {"filename": "apple.jpg"}, {"user_id": 123}
            Default: None

    Raises:
        Exception: If database operation fails
            The exception is re-raised after rollback

    Example:
        >>> log_prediction("freshapples", 0.95, {"filename": "apple.jpg"})
        Logged prediction: freshapples (score: 0.95)

    Database Behavior:
        - Creates new session for this operation
        - Adds record to session
        - Commits transaction (saves to disk)
        - Closes session when done
        - Rolls back if error occurs
    """
    # Create a new database session
    # Each session is independent and should be used for one operation
    db = SessionLocal()

    try:
        # Convert metadata dictionary to JSON string if provided
        # JSON format allows storing complex data in a single text field
        # Example: {"filename": "apple.jpg"} -> '{"filename": "apple.jpg"}'
        meta_json = json.dumps(meta) if meta else None

        # Create a new Prediction object
        # The timestamp is automatically set to current UTC time
        prediction = Prediction(
            label=label,
            score=score,
            meta=meta_json
        )

        # Add the prediction to the session
        # This stages the record for insertion
        db.add(prediction)

        # Commit the transaction
        # This actually writes the data to the database file
        db.commit()

        # Print confirmation (useful for debugging)
        print(f"Logged prediction: {label} (score: {score})")

    except Exception as e:
        # If any error occurs, rollback the transaction
        # This undoes any changes made in this session
        db.rollback()

        # Print error message
        print(f"Error logging prediction: {e}")

        # Re-raise the exception so caller knows operation failed
        raise

    finally:
        # Always close the session, even if an error occurred
        # This releases database resources
        db.close()


def counts_by_label():
    """
    Get prediction counts grouped by label.

    This function queries the database to count how many times each
    fruit/ripeness class has been predicted. Results are sorted
    alphabetically by label.

    Returns:
        list: List of tuples (label, count)
            - label (str): The class name
            - count (int): Number of predictions for that class
            Sorted alphabetically by label
            Empty list if no predictions or error occurs

    Example:
        >>> counts = counts_by_label()
        >>> print(counts)
        [
            ('freshapples', 15),
            ('freshbanana', 8),
            ('rottenbanana', 3),
            ('unripe orange', 5)
        ]

    SQL Query Generated:
        SELECT label, COUNT(id) as count
        FROM predictions
        GROUP BY label
        ORDER BY label

    Usage:
        - Statistics endpoint in Flask API
        - Bar chart in Streamlit dashboard
        - Analytics and reporting
    """
    # Create a new database session
    db = SessionLocal()

    try:
        # Build and execute the query
        # This uses SQLAlchemy's query builder syntax
        results = db.query(
            Prediction.label,                      # SELECT label
            func.count(Prediction.id).label("count")  # COUNT(id) AS count
        ).group_by(                                # GROUP BY label
            Prediction.label
        ).order_by(                                # ORDER BY label
            Prediction.label
        ).all()                                     # Execute and fetch all results

        # Convert query results to list of tuples
        # SQLAlchemy returns Row objects, we extract (label, count)
        # Example: Row(label='freshapples', count=15) -> ('freshapples', 15)
        return [(row.label, row.count) for row in results]

    except Exception as e:
        # If query fails, print error and return empty list
        print(f"Error querying counts: {e}")
        return []

    finally:
        # Always close the session
        db.close()


def get_all_predictions(limit: int = 100):
    """
    Get recent predictions from the database.

    This function retrieves the most recent predictions, ordered by
    timestamp (newest first). Useful for viewing prediction history
    or analyzing recent activity.

    Args:
        limit (int, optional): Maximum number of predictions to return
            Default: 100
            Use lower values for better performance
            Use higher values for complete history

    Returns:
        list: List of Prediction objects, newest first
            Each object has: id, label, score, meta, timestamp
            Empty list if no predictions or error occurs

    Example:
        >>> recent = get_all_predictions(limit=10)
        >>> for pred in recent:
        ...     print(f"{pred.timestamp}: {pred.label} ({pred.score})")
        2025-10-27 10:30:45: freshapples (0.95)
        2025-10-27 10:29:12: rottenbanana (0.87)
        ...

    SQL Query Generated:
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT 100

    Usage:
        - Viewing prediction history
        - Debugging and analysis
        - Exporting data for reports
    """
    # Create a new database session
    db = SessionLocal()

    try:
        # Query predictions ordered by timestamp (newest first)
        # limit() restricts the number of results returned
        predictions = db.query(Prediction).order_by(
            Prediction.timestamp.desc()  # ORDER BY timestamp DESC
        ).limit(limit).all()             # LIMIT and fetch all results

        return predictions

    except Exception as e:
        # If query fails, print error and return empty list
        print(f"Error querying predictions: {e}")
        return []

    finally:
        # Always close the session
        db.close()
