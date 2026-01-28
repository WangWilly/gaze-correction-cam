"""
User Settings Database Module

Provides SQLite database functionality for storing and retrieving user settings.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class UserSettingsDB:
    """Manages user settings in SQLite database."""
    
    def __init__(self, db_path: str = "./user_settings.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Create database table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS UserSetting (
                    name TEXT PRIMARY KEY,
                    cfg TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def get_setting(self, name: str) -> Optional[dict]:
        """
        Retrieve a setting by name.
        
        Args:
            name: Setting name
            
        Returns:
            Setting configuration as dictionary, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cfg FROM UserSetting WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
    
    def save_setting(self, name: str, cfg: dict):
        """
        Save or update a setting.
        
        Args:
            name: Setting name
            cfg: Setting configuration as dictionary
        """
        now = datetime.now().isoformat()
        cfg_json = json.dumps(cfg)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if setting exists
            cursor.execute(
                "SELECT created_at FROM UserSetting WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update existing
                cursor.execute(
                    "UPDATE UserSetting SET cfg = ?, updated_at = ? WHERE name = ?",
                    (cfg_json, now, name)
                )
            else:
                # Insert new
                cursor.execute(
                    "INSERT INTO UserSetting (name, cfg, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (name, cfg_json, now, now)
                )
            conn.commit()
    
    def delete_setting(self, name: str) -> bool:
        """
        Delete a setting.
        
        Args:
            name: Setting name
            
        Returns:
            True if setting was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM UserSetting WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
    
    def list_settings(self) -> list[str]:
        """
        List all setting names.
        
        Returns:
            List of setting names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM UserSetting ORDER BY name")
            return [row[0] for row in cursor.fetchall()]
