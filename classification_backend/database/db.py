import pymysql
from typing import List, Dict, Any
from config.config import Config

class DatabaseService:
    def __init__(self):
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                autocommit=True
            )
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
    
    def get_cursor(self):
        """Get database cursor with dict result"""
        if not self.connection or not self.connection.open:
            self._connect()
        return self.connection.cursor(pymysql.cursors.DictCursor)
    
    def fetch_plant_conditions(self) -> List[Dict[str, Any]]:
        """Fetch all plant conditions from database"""
        cursor = self.get_cursor()
        try:
            cursor.execute("SELECT * FROM kondisi_tanaman WHERE label != 'Tidak Ada'")
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            print(f"Error fetching plant conditions: {e}")
            return []
        finally:
            cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Global database instance
db_service = DatabaseService()