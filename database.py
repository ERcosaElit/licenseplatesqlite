import sqlite3
import os
import threading
import sys


class VehicleDatabase:

    def __init__(self):
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)  # exe mappája
        else:
            base_path = os.path.abspath(".")

        self.db_path = os.path.join(base_path, "adatbazis2.db")
        self.vehicle_cache = {}  # Gyorsítótár a jármű adatokhoz
        self.lock = threading.Lock()

    def connect(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Hiba az SQLite adatbázishoz való kapcsolódás során: {e}")
            return None

    def disconnect(self):
        pass  # Minden kapcsolat automatikusan lezárul

    def get_vehicle_data(self, license_plate):
        #ármű adatok lekérdezése rendszám alapján
        # Ellenőrizzük a gyorsítótárat
        with self.lock:
            if license_plate in self.vehicle_cache:
                print(f"Találat a gyorsítótárban: {license_plate}")
                return self.vehicle_cache[license_plate]

        conn = self.connect()
        if not conn:
            print("Nem sikerült kapcsolódni az adatbázishoz!")
            return None

        try:
            cursor = conn.cursor()

            # SQLite lekérdezés
            query = """
                SELECT 
                    lplate, 
                    Uzembentartoneve, 
                    Automarka, 
                    Model, 
                    Gyartasidatum, 
                    Szin, 
                    Muszakidatuma, 
                    Moielsoforgalombahelyezesdatuma
                FROM vehicle_data2
                WHERE lplate = ?
            """
            print(f"Lekérdezés végrehajtása: {license_plate}")
            cursor.execute(query, (license_plate,))

            result = cursor.fetchone()

            if result:
                print(f"Találat: {result}")
                # Adatok formázása
                vehicle_data = {
                    'rendszam': result[0],
                    'uzembentarto': result[1],
                    'marka': result[2],
                    'model': result[3],
                    'gyartas_datum': result[4],
                    'szin': result[5],
                    'muszaki_datum': result[6],
                    'forgalomba_helyezes': result[7]
                }

                # Adat mentése a gyorsítótárba
                with self.lock:
                    self.vehicle_cache[license_plate] = vehicle_data
                return vehicle_data
            else:
                print(f"Nincs találat a következő rendszámra: {license_plate}")
                return None

        except sqlite3.Error as e:
            print(f"SQLite hiba a lekérdezés során: {e}")
            return None
        finally:
            conn.close()
