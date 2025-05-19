import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QApplication)
from PyQt5.QtCore import Qt
from database import VehicleDatabase


class ManualLicensePlateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vehicle_db = VehicleDatabase()
        self.setup_ui()

    def open_manual_dialog(self):
        """Kézi rendszám beviteli ablak megnyitása"""
        from manual import ManualLicensePlateDialog  # Relatív importálás a jelenlegi mappából
        dialog = ManualLicensePlateDialog(self)
        dialog.exec_()

    def setup_ui(self):
        """Felhasználói felület beállítása"""
        self.setWindowTitle("Rendszám Kézi Bevitele")
        self.setGeometry(300, 300, 600, 400)

        # Fő elrendezés
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Beviteli terület
        input_layout = QHBoxLayout()

        # Rendszám beviteli mező
        self.license_plate_label = QLabel("Rendszám:")
        input_layout.addWidget(self.license_plate_label)

        self.license_plate_input = QLineEdit()
        self.license_plate_input.setPlaceholderText("Írja be a rendszámot (pl. ABC-123)")
        self.license_plate_input.setMinimumWidth(200)
        input_layout.addWidget(self.license_plate_input)

        # OK gomb
        self.ok_button = QPushButton("Keresés")
        self.ok_button.clicked.connect(self.search_license_plate)
        input_layout.addWidget(self.ok_button)

        main_layout.addLayout(input_layout)

        # Eredmény megjelenítése
        self.result_label = QLabel("Adja meg a rendszámot a kereséshez")
        main_layout.addWidget(self.result_label)

        # Táblázat az adatok megjelenítéséhez
        self.data_table = QTableWidget(0, 2)  # 0 sor, 2 oszlop
        self.data_table.setHorizontalHeaderLabels(["Tulajdonság", "Érték"])
        self.data_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        main_layout.addWidget(self.data_table)

        # Bezárás gomb
        self.close_button = QPushButton("Bezárás")
        self.close_button.clicked.connect(self.close)
        main_layout.addWidget(self.close_button)

    def search_license_plate(self):
        """Rendszám keresése az adatbázisban"""
        license_plate = self.license_plate_input.text().strip()
        if not license_plate:
            QMessageBox.warning(self, "Figyelmeztetés", "Kérem adjon meg egy rendszámot!")
            return

        # Adatbázis lekérdezés
        print(f"Rendszám keresése: {license_plate}")
        vehicle_data = self.vehicle_db.get_vehicle_data(license_plate)

        if vehicle_data:
            print(f"Találat: {vehicle_data}")
            self.result_label.setText(f"Rendszám találat: {license_plate}")
            self.display_vehicle_data(vehicle_data)
        else:
            print(f"Nincs találat a következő rendszámra: {license_plate}")
            self.result_label.setText(f"Nincs találat a következő rendszámra: {license_plate}")
            self.data_table.setRowCount(0)  # Táblázat ürítése
            QMessageBox.information(self, "Nincs találat",
                                    f"A(z) {license_plate} rendszámú jármű nem található az adatbázisban.")

    def display_vehicle_data(self, vehicle_data):
        """Járműadatok megjelenítése a táblázatban"""
        # Táblázat ürítése
        self.data_table.setRowCount(0)

        # Adatok hozzáadása
        properties = [
            ("Rendszám", "rendszam"),
            ("Üzembentartó", "uzembentarto"),
            ("Márka", "marka"),
            ("Modell", "model"),
            ("Gyártási dátum", "gyartas_datum"),
            ("Szín", "szin"),
            ("Műszaki érvényesség", "muszaki_datum"),
            ("Forgalomba helyezés", "forgalomba_helyezes")
        ]

        for i, (display_name, key) in enumerate(properties):
            value = vehicle_data.get(key, "")
            print(f"{display_name}: {value}")  # Konzolra kiírás

            self.data_table.insertRow(i)
            self.data_table.setItem(i, 0, QTableWidgetItem(display_name))
            self.data_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def closeEvent(self, event):
        """Ablak bezárásakor megszakítjuk az adatbázis kapcsolatot"""
        self.vehicle_db.disconnect()
        event.accept()


# Önálló teszteléshez
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ManualLicensePlateDialog()
    dialog.show()
    sys.exit(app.exec_())
