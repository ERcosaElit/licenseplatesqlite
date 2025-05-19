import sys
from PyQt5.QtWidgets import QApplication
from gui import VideoLicensePlateGUI

def main():
    """Main entry point for the license plate recognition application"""
    app = QApplication(sys.argv)
    window = VideoLicensePlateGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
