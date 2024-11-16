import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle("PyQt5 Test Window")

        # Set the window dimensions
        self.setGeometry(100, 100, 400, 300)

        # Create a button widget
        self.button = QPushButton("Click Me", self)
        self.button.setGeometry(150, 120, 100, 40)

        # Connect the button click event to the function
        self.button.clicked.connect(self.show_message)

    def show_message(self):
        # Display a message box when the button is clicked
        QMessageBox.information(self, "Message", "Button Clicked!")


if __name__ == "__main__":
    # Create the application object
    app = QApplication(sys.argv)

    # Create an instance of MyApp
    window = MyApp()

    # Show the window
    window.show()

    # Execute the application's event loop
    sys.exit(app.exec_())
