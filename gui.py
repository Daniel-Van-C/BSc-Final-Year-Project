# gui.py: GUI classes used in YouTube Comments Analyser application.
#
# Menu and MainWindow classes originally created by PyQt5 UI code generator 5.15.6
# Writen and modified by Daniel Van Cuylenburg (k19012373).
#

# Imports.
from PyQt5.QtCore import QAbstractTableModel, QCoreApplication, QRect, QMetaObject, Qt
from PyQt5.QtGui import QPixmap, QMovie, QFont, QFont
from PyQt5.QtWidgets import (QButtonGroup, QLineEdit, QWidget, QPushButton,
                             QComboBox, QMenuBar, QStatusBar, QDesktopWidget,
                             QHeaderView, QLabel, QTableView, QRadioButton,
                             QMenu, QStatusBar, QAction, QGroupBox, QGridLayout,
                             QHBoxLayout, QCheckBox)

from numpy import array, shape
from os import listdir, getcwd, startfile
from os.path import join

# Current directory path.
CURRENT_PATH = join(getcwd(), "Python")


class QLineEdit(QLineEdit):
    """QLineEdit overriding class.
    
    Allows the QLineEdit text prompt to dissapear when clicked.

    Attributes:
        clicked_once (bool): True if the QLineEdit has been clicked
            more than once, False otherwise.
    """

    def __init__(self, parent):
        """Inits QLineEdit class with parent object."""
        super(QLineEdit, self).__init__(parent)
        self.clicked_once = False

    def mousePressEvent(self, event):
        """If clicked for the first time, hide the text prompt.

        Args:
            event (PyQt5.QtGui.QMouseEvent): PyQt5 object storing
                mouse click information. Not used.
        """
        if not self.clicked_once:
            self.setText(QCoreApplication.translate("menu_window", ""))
            self.clicked_once = True


class Menu:
    """Menu GUI class.
    
    Attributes:
        menu_window (PyQt5.QtWidgets.QMainWindow): Menu window object.
        url (str): Text entered into text box.
        spell_check (bool): True if spell check check box is selected, False otherwise.
        option (str): Tracks which button the user clicks. "YouTube" or "CSV".
        file_selection (str): Name of file currently selected in the drop box.
    """

    def __init__(self, window):
        """Inits Menu class."""
        self.menu_window = window
        self.url = ""
        self.spell_check = False
        self.option = None

        self.setup_ui()

    def from_url(self):
        """Sets youtube option & URL variables. Closes Menu window."""
        self.url = self.text_box.text()
        self.option = "YouTube"
        self.menu_window.close()

    def from_file(self):
        """Sets from CSV file option. Closes Menu window."""
        self.option = "CSV"
        self.menu_window.close()

    def set_file_selection(self):
        """Sets file_selection variable as current file selected in combo box."""
        self.file_selection = self.combo_box.currentText()

    def check_box_clicked(self, state):
        """Set spell_check variable based on if the check box is checked.

        Args:
            state (int): 2 if check box is checked, 0 otherwise.
        """
        if state == 2:  # If the check box is ticked.
            self.spell_check = True
        else:
            self.spell_check = False

    def setup_ui(self):
        """Sets up the GUI."""
        self.menu_window.setObjectName("menu_window")
        self.menu_window.resize(685, 206)  # Resize menu window.

        self.central_widget = QWidget(self.menu_window)
        self.central_widget.setObjectName("central_widget")

        # Import from video URL button object.
        self.url_button = QPushButton(self.central_widget)
        self.url_button.setEnabled(True)
        self.url_button.setGeometry(QRect(20, 80, 301, 91))
        font = QFont()  # Font object used to set text size.
        font.setPointSize(16)
        self.url_button.setFont(font)
        self.url_button.setObjectName("url_button")
        # Runs from_url() when clicked.
        self.url_button.clicked.connect(self.from_url)

        # Import from CSV file button object.
        self.csv_button = QPushButton(self.central_widget)
        self.csv_button.setGeometry(QRect(360, 80, 291, 91))
        font.setPointSize(16)
        self.csv_button.setFont(font)
        self.csv_button.setObjectName("csv_button")
        # Runs from_file() when clicked.
        self.csv_button.clicked.connect(self.from_file)

        # URL text box object.
        self.text_box = QLineEdit(self.central_widget)
        self.text_box.setGeometry(QRect(20, 10, 301, 41))
        self.text_box.setClearButtonEnabled(False)
        self.text_box.setObjectName("text_box")
        # Sets font size to 12.
        font.setPointSize(12)
        self.text_box.setFont(font)

        # Get all file names of files found in "Videos" folder.
        file_names = []
        for video_file in listdir(join(CURRENT_PATH, "Videos")):
            file_names.append(video_file)

        # Combo box object.
        self.combo_box = QComboBox(self.central_widget)
        self.combo_box.setGeometry(QRect(340, 10, 331, 41))
        self.combo_box.setObjectName("combo_box")
        # Sets font size to 12.
        font.setPointSize(12)
        self.combo_box.setFont(font)

        # Check box object.
        self.check_box = QCheckBox(self.central_widget)
        self.check_box.setGeometry(QRect(20, 60, 300, 17))
        self.check_box.setObjectName("check_box")
        self.check_box.stateChanged.connect(self.check_box_clicked)
        self.combo_box.addItems(file_names)
        # Runs set_file_selection() when clicked.
        self.combo_box.currentIndexChanged.connect(self.set_file_selection)
        # Sets current file as first file in combo box.
        self.set_file_selection()

        # Menu bar configuration
        self.menu_window.setCentralWidget(self.central_widget)
        self.menu_bar = QMenuBar(self.menu_window)
        self.menu_bar.setGeometry(QRect(0, 0, 685, 21))
        self.menu_bar.setObjectName("menu_bar")
        self.menu_window.setMenuBar(self.menu_bar)
        self.status_bar = QStatusBar(self.menu_window)
        self.status_bar.setObjectName("status_bar")
        self.menu_window.setStatusBar(self.status_bar)

        self.retranslate_ui()
        QMetaObject.connectSlotsByName(self.menu_window)

        # Move window to centre of screen.
        qt_rectangle = self.menu_window.frameGeometry()
        qt_rectangle.moveCenter(QDesktopWidget().availableGeometry().center())
        self.menu_window.move(qt_rectangle.topLeft())

    def retranslate_ui(self):
        """Set text on all objects."""
        translate = QCoreApplication.translate
        self.menu_window.setWindowTitle(translate("menu_window", "Menu"))
        self.url_button.setText(
            translate("menu_window", "Import from Youtube URL"))
        self.csv_button.setText(translate("menu_window",
                                          "Import from CSV file"))
        self.text_box.setText(translate("menu_window", "Enter YouTube URL"))
        self.combo_box.setToolTip(translate("menu_window", "Select CSV File"))
        self.check_box.setText(
            translate(
                "menu_window",
                "Spelling Correction Enabled (May Increase Processing Time)"))


class ListTableModel(QAbstractTableModel):
    """Class to display a list as a table.
    
    Class to display count & TF-IDF vectorizer data in a table.
    Inherits PyQt5.QtCore.QAbstractTableModel.
    
    Attributes:
        data (list): Vectorizer data.
        cols (list): Names of table columns.
    """

    def __init__(self, data):
        """Inits ListTableModel with data."""
        super(ListTableModel, self).__init__()
        self.data = data
        self.cols = ["n-gram", "Score"]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self.data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self.data)

    def columnCount(self, index):
        return len(self.data[0])

    def headerData(self, p_int, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.cols[p_int]
            elif orientation == Qt.Vertical:
                return p_int + 1
        return None


class DataframeTableModel(QAbstractTableModel):
    """Class to populate a table view with a pandas dataframe.
    
    Inherits PyQt5.QtCore.QAbstractTableModel.
    
    This class has been taken and adapted from:
    https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/

    Attributes:
        data (DataFrame): Report or processed comments.
        cols (list): Names of table columns.
        r (int): Number of rows.
        c (int): Number of columns.
    """

    def __init__(self, data, parent=None):
        """Inits DataframeTableModel with data."""
        QAbstractTableModel.__init__(self, parent)
        self.data = array(data.values)
        self.cols = data.columns
        self.r, self.c = shape(self.data)

    def rowCount(self, parent=None):
        return self.r

    def columnCount(self, parent=None):
        return self.c

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return self.data[index.row(), index.column()]
        return None

    def headerData(self, p_int, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.cols[p_int]
            elif orientation == Qt.Vertical:
                return p_int
        return None


class MainWindow:
    """Main Window GUI class.
    
    Attributes:
        comments (pandas DataFrame): Video's comments.
        screen (str): Currently displayed screen. "report" or "comments" or
            "frequency unigram", "importance unigram", "frequency bigram",
            "importance bigram", "frequency trigram", "importance trigram".
        chart (str): Currently displayed chart on frequency and
            importance screens. "bar chart" or "word cloud".
        filter (str): Currently set time range filter.
            "all" or "30" or "7" or "48".
        table (list of lists): Currently selected vectorizer data.
        sort_by (str): Currently selected sort by option on
            list of comments screen. "positive" or "negative" or 
    
        report (pandas DataFrame): Report about comments.
    """

    def __init__(self, window, file_name, report, comments_time_ranges,
                 emotional_analysis, vectorizer_data):
        """Inits MainWindow class."""
        self.comments = None
        self.screen = "report"
        self.chart = "bar chart"
        self.filter = "all"
        self.table = None
        self.sort_by = "positive"

        self.import_data(file_name, report, comments_time_ranges,
                         emotional_analysis, vectorizer_data)

        self.setup_ui(window)

    def refresh_ui(self):
        """Display appropriate GUI elements based on current page."""
        self.hide_all()

        if self.screen == "report":
            self.main_table.setStyleSheet(self.report_table_qss)
            self.main_table.setModel(DataframeTableModel(self.report))
            header = self.main_table.horizontalHeader()
            # Resize headers.
            for i in range(0, len(header)):
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            # Resize row height.
            for i in range(0, len(self.report)):
                self.main_table.setRowHeight(i, 75)
            self.main_table.show()

        elif self.screen == "comments":
            self.sort_by_box.show()
            self.time_box.show()
            self.filter_buttons_controls()
            self.sort_buttons_controls()
            self.main_table.setStyleSheet(self.comments_table_qss)
            self.main_table.setModel(
                DataframeTableModel(self.comments[[
                    "Comment", "Time", "Polarity", "Sentiment", "Subjectivity",
                    "Offensive?", "Emotion"
                ]]))

            # Resize headers.
            header = self.main_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

            # Resize row height.
            for i in range(len(self.comments)):
                self.main_table.setRowHeight(i, 60)
            self.main_table.show()

        elif self.screen in [
                "emotions", "frequency unigram", "importance unigram",
                "frequency bigram", "importance bigram", "frequency trigram",
                "importance trigram"
        ]:
            self.chart_box.show()
            self.time_box.show()
            self.filter_buttons_controls()
            # If there are no n-grams with a frequency larger than 1.
            if self.table and self.table[0][1] == 1 and self.table[-1][1] == 1:
                self.chart_table.setText(
                    "No n-grams with a frequency larger than 1!")
            elif self.table and (self.table[0][1] != 1 or
                                 self.table[-1][1] != 1):
                self.vectorizer_table.setModel(ListTableModel(self.table))

                # Resize headers.
                header = self.vectorizer_table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.Stretch)
                header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

                # Resize row height.
                for i in range(len(self.table)):
                    self.vectorizer_table.setRowHeight(i, 48)
                self.vectorizer_table.show()

                # Display correct chart.
                file_name = self.screen + " " + self.chart + " " + self.filter + ".png"
                self.chart_table.setPixmap(QPixmap(
                    join(CURRENT_PATH, "Videos", self.file_name,
                         "Charts", file_name)))
            else:  # If there is no data or this time range.
                self.chart_table.setText("No data for this time range!")
            self.chart_table.show()

    def hide_all(self):
        """Hide all GUI elements."""
        self.main_table.hide()
        self.vectorizer_table.hide()
        self.chart_table.hide()
        self.sort_by_box.hide()
        self.chart_box.hide()
        self.time_box.hide()

    def show_report_page(self):
        """Set current page to report page."""
        self.screen = "report"
        self.refresh_ui()

    def show_emotional_analysis(self):
        """Set current page to emotional analysis page."""
        self.screen = "emotions"
        self.refresh_ui()

    def show_comments_page(self):
        """Set current page to list of comments page."""
        self.screen = "comments"
        self.refresh_ui()

    def show_unigram_frequency_page(self):
        """Set current page to unigram frequency page."""
        self.screen = "frequency unigram"
        self.refresh_ui()

    def show_unigram_importance_page(self):
        """Set current page to unigram importance page."""
        self.screen = "importance unigram"
        self.refresh_ui()

    def show_bigram_frequency_page(self):
        """Set current page to bigram frequency page."""
        self.screen = "frequency bigram"
        self.refresh_ui()

    def show_bigram_importance_page(self):
        """Set current page to bigram importance page."""
        self.screen = "importance bigram"
        self.refresh_ui()

    def show_trigram_frequency_page(self):
        """Set current page to trigram frequency page."""
        self.screen = "frequency trigram"
        self.refresh_ui()

    def show_trigram_importance_page(self):
        """Set current page to trigram importance page."""
        self.screen = "importance trigram"
        self.refresh_ui()

    def show_word_cloud(self):
        """Set current image to word cloud."""
        self.chart = "word cloud"
        self.refresh_ui()

    def show_bar_chart(self):
        """Set current image to bar chart."""
        self.chart = "bar chart"
        self.refresh_ui()

    def sort_by_negative(self):
        """Set current sort option to most negative."""
        self.sort_by = "negative"
        self.refresh_ui()

    def sort_by_positive(self):
        """Set current sort option to most positive."""
        self.sort_by = "positive"
        self.refresh_ui()

    def sort_by_latest(self):
        """Set current sort option to most recent."""
        self.sort_by = "latest"
        self.refresh_ui()

    def sort_by_oldest(self):
        """Set current sort option to oldest."""
        self.sort_by = "oldest"
        self.refresh_ui()

    def sort_by_most_subjective(self):
        """Set current sort option to most subjective."""
        self.sort_by = "most subjective"
        self.refresh_ui()

    def sort_by_least_subjective(self):
        """Set current sort option to least subjective."""
        self.sort_by = "least subjective"
        self.refresh_ui()

    def sort_buttons_controls(self):
        """Sort comments by the current sort option selected."""
        try:
            if self.sort_by == "negative":
                self.comments = self.comments.sort_values(by="Polarity", ascending=True)
            elif self.sort_by == "positive":
                self.comments = self.comments.sort_values(by="Polarity", ascending=False)
            elif self.sort_by == "latest":
                self.comments = self.comments.sort_values(by="Time", ascending=False)
            elif self.sort_by == "oldest":
                self.comments = self.comments.sort_values(by="Time", ascending=True)
            elif self.sort_by == "most subjective":
                self.comments = self.comments.sort_values(by="Subjectivity", ascending=False)
            elif self.sort_by == "least subjective":
                self.comments = self.comments.sort_values(by="Subjectivity", ascending=True)
        except:
            print("Sort Error.")
            self.comments = self.comments_time_ranges[0]

    def filter_buttons_controls(self):
        """Select the appropriate tables and list of comments to display.
        
        This is based on the currently selected radio buttons.
        """
        pageList = [
            "comments", "emotions", "frequency unigram", "importance unigram",
            "frequency bigram", "importance bigram", "frequency trigram",
            "importance trigram"
        ]
        filterList = ["all", "30", "7", "48"]
        tableList = [[[], [], [], []], self.emotional_analysis,
                     self.frequencyListU, self.importanceListU,
                     self.frequencyListB, self.importanceListB,
                     self.frequencyListT, self.importanceListT]
        for page in range(len(pageList)):
            if self.screen == pageList[page]:
                for filter in range(len(filterList)):
                    if self.filter == filterList[filter]:
                        self.comments = self.comments_time_ranges[filter]
                        self.table = tableList[page][filter]

    def filter_all_time(self):
        """Set current filter to all time."""
        self.filter = "all"
        self.refresh_ui()

    def filter_thirty_days(self):
        """Set current filter to 30 days."""
        self.filter = "30"
        self.refresh_ui()

    def filter_seven_days(self):
        """Set current filter to 7 days."""
        self.filter = "7"
        self.refresh_ui()

    def filter_twenty_four_hours(self):
        """Set current filter to 48 hours."""
        self.filter = "48"
        self.refresh_ui()

    def open_comments(self):
        """Open CSV file of comments."""
        try:
            startfile(join(CURRENT_PATH, "Videos", self.file_name, "processed_all.csv"))
        except:
            print("Could not find Microsoft Excel installation!")

    def connect_menu_bar(self):
        """Connect menu bars actions to respective functions."""
        self.action_report.triggered.connect(self.show_report_page)
        self.action_emotional_analysis.triggered.connect(
            self.show_emotional_analysis)
        self.action_comments.triggered.connect(self.show_comments_page)
        self.action_unigram_frequency.triggered.connect(
            self.show_unigram_frequency_page)
        self.action_unigram_importance.triggered.connect(
            self.show_unigram_importance_page)
        self.action_bigram_frequency.triggered.connect(
            self.show_bigram_frequency_page)
        self.action_bigram_importance.triggered.connect(
            self.show_bigram_importance_page)
        self.action_trigram_frequency.triggered.connect(
            self.show_trigram_frequency_page)
        self.action_trigram_importance.triggered.connect(
            self.show_trigram_importance_page)

    def import_data(self, file_name, report, comments_time_ranges,
                    emotional_analysis, vectorizer_data):
        """Import the data used in the application.

        Args:
            file_name (str): Title of YouTube video (and file name).
            report (DataFrame): Report of average statistics about the comments.
            comments_time_ranges (list of DataFrames): List of subsets of processed comments by time range. 
            emotional_analysis (list of DataFrames): List of emotional analysis subsets by time range.
            vectorizer_data (list of lists of DataFrames): Results of count and TF-IDF vectorization by time range.
        """
        self.file_name = file_name
        self.comments_time_ranges = comments_time_ranges
        self.report = report
        self.emotional_analysis = emotional_analysis
        self.frequencyListU, self.importanceListU, self.frequencyListB, self.importanceListB, self.frequencyListT, self.importanceListT = [
            [], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []
        ]
        list_name = [
            self.frequencyListU, self.importanceListU, self.frequencyListB,
            self.importanceListB, self.frequencyListT, self.importanceListT
        ]

        # Assigns each variable with the correct data n-gram vectorizer data.
        for type in range(len(list_name)):
            for time_range in range(0, 4):
                vectorizer_data[time_range][type] = vectorizer_data[time_range][
                    type].apply(lambda x: round(x, 2))
                for index, value in vectorizer_data[time_range][type].items():
                    list_name[type][time_range].append([index, value])

    def setup_ui(self, main_window):
        """Sets up the GUI.

        Args:
            menu_window (PyQt5.QtWidgets.QMainWindow): Main window object.
        """
        main_window.setObjectName("main_window")
        main_window.resize(1280, 1000)

        # Uncomment to disable window maximization.
        # main_window.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        self.central_widget = QWidget(main_window)
        self.central_widget.setObjectName("central_widget")

        # Stylesheets.
        stylesheets_path = join(CURRENT_PATH, "StyleSheets")
        self.report_table_qss = open(
            join(stylesheets_path, "ReportTable.qss"), "r").read()
        self.comments_table_qss = open(
            join(stylesheets_path, "CommentsTable.qss"), "r").read()
        self.vectorizer_table_qss = open(
            join(stylesheets_path, "VectorizerTable.qss"), "r").read()
        self.group_box_qss = open(
            join(stylesheets_path, "GroupBox.qss"), "r").read()

        # Video title object.
        self.title_label = QLabel(self.central_widget)
        self.title_label.setGeometry(QRect(20, 0, 661, 31))
        font = QFont()  # Font object used to set text properties.
        font.setPointSize(15)
        self.title_label.setFont(font)
        self.title_label.setObjectName("title_label")
        title = self.file_name
        # If the title is longer than 38 characters, shorten it.
        if len(title) > 38:  
            title = title[:38] + "..."
        self.title_label.setText(
            QCoreApplication.translate(
                "main_window", "Video Title: " + title + " Comments: " +
                str(self.comments_time_ranges[0]["Comment"].count().item())))
        self.title_label.show()

        # Tables.
        # Report/Comments table object.
        self.main_table = QTableView(self.central_widget)
        self.main_table.setGeometry(QRect(20, 70, 1241, 881))
        self.main_table.setObjectName("main_table")

        # n-gram frequency/importance table object.
        self.vectorizer_table = QTableView(self.central_widget)
        self.vectorizer_table.setGeometry(QRect(20, 72, 271, 881))
        self.vectorizer_table.setObjectName("vectorizer_table")
        self.vectorizer_table.setStyleSheet(self.vectorizer_table_qss)

        # Bar chart/word cloud image object.
        self.chart_table = QLabel(self.central_widget)
        self.chart_table.setGeometry(QRect(296, 72, 961, 881))
        self.chart_table.setObjectName("chart_table")
        self.chart_table.setScaledContents(True)
        font.setPointSize(30)
        self.chart_table.setFont(font)

        # Button boxes.
        # Sort by button box object.
        self.sort_by_box = QGroupBox(self.central_widget)
        self.sort_by_box.setGeometry(QRect(685, -10, 330, 75))
        self.sort_by_box.setObjectName("sort_by_box")
        self.sort_by_box.setStyleSheet(self.group_box_qss)
        sort_by_widget = QWidget(self.sort_by_box)
        sort_by_widget.setGeometry(QRect(50, 30, 270, 40))
        sort_by_widget.setObjectName("widget")

        # Bar chart/word cloud option button box object.
        self.chart_box = QGroupBox(self.central_widget)
        self.chart_box.setGeometry(QRect(425, 10, 260, 55))
        self.chart_box.setObjectName("chart_box")
        # self.chart_box.setStyleSheet("font-size: 13px;")
        # self.chart_box.setStyleSheet("border: 10px white;")
        self.chart_box.setStyleSheet(self.group_box_qss)
        self.chart_widget = QWidget(self.chart_box)
        self.chart_widget.setGeometry(QRect(65, 30, 191, 15))
        self.chart_widget.setObjectName("horizontalLayoutWidget")

        # Time range filter box object.
        self.time_box = QGroupBox(self.central_widget)
        self.time_box.setGeometry(QRect(10, 10, 390, 55))
        self.time_box.setObjectName("time_box")
        self.time_box.setStyleSheet(self.group_box_qss)
        self.time_widget = QWidget(self.time_box)
        self.time_widget.setGeometry(QRect(65, 30, 320, 15))
        self.time_widget.setObjectName("horizontalLayoutWidget_2")

        # Buttons.
        # Button to open CSV file object.
        self.open_csv_button = QPushButton(self.central_widget)
        self.open_csv_button.setGeometry(QRect(1080, 10, 180, 55))
        self.open_csv_button.setObjectName("open_csv_button")
        self.open_csv_button.clicked.connect(self.open_comments)

        # Word cloud button object.
        self.word_cloud_button = QRadioButton(self.chart_widget)
        self.word_cloud_button.setGeometry(QRect(550, 40, 82, 17))
        self.word_cloud_button.setObjectName("word_cloud_button")
        self.word_cloud_button.clicked.connect(self.show_word_cloud)

        # Bar chart button object.
        self.bar_chart_button = QRadioButton(self.chart_widget)
        self.bar_chart_button.setGeometry(QRect(460, 40, 82, 17))
        self.bar_chart_button.setObjectName("bar_chart_button")
        self.bar_chart_button.clicked.connect(self.show_bar_chart)
        self.bar_chart_button.setChecked(True)

        # Filter by last 7 days button object.
        self.filter_7_button = QRadioButton(self.time_widget)
        self.filter_7_button.setGeometry(QRect(160, 40, 82, 17))
        self.filter_7_button.setObjectName("filter_7_button")
        self.filter_7_button.clicked.connect(self.filter_seven_days)

        # Filter by last 48 hours button object.
        self.filter_48_button = QRadioButton(self.time_widget)
        self.filter_48_button.setGeometry(QRect(220, 40, 82, 17))
        self.filter_48_button.setObjectName("filter_48_button")
        self.filter_48_button.clicked.connect(self.filter_twenty_four_hours)

        # Filter by last 30 days button object.
        self.filter_30_button = QRadioButton(self.time_widget)
        self.filter_30_button.setGeometry(QRect(90, 40, 82, 17))
        self.filter_30_button.setObjectName("filter_30_button")
        self.filter_30_button.clicked.connect(self.filter_thirty_days)

        # Filter by all time button object.
        self.filter_all_button = QRadioButton(self.time_widget)
        self.filter_all_button.setGeometry(QRect(20, 40, 82, 17))
        self.filter_all_button.setObjectName("filter_all_button")
        self.filter_all_button.clicked.connect(self.filter_all_time)
        self.filter_all_button.setChecked(True)

        # Sort by most negative comments button object.
        self.sort_by_negative_button = QRadioButton(sort_by_widget)
        self.sort_by_negative_button.setGeometry(QRect(810, 20, 131, 17))
        self.sort_by_negative_button.setObjectName("sort_by_negative_button")
        self.sort_by_negative_button.clicked.connect(self.sort_by_negative)

        # Sort by most positive comments button object.
        self.sort_by_positive_button = QRadioButton(sort_by_widget)
        self.sort_by_positive_button.setGeometry(QRect(650, 20, 121, 17))
        self.sort_by_positive_button.setObjectName("sort_by_positive_button")
        self.sort_by_positive_button.clicked.connect(self.sort_by_positive)
        self.sort_by_positive_button.setChecked(True)

        # Sort by latest comments button object.
        self.sort_by_latest_button = QRadioButton(sort_by_widget)
        self.sort_by_latest_button.setGeometry(QRect(980, 20, 91, 17))
        self.sort_by_latest_button.setObjectName("sort_by_latest_button")
        self.sort_by_latest_button.clicked.connect(self.sort_by_latest)

        # Sort by oldest comments button object.
        self.sort_by_oldest_button = QRadioButton(sort_by_widget)
        self.sort_by_oldest_button.setGeometry(QRect(980, 40, 91, 17))
        self.sort_by_oldest_button.setObjectName("sort_by_oldest_button")
        self.sort_by_oldest_button.clicked.connect(self.sort_by_oldest)

        # Sort by most subjective comments button object.
        self.sort_by_most_subjective_button = QRadioButton(sort_by_widget)
        self.sort_by_most_subjective_button.setGeometry(QRect(650, 40, 141, 17))
        self.sort_by_most_subjective_button.setObjectName(
            "sort_by_most_subjective")
        self.sort_by_most_subjective_button.clicked.connect(
            self.sort_by_most_subjective)

        # Sort by most objective comments button object.
        self.sort_by_most_objective_button = QRadioButton(sort_by_widget)
        self.sort_by_most_objective_button.setGeometry(QRect(810, 40, 141, 17))
        self.sort_by_most_objective_button.setObjectName(
            "sort_by_least_subjective")
        self.sort_by_most_objective_button.clicked.connect(
            self.sort_by_least_subjective)

        # Layouts.
        # Sort by layout object.
        self.sort_by_layout = QGridLayout(sort_by_widget)
        self.sort_by_layout.setContentsMargins(0, 0, 0, 0)
        self.sort_by_layout.setObjectName("sort_by_layout")
        self.sort_by_layout.addWidget(self.sort_by_positive_button, 0, 0, 1, 1)
        self.sort_by_layout.addWidget(self.sort_by_most_subjective_button, 0, 1,
                                      1, 1)
        self.sort_by_layout.addWidget(self.sort_by_latest_button, 0, 2, 1, 1)
        self.sort_by_layout.addWidget(self.sort_by_negative_button, 1, 0, 1, 1)
        self.sort_by_layout.addWidget(self.sort_by_most_objective_button, 1, 1,
                                      1, 1)
        self.sort_by_layout.addWidget(self.sort_by_oldest_button, 1, 2, 1, 1)

        # Box chart/word cloud layout object.
        self.chart_layout = QHBoxLayout(self.chart_widget)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_layout.setObjectName("chart_layout")
        self.chart_layout.addWidget(self.bar_chart_button)
        self.chart_layout.addWidget(self.word_cloud_button)

        # Time range filter layout object.
        self.time_layout = QHBoxLayout(self.time_widget)
        self.time_layout.setContentsMargins(0, 0, 0, 0)
        self.time_layout.setObjectName("time_layout")
        self.time_layout.addWidget(self.filter_all_button)
        self.time_layout.addWidget(self.filter_30_button)
        self.time_layout.addWidget(self.filter_7_button)
        self.time_layout.addWidget(self.filter_48_button)

        # "Sort By" text object.
        self.sort_by_label = QLabel(self.sort_by_box)
        self.sort_by_label.setGeometry(QRect(10, 20, 61, 51))
        self.sort_by_label.setFont(font)
        self.sort_by_label.setTextFormat(Qt.PlainText)
        self.sort_by_label.setObjectName("sort_by_label")

        # "Chart" text object.
        self.chart_label = QLabel(self.chart_box)
        self.chart_label.setGeometry(QRect(10, 20, 61, 31))
        self.chart_label.setFont(font)
        self.chart_label.setTextFormat(Qt.PlainText)
        self.chart_label.setObjectName("chart_label")

        # "Time Range" text object.
        self.time_label = QLabel(self.time_box)
        self.time_label.setGeometry(QRect(10, 10, 61, 51))
        self.time_label.setFont(font)
        self.time_label.setTextFormat(Qt.PlainText)
        self.time_label.setObjectName("time_label")

        # Menu bar.
        # Menu bar action objects.
        self.action_report = QAction(main_window)
        self.action_report.setObjectName("action_report")
        self.action_emotional_analysis = QAction(main_window)
        self.action_emotional_analysis.setObjectName(
            "action_emotional_analysis")
        self.action_comments = QAction(main_window)
        self.action_comments.setObjectName("action_comments")
        self.action_unigram_frequency = QAction(main_window)
        self.action_unigram_frequency.setObjectName("action_unigram_frequency")
        self.action_unigram_importance = QAction(main_window)
        self.action_unigram_importance.setObjectName(
            "action_unigram_importance")
        self.action_bigram_frequency = QAction(main_window)
        self.action_bigram_frequency.setObjectName("action_bigram_frequency")
        self.action_bigram_importance = QAction(main_window)
        self.action_bigram_importance.setObjectName("action_bigram_importance")
        self.action_trigram_frequency = QAction(main_window)
        self.action_trigram_frequency.setObjectName("action_trigram_frequency")
        self.action_trigram_importance = QAction(main_window)
        self.action_trigram_importance.setObjectName(
            "action_trigram_importance")

        # Menu bar object configuration.
        main_window.setCentralWidget(self.central_widget)
        self.menu_bar = QMenuBar(main_window)
        self.menu_bar.setGeometry(QRect(0, 0, 1280, 21))
        self.menu_bar.setObjectName("menu_bar")
        self.menu_data = QMenu(self.menu_bar)
        self.menu_data.setObjectName("menu_data")
        main_window.setMenuBar(self.menu_bar)
        self.status_bar = QStatusBar(main_window)
        self.status_bar.setObjectName("status_bar")
        main_window.setStatusBar(self.status_bar)

        # Add menu bar actions to menu bar object.
        self.menu_data.addAction(self.action_report)
        self.menu_data.addAction(self.action_emotional_analysis)
        self.menu_data.addAction(self.action_comments)
        self.menu_data.addSeparator()
        self.menu_data.addAction(self.action_unigram_frequency)
        self.menu_data.addAction(self.action_unigram_importance)
        self.menu_data.addSeparator()
        self.menu_data.addAction(self.action_bigram_frequency)
        self.menu_data.addAction(self.action_bigram_importance)
        self.menu_data.addSeparator()
        self.menu_data.addAction(self.action_trigram_frequency)
        self.menu_data.addAction(self.action_trigram_importance)
        self.menu_data.addSeparator()
        self.menu_bar.addAction(self.menu_data.menuAction())
        self.connect_menu_bar()

        self.show_report_page()

        self.retranslate_ui(main_window)
        QMetaObject.connectSlotsByName(main_window)

        # Move window to centre of screen.
        qt_rectangle = main_window.frameGeometry()
        qt_rectangle.moveCenter(QDesktopWidget().availableGeometry().center())
        main_window.move(qt_rectangle.topLeft())

    def retranslate_ui(self, main_window):
        """Set text on all objects."""
        translate = QCoreApplication.translate
        main_window.setWindowTitle(translate("main_window", "Main Window"))
        self.word_cloud_button.setText(translate("main_window", "Word Cloud"))
        self.bar_chart_button.setText(translate("main_window", "Bar Chart"))
        self.sort_by_negative_button.setText(
            translate("main_window", "Most Negative"))
        self.sort_by_positive_button.setText(
            translate("main_window", "Most Positive"))
        self.sort_by_latest_button.setText(translate("main_window", "Latest"))
        self.sort_by_oldest_button.setText(translate("main_window", "Oldest"))
        self.sort_by_most_subjective_button.setText(
            translate("main_window", "Most Subjective"))
        self.sort_by_most_objective_button.setText(
            translate("main_window", "Most Objective"))
        self.open_csv_button.setText(
            translate("main_window", "Open Comments as CSV File"))
        self.filter_7_button.setText(translate("main_window", "7 Days"))
        self.filter_48_button.setText(translate("main_window", "48 Hours"))
        self.filter_30_button.setText(translate("main_window", "30 Days"))
        self.filter_all_button.setText(translate("main_window", "All Time"))
        self.menu_data.setTitle(translate("main_window", "Select Data View"))
        self.action_report.setText(translate("main_window", "Report"))
        self.action_emotional_analysis.setText(
            translate("MainWindow", "Emotional Analysis"))
        self.action_unigram_frequency.setText(
            translate("main_window", "Unigram Frequency"))
        self.action_unigram_importance.setText(
            translate("main_window", "Unigram Importance"))
        self.action_comments.setText(
            translate("main_window", "List of Comments"))
        self.action_bigram_frequency.setText(
            translate("main_window", "Bigram Frequency"))
        self.action_bigram_importance.setText(
            translate("main_window", "Bigram Importance"))
        self.action_trigram_frequency.setText(
            translate("main_window", "Trigram Frequency"))
        self.action_trigram_importance.setText(
            translate("main_window", "Trigram Importance"))
        self.sort_by_label.setText(translate("MainWindow", "Sort\nBy"))
        self.chart_label.setText(translate("MainWindow", "Chart"))
        self.time_label.setText(translate("MainWindow", "Time\nRange"))
